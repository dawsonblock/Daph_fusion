"""Curvature bank for AGX (Phase 6).

Caches empirical Fisher diagonals per expert × parameter so that AGX
candidates using FISHER, TIES_FISHER, or DARE_TIES_FISHER can access
curvature without recomputing in the search inner loop.

Usage:
    bank = CurvatureBank.build(base_model, experts, calibration_data, ...)
    fisher_for_expert_0 = bank.get_fisher("expert_0", "transformer.h.0.attn.c_attn.weight")
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class CurvatureBank:
    """Cached curvature data for all experts."""
    # expert_name -> param_name -> Fisher diagonal tensor
    fisher: Dict[str, Dict[str, Tensor]] = field(default_factory=dict)
    # Provenance metadata
    base_model_hash: str = ""
    calibration_data_hash: str = ""
    n_samples: int = 0
    mode: str = "exact_per_sample"
    dtype: str = "float32"

    def get_fisher(self, expert_name: str, param_name: str) -> Optional[Tensor]:
        """Get the Fisher diagonal for a specific expert and parameter."""
        return self.fisher.get(expert_name, {}).get(param_name)

    def get_fisher_list(self, param_name: str, expert_names: List[str]) -> List[Tensor]:
        """Get Fisher diagonals for all experts for a parameter."""
        result = []
        for name in expert_names:
            f = self.get_fisher(name, param_name)
            if f is not None:
                result.append(f.float())
            else:
                result.append(None)
        return result

    def to_metadata_dict(self) -> dict:
        return {
            "base_model_hash": self.base_model_hash,
            "calibration_data_hash": self.calibration_data_hash,
            "n_samples": self.n_samples,
            "mode": self.mode,
            "dtype": self.dtype,
            "experts": list(self.fisher.keys()),
            "param_counts": {
                name: len(params) for name, params in self.fisher.items()
            },
        }

    @classmethod
    def build(
        cls,
        base_model: nn.Module,
        experts: Sequence[nn.Module],
        calibration_texts: Sequence[str],
        tokenizer: Any,
        device: str = "cpu",
        max_length: int = 128,
        max_samples: int = 50,
        expert_names: Optional[List[str]] = None,
    ) -> "CurvatureBank":
        """Build a curvature bank by computing empirical Fisher for each expert.

        Uses exact per-sample gradients: for each calibration sample, compute
        the causal LM loss, backpropagate, and accumulate squared gradients.

        F_{i,k} = (1/N) Σ_n (∂L_n/∂θ_{i,k})^2

        This is the TRUE empirical Fisher, not |delta|^2.
        """
        if expert_names is None:
            expert_names = [f"expert_{i}" for i in range(len(experts))]

        # Hash calibration data for provenance
        cal_hash = hashlib.sha256()
        for text in calibration_texts[:max_samples]:
            cal_hash.update(text.encode("utf-8"))
            cal_hash.update(b"\n---\n")
        cal_hash_hex = cal_hash.hexdigest()

        # Hash base model
        base_hash = hashlib.sha256()
        for name, param in base_model.named_parameters():
            base_hash.update(name.encode("utf-8"))
            base_hash.update(param.detach().float().sum().item().hex().encode())
        base_hash_hex = base_hash.hexdigest()[:16]

        bank = cls(
            base_model_hash=base_hash_hex,
            calibration_data_hash=cal_hash_hex[:16],
            n_samples=min(len(calibration_texts), max_samples),
            mode="exact_per_sample",
            dtype="float32",
        )

        texts = list(calibration_texts[:max_samples])

        for expert_idx, (expert, expert_name) in enumerate(zip(experts, expert_names)):
            print(f"  Building Fisher for {expert_name} ({len(texts)} samples)...")
            expert = expert.to(device)
            expert.eval()

            # Initialize Fisher accumulators
            fisher_acc: Dict[str, Tensor] = {}
            for name, param in expert.named_parameters():
                fisher_acc[name] = torch.zeros_like(param.detach().float(), device="cpu")

            for text in texts:
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask")
                if input_ids.shape[1] < 2:
                    continue

                expert.zero_grad(set_to_none=True)
                outputs = expert(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                )
                loss.backward()

                for name, param in expert.named_parameters():
                    if param.grad is not None:
                        fisher_acc[name] += param.grad.detach().float().cpu().pow(2)

                expert.zero_grad(set_to_none=True)

            # Average
            n = len(texts)
            if n > 0:
                for name in fisher_acc:
                    fisher_acc[name] /= n

            bank.fisher[expert_name] = fisher_acc
            expert.cpu()

        return bank
