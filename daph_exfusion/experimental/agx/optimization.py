"""
Optuna & Derivative-Free Optimization Engines (Phase 15) and the
Adaptive Geometry ExFusion (AGX v2.4) Layerwise Search Engine (Phase 5).

theta*_l = theta_{0,l} + sum_i lambda_{i,l} * Op_l(Delta_{i,l})
subject to: D_repr,l = 1 - CKA(H_{0,l}, H_{m,l}) <= max_cka_drift
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from daph_exfusion.geometry.representations import compute_linear_cka, MetricResult
from daph_exfusion.geometry.operators import (
    SINGLE_EXPERT_OPS,
    CROSS_EXPERT_OPS,
    transform_single_delta,
    transform_expert_set,
)
from daph_exfusion.experimental.agx.candidate import LayerMergeConfig, MergeCandidate


def generate_random_layerwise_candidate(
    num_layers: int,
    num_experts: int,
) -> MergeCandidate:
    operators = ["RAW", "NORMALIZED", "TIES", "DARE", "FISHER", "PROJECT", "DELTA_DROPOUT"]
    layer_configs = {}
    for l in range(num_layers):
        op = random.choice(operators)
        lambdas = tuple(
            round(random.uniform(0.05, 0.45), 3) for _ in range(num_experts)
        )
        layer_configs[l] = LayerMergeConfig(
            operator=op,
            lambdas=lambdas,
            ties_trim=0.2,
            dare_drop=0.2,
            fisher_gamma=0.5,
        )
    return MergeCandidate(layer_configs=layer_configs)


# =============================================================================
# Layerwise merge operators (AGX v2.4)
# =============================================================================


def _transform_delta(
    delta: Tensor,
    config: LayerMergeConfig,
    generator: Optional[torch.Generator],
) -> Tensor:
    """Applies a SINGLE-EXPERT per-layer operator Op_l to one expert delta.

    For cross-expert operators (TIES, FISHER, TIES_FISHER, etc.), this
    function raises — they MUST go through the cross-expert path in
    ``apply_layer_merge_operator`` via ``transform_expert_set``.
    """
    op = config.operator.upper()
    if op in CROSS_EXPERT_OPS:
        raise ValueError(
            f"Operator '{op}' is a cross-expert operator and cannot be applied "
            f"to a single delta. Use the cross-expert path in "
            f"apply_layer_merge_operator which calls transform_expert_set()."
        )
    return transform_single_delta(
        delta,
        op,
        generator=generator,
        dare_drop=config.dare_drop,
    )


def apply_layer_merge_operator(
    target_layer: nn.Module,
    base_layer: nn.Module,
    expert_layers: Sequence[nn.Module],
    operator: str,
    lambdas: Sequence[float],
    ties_trim: float = 0.2,
    dare_drop: float = 0.0,
    seed: int = 17,
    fisher_diagonals: Optional[Dict[int, List[Tensor]]] = None,
    sign_mode: str = "magnitude",
) -> None:
    """Merges one layer in place:
    theta*_l = theta_{0,l} + Op_l({Delta_{i,l}}).

    For single-expert operators (RAW, NORMALIZED, DARE, DELTA_DROPOUT,
    PROJECT), applies per-delta and sums with lambdas.

    For cross-expert operators (TIES, FISHER, TIES_FISHER, DARE_TIES,
    DARE_TIES_FISHER), calls transform_expert_set() on the full delta set
    so that sign election and Fisher weighting actually execute.
    """
    if len(expert_layers) != len(lambdas):
        raise ValueError(
            f"expert_layers count {len(expert_layers)} != lambdas count {len(lambdas)}"
        )

    config = LayerMergeConfig(
        operator=operator,
        lambdas=tuple(float(v) for v in lambdas),
        ties_trim=ties_trim,
        dare_drop=dare_drop,
    )
    op = operator.upper()
    generator = torch.Generator().manual_seed(seed)

    base_params = dict(base_layer.named_parameters())
    expert_params = [dict(e.named_parameters()) for e in expert_layers]

    with torch.no_grad():
        for name, target_param in target_layer.named_parameters():
            base_param = base_params.get(name)
            if base_param is None:
                continue

            # Collect deltas for this parameter across all experts
            deltas = []
            for lam, e_params in zip(config.lambdas, expert_params):
                expert_param = e_params.get(name)
                if expert_param is None:
                    continue
                delta = (expert_param.detach().float() - base_param.detach().float()) * lam
                deltas.append(delta)

            if not deltas:
                continue

            merged = base_param.detach().clone().float()

            if op in CROSS_EXPERT_OPS:
                # Cross-expert path: use transform_expert_set
                fishers = None
                if op in ("FISHER", "TIES_FISHER", "DARE_TIES_FISHER"):
                    if fisher_diagonals is None:
                        raise ValueError(
                            f"Operator '{op}' requires fisher_diagonals but none were provided."
                        )
                    fishers = fisher_diagonals.get(name)
                    if fishers is None:
                        # Fallback: uniform Fisher if not in bank
                        fishers = [torch.ones_like(d) for d in deltas]
                merged_delta = transform_expert_set(
                    deltas,
                    op,
                    fisher_diagonals=fishers,
                    trim_fraction=ties_trim,
                    fisher_gamma=config.fisher_gamma,
                    generator=generator,
                    dare_drop=dare_drop,
                    sign_mode=sign_mode,
                )
                merged += merged_delta
            else:
                # Single-expert path: apply per-delta and sum
                for delta in deltas:
                    merged += _transform_delta(delta, config, generator)

            target_param.copy_(merged.to(target_param.dtype))


# =============================================================================
# Validation objective
# =============================================================================


def compute_validation_nll(
    model: nn.Module, validation_batch: Dict[str, Tensor]
) -> float:
    """Shift cross-entropy NLL on a tokenized validation batch."""
    model.eval()
    input_ids = validation_batch["input_ids"]
    attention_mask = validation_batch.get("attention_mask")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous().bool()
            shift_labels = torch.where(
                shift_mask,
                shift_labels,
                torch.tensor(-100, device=shift_labels.device),
            )
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
    return float(loss.item())


# =============================================================================
# AGX v2.4 Layerwise Geometry Search Engine
# =============================================================================


def _default_layer_modules(model: nn.Module) -> Sequence[nn.Module]:
    """HF GPT-2 style layer accessor: model.transformer.h."""
    return model.transformer.h


def _default_hidden_states(
    model: nn.Module, batch: Dict[str, Tensor]
) -> Sequence[Tensor]:
    outputs = model(
        batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        output_hidden_states=True,
    )
    return outputs.hidden_states


@dataclass
class CandidateEvaluation:
    candidate_hash: str
    validation_nll: float
    feasible: bool
    max_layer_drift: float


class LayerwiseGeometrySearchEngine:
    """Searches layerwise merge geometries G*_l under CKA drift safeguards.

    Candidates whose per-layer representation drift
    D_repr,l = 1 - CKA(H_{0,l}, H_{m,l}) exceeds max_cka_drift are rejected
    (objective = +inf), guaranteeing merged models stay within a bounded
    representation neighborhood of the base model.
    """

    def __init__(
        self,
        base_model: nn.Module,
        experts: Sequence[nn.Module],
        validation_batch: Dict[str, Tensor],
        max_cka_drift: float = 0.15,
        layer_module_fn: Callable[
            [nn.Module], Sequence[nn.Module]
        ] = _default_layer_modules,
        hidden_state_fn: Callable[
            [nn.Module, Dict[str, Tensor]], Sequence[Tensor]
        ] = _default_hidden_states,
        curvature_bank: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ) -> None:
        self.base_model = base_model
        self.experts = list(experts)
        self.val_batch = validation_batch
        self.max_cka_drift = max_cka_drift
        self._layer_module_fn = layer_module_fn
        self._hidden_state_fn = hidden_state_fn
        self.curvature_bank = curvature_bank

    @property
    def num_layers(self) -> int:
        return len(self._layer_module_fn(self.base_model))

    def build_candidate_model(self, candidate: MergeCandidate) -> nn.Module:
        """Constructs the candidate merged model layer by layer."""
        merged_model = copy.deepcopy(self.base_model)
        merged_layers = self._layer_module_fn(merged_model)
        base_layers = self._layer_module_fn(self.base_model)
        expert_layer_lists = [self._layer_module_fn(e) for e in self.experts]

        for layer_idx, config in candidate.layer_configs.items():
            if layer_idx >= len(base_layers):
                continue
            apply_layer_merge_operator(
                target_layer=merged_layers[layer_idx],
                base_layer=base_layers[layer_idx],
                expert_layers=[layers[layer_idx] for layers in expert_layer_lists],
                operator=config.operator,
                lambdas=config.lambdas,
                ties_trim=config.ties_trim,
                dare_drop=config.dare_drop,
                fisher_diagonals=getattr(self, 'curvature_bank', None),
                sign_mode=getattr(config, 'sign_mode', 'magnitude'),
            )
        return merged_model

    def measure_representation_drift(self, merged_model: nn.Module) -> List[float]:
        """Per-layer drift D_repr,l = 1 - CKA(H_{0,l}, H_{m,l}) on validation batch.

        Uses the repaired token-observation CKA. Invalid CKA results (e.g.
        from a degenerate batch) are treated as max drift (1.0) so that a
        broken measurement cannot silently pass the drift safeguard.
        """
        with torch.no_grad():
            h_base = self._hidden_state_fn(self.base_model, self.val_batch)
            h_merged = self._hidden_state_fn(merged_model, self.val_batch)
        # Extract attention mask from val_batch (it's a dict, not an object)
        mask = self.val_batch.get("attention_mask") if isinstance(self.val_batch, dict) else getattr(self.val_batch, "attention_mask", None)
        drifts: List[float] = []
        for hb, hm in zip(h_base, h_merged):
            cka_result: MetricResult = compute_linear_cka(hb, hm, attention_mask=mask)
            if not cka_result.valid or cka_result.value is None:
                # Conservative: treat invalid CKA as maximum drift so the
                # candidate is rejected rather than silently accepted.
                drifts.append(1.0)
            else:
                drifts.append(1.0 - cka_result.value)
        return drifts

    def evaluate_candidate(self, candidate: MergeCandidate) -> Tuple[float, bool]:
        """Returns (validation NLL, feasible). Infeasible candidates score +inf."""
        # 1. Construct candidate merged model layer by layer
        merged_model = self.build_candidate_model(candidate)

        # 2. Measure CKA Representation Drift on Validation Batch
        drifts = self.measure_representation_drift(merged_model)
        for drift in drifts:
            if drift > self.max_cka_drift:
                # Reject candidate violating drift safeguard
                return float("inf"), False

        # 3. Compute Validation NLL
        val_nll = compute_validation_nll(merged_model, self.val_batch)
        return val_nll, True

    def search(
        self,
        num_candidates: int = 16,
        seed: int = 17,
    ) -> Tuple[Optional[MergeCandidate], List[CandidateEvaluation]]:
        """Random layerwise geometry search; returns best feasible candidate."""
        random.seed(seed)
        history: List[CandidateEvaluation] = []
        best_candidate: Optional[MergeCandidate] = None
        best_nll = float("inf")

        for _ in range(num_candidates):
            candidate = generate_random_layerwise_candidate(
                num_layers=self.num_layers,
                num_experts=len(self.experts),
            )
            merged_model = self.build_candidate_model(candidate)
            drifts = self.measure_representation_drift(merged_model)
            max_drift = max(drifts) if drifts else 0.0
            feasible = max_drift <= self.max_cka_drift
            nll = (
                compute_validation_nll(merged_model, self.val_batch)
                if feasible
                else float("inf")
            )
            history.append(
                CandidateEvaluation(
                    candidate_hash=candidate.compute_hash(),
                    validation_nll=nll,
                    feasible=feasible,
                    max_layer_drift=max_drift,
                )
            )
            if feasible and nll < best_nll:
                best_nll = nll
                best_candidate = candidate

        return best_candidate, history
