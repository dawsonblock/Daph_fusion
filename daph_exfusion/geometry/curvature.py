"""
Empirical Fisher & Curvature Estimation (Phase 21-22).

Includes disk-backed / CPU-offloaded diagonal accumulation for large models
(ISSUES.md - Issue 3): gradients are squared and accumulated in CPU RAM or
memory-mapped float32 file buffers instead of GPU VRAM.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def build_empirical_fisher_diagonals(
    model: nn.Module,
    calibration_batch: Any,
    micro_batch_size: int = 1,
) -> Dict[str, Tensor]:
    model.eval()
    fisher_diagonals: Dict[str, Tensor] = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    if isinstance(calibration_batch, dict):
        input_ids = calibration_batch["input_ids"]
        attention_mask = calibration_batch.get("attention_mask")
    elif isinstance(calibration_batch, torch.Tensor):
        input_ids = calibration_batch
        attention_mask = None
    else:
        return fisher_diagonals

    num_samples = input_ids.shape[0]
    for i in range(0, num_samples, micro_batch_size):
        model.zero_grad()
        sub_input = input_ids[i : i + micro_batch_size]
        sub_mask = (
            attention_mask[i : i + micro_batch_size]
            if attention_mask is not None
            else None
        )

        outputs = model(sub_input, attention_mask=sub_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = sub_input[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_diagonals[name] += param.grad.square() * (
                        sub_input.shape[0] / num_samples
                    )

    return fisher_diagonals


def _normalize_calibration_batch(
    calibration_batch: Any,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if isinstance(calibration_batch, dict):
        return calibration_batch["input_ids"], calibration_batch.get("attention_mask")
    if isinstance(calibration_batch, torch.Tensor):
        return calibration_batch, None
    return None, None


def build_empirical_fisher_diagonals_offloaded(
    model: nn.Module,
    calibration_batch: Any,
    micro_batch_size: int = 1,
    use_mmap: bool = False,
    mmap_dir: Optional[str] = None,
) -> Dict[str, Tensor]:
    """Memory-efficient empirical Fisher diagonal accumulation (ISSUES.md - Issue 3).

    Squared gradients are accumulated outside GPU VRAM:
      - use_mmap=False: pinned CPU float32 buffers.
      - use_mmap=True: disk-backed memory-mapped float32 buffers created with
        torch.from_file (shared mmap), enabling 70B+ scale accumulation without
        resident RAM pressure.
    """
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be >= 1")

    model.eval()
    fisher_diagonals: Dict[str, Tensor] = {}
    temp_dir = mmap_dir or (
        tempfile.mkdtemp(prefix="daph_fisher_") if use_mmap else None
    )

    # Initialize CPU or memory-mapped storage
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if use_mmap:
            assert temp_dir is not None
            os.makedirs(temp_dir, exist_ok=True)
            mmap_path = os.path.join(temp_dir, f"{name.replace('.', '_')}.bin")
            tensor = torch.from_file(
                mmap_path,
                shared=True,
                size=param.numel(),
                dtype=torch.float32,
            ).view(param.shape)
            tensor.zero_()
            fisher_diagonals[name] = tensor
        else:
            fisher_diagonals[name] = torch.zeros(
                param.shape, device="cpu", dtype=torch.float32
            )

    input_ids, attention_mask = _normalize_calibration_batch(calibration_batch)
    if input_ids is None:
        return fisher_diagonals

    first_parameter = next(model.parameters(), None)
    model_device = (
        first_parameter.device if first_parameter is not None else torch.device("cpu")
    )

    # Accumulate gradients micro-batch by micro-batch
    num_samples = input_ids.shape[0]
    for start_idx in range(0, num_samples, micro_batch_size):
        model.zero_grad(set_to_none=True)
        sub_ids = input_ids[start_idx : start_idx + micro_batch_size].to(model_device)
        sub_mask = (
            attention_mask[start_idx : start_idx + micro_batch_size].to(model_device)
            if attention_mask is not None
            else None
        )

        outputs = model(sub_ids, attention_mask=sub_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            sub_ids[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_sq = param.grad.detach().cpu().float().square()
                    fisher_diagonals[name] += grad_sq * (sub_ids.shape[0] / num_samples)

    model.zero_grad(set_to_none=True)
    return fisher_diagonals
