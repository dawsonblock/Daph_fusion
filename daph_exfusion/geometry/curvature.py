"""
Empirical Fisher & Curvature Estimation (Phase 21-22).
"""

from __future__ import annotations

from typing import Any, Dict, List

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
