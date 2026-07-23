"""
Loss distribution, causal shift, and padding mask diagnostics (Phase 2).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def audit_causal_loss_and_padding(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: str = "cpu",
) -> Dict[str, Any]:
    model.eval()
    model.to(device)

    all_losses: List[float] = []
    sample_stats: List[Dict[str, Any]] = []

    for idx, text in enumerate(texts):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            valid_loss = loss[shift_labels.view(-1) != -100]
            if valid_loss.numel() > 0:
                mean_loss = valid_loss.mean().item()
                all_losses.extend(valid_loss.tolist())
            else:
                mean_loss = 0.0

            sample_stats.append(
                {
                    "sample_id": idx,
                    "token_count": int((shift_labels != -100).sum().item()),
                    "mean_nll": mean_loss,
                    "ppl": math.exp(mean_loss) if mean_loss < 20 else float("inf"),
                }
            )

    losses_np = np.array(all_losses) if all_losses else np.array([0.0])
    return {
        "mean_nll": float(np.mean(losses_np)),
        "std_nll": float(np.std(losses_np)),
        "p90_nll": float(np.percentile(losses_np, 90)),
        "p95_nll": float(np.percentile(losses_np, 95)),
        "p99_nll": float(np.percentile(losses_np, 99)),
        "max_nll": float(np.max(losses_np)),
        "sample_count": len(texts),
        "samples": sample_stats,
    }
