import math
import random
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetentionResult(NamedTuple):
    valid: bool
    value: Optional[float]
    reason: Optional[str]


def compute_expert_advantage(base_loss: float, expert_loss: float) -> float:
    """Computes expert advantage G_d = L_base - L_expert."""
    return base_loss - expert_loss


def calculate_retention(
    base_loss: float,
    expert_loss: float,
    merged_loss: float,
    epsilon: float = 1e-4,
) -> RetentionResult:
    """
    Computes specialist-relative retention:
      R_d = (L_base - L_merged) / (L_base - L_expert)
    Only valid if G_d = L_base - L_expert > epsilon.
    """
    advantage = compute_expert_advantage(base_loss, expert_loss)
    if advantage <= epsilon:
        return RetentionResult(
            valid=False,
            value=None,
            reason="expert_does_not_outperform_base",
        )

    retention_val = (base_loss - merged_loss) / advantage
    return RetentionResult(
        valid=True,
        value=float(retention_val),
        reason=None,
    )


def compute_domain_metrics(
    base_loss: float,
    expert_loss: float,
    merged_loss: float,
) -> Dict[str, Any]:
    abs_gain = base_loss - merged_loss
    rel_base_gain = (base_loss - merged_loss) / base_loss if base_loss > 0 else 0.0
    retention = calculate_retention(base_loss, expert_loss, merged_loss)
    regression = max(0.0, merged_loss - base_loss)
    ppl = math.exp(merged_loss) if merged_loss < 20 else float("inf")

    return {
        "absolute_gain": abs_gain,
        "relative_base_gain": rel_base_gain,
        "retention_valid": retention.valid,
        "retention_value": retention.value,
        "retention_reason": retention.reason,
        "regression": regression,
        "nll": merged_loss,
        "ppl": ppl,
    }


def compute_domain_nll(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 128,
) -> Tuple[float, float]:
    model.eval()
    model.to(device)

    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
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
                reduction="sum",
            )

            num_tokens = (shift_labels != -100).sum().item()
            total_nll += loss.item()
            total_tokens += num_tokens

    if total_tokens == 0:
        return 0.0, float("inf")

    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll) if mean_nll < 20 else float("inf")
    return mean_nll, ppl


def compute_pareto_metrics(
    base_losses: Dict[str, float],
    merged_losses: Dict[str, float],
) -> Dict[str, float]:
    """Computes Pareto-style domain trade-off metrics."""
    degradations = {d: merged_losses[d] - base_losses[d] for d in base_losses}
    improvements = {d: base_losses[d] - merged_losses[d] for d in base_losses}

    mean_degradation = float(np.mean(list(degradations.values())))
    worst_degradation = float(max(degradations.values()))
    improved_count = sum(1 for v in improvements.values() if v > 0)
    degraded_count = sum(1 for v in degradations.values() if v > 0)
    max_domain_gain = float(max(improvements.values()))
    max_domain_loss = float(max(degradations.values()))

    return {
        "mean_degradation": mean_degradation,
        "worst_degradation": worst_degradation,
        "improved_domains_count": float(improved_count),
        "degraded_domains_count": float(degraded_count),
        "max_domain_gain": max_domain_gain,
        "max_domain_loss": max_domain_loss,
    }


def seed_everything(seed: int) -> torch.Generator:
    """Central RNG initialization returning an explicit torch.Generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
