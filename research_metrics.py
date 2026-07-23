import math
import random
from typing import Dict, List, NamedTuple, Optional
import numpy as np
import torch


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
        value=retention_val,
        reason=None,
    )


def compute_pareto_metrics(
    base_losses: Dict[str, float],
    merged_losses: Dict[str, float],
) -> Dict[str, float]:
    """Computes Pareto-style domain trade-off metrics."""
    degradations = {
        d: merged_losses[d] - base_losses[d] for d in base_losses
    }
    improvements = {
        d: base_losses[d] - merged_losses[d] for d in base_losses
    }
    
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
