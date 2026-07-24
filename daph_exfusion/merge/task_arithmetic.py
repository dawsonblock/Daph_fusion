"""Task Arithmetic merge module (Phase 4 — canonical dense baseline).

Task Arithmetic is the primary dense merge baseline.

Definition:
    Δᵢ = θᵢ - θ₀
    θ* = θ₀ + α Σᵢ λᵢ Δᵢ

Variants:
    TA-U (uniform):           λᵢ = 1/N
    TA-S (scale-optimized):   λᵢ = 1/N, α searched over {0.25, 0.5, ..., 1.25}
    TA-O (coefficient-opt):   λᵢ optimized subject to constraints

Constraints tested:
    unconstrained, λᵢ ≥ 0, Σλᵢ = 1, Σλᵢ ≤ 1

Every later method must beat optimized Task Arithmetic, not naive equal
averaging. TA-O is the actual baseline ceiling.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from daph_exfusion.merge.types import (
    MergeConfig,
    MergeMethod,
    MergeResult,
    OperatorTrace,
    extract_task_vectors,
    validate_parameter_names,
)


def merge_task_arithmetic(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    device: str = "cpu",
) -> MergeResult:
    """Execute Task Arithmetic merge.

    θ* = θ₀ + α Σᵢ λᵢ Δᵢ

    Args:
        base_model: Base model θ₀.
        experts: List of specialist models.
        config: Merge configuration (method must be task_arithmetic).
        device: Device for computation.

    Returns:
        MergeResult with the merged model and operator trace.
    """
    if config.method != MergeMethod.TASK_ARITHMETIC:
        raise ValueError(
            f"merge_task_arithmetic called with method={config.method}, "
            f"expected task_arithmetic"
        )

    n_experts = len(experts)
    validate_parameter_names(experts, base_model)

    # Coefficients: empty = uniform
    if config.lambdas:
        lambdas = list(config.lambdas)
        if len(lambdas) != n_experts:
            raise ValueError(
                f"lambdas length {len(lambdas)} != n_experts {n_experts}"
            )
    else:
        lambdas = [1.0 / n_experts] * n_experts

    scale = config.task_scale

    # Extract task vectors in FP32
    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    task_vectors = extract_task_vectors(experts, base_cpu)

    # Create merged model
    merged = copy.deepcopy(base_cpu)
    merged_params = dict(merged.named_parameters())

    with torch.no_grad():
        for name, param in merged_params.items():
            deltas = []
            for i, tv in enumerate(task_vectors):
                if name in tv:
                    deltas.append(tv[name] * lambdas[i])
            if not deltas:
                continue
            merged_delta = sum(deltas)
            param.copy_(param.detach().float() + merged_delta * scale)

    merged.to(device)

    trace = OperatorTrace(
        method="task_arithmetic",
        operators=["TASK_ARITHMETIC"],
        task_scale=scale,
        lambdas=lambdas,
        fisher_used=False,
        activation_covariance_used=False,
        dare_used=False,
        ties_used=False,
        config_hash=config.config_hash(),
    )

    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method="task_arithmetic",
    )


def merge_frozen(
    base_model: nn.Module,
    config: MergeConfig,
    device: str = "cpu",
) -> MergeResult:
    """Frozen merge: return base model unchanged.

    Used as a control and as the default for normalization/embedding
    groups where merging is not beneficial.
    """
    merged = copy.deepcopy(base_model.cpu())
    merged.to(device)

    trace = OperatorTrace(
        method="frozen",
        operators=["FROZEN"],
        config_hash=config.config_hash(),
    )

    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method="frozen",
    )


# =============================================================================
# Scale search (TA-S)
# =============================================================================


DEFAULT_SCALE_GRID: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25)


def search_task_arithmetic_scale(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    evaluator,
    scale_grid: Optional[Sequence[float]] = None,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Search global scale α for Task Arithmetic.

    Args:
        base_model: Base model.
        experts: Specialist models.
        evaluator: Callable(merged_model) -> float (lower is better, e.g. NLL).
        scale_grid: Scale values to search. Defaults to {0.25, ..., 1.25}.
        device: Device for computation.

    Returns:
        (best_scale, best_score)
    """
    if scale_grid is None:
        scale_grid = DEFAULT_SCALE_GRID

    best_scale = 1.0
    best_score = float("inf")

    for scale in scale_grid:
        config = MergeConfig(
            method=MergeMethod.TASK_ARITHMETIC,
            task_scale=scale,
        )
        result = merge_task_arithmetic(base_model, experts, config, device=device)
        score = evaluator(result.merged_model)
        if score < best_score:
            best_score = score
            best_scale = scale

    return best_scale, best_score
