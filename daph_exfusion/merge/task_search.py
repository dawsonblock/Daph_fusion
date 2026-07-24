"""Task Arithmetic search: simplex grid over (λ₁,...,λ_N, α).

This is the production search for optimized Task Arithmetic.

For N experts, the simplex grid at resolution r produces:
    C(N+r-1, r) = (N+r-1)! / (r! (N-1)!) configurations.

For 3 experts at resolution 0.1: 66 combinations.
Multiply by 5 α values: 330 configurations.

That's trivial at DistilGPT2 scale. No optimizer framework needed.

Constrained objective:
    max_w R_mean(w)
    subject to:
        R_min(w) ≥ τ       (minimum specialist retention)
        G_regression(w) ≤ δ (maximum general regression)

Defaults: τ=0.70, δ=0.25.

Modes:
    TA-0: uniform λᵢ = 1/N, search α only
    TA-1: search (λ₁,...,λ_N, α) on simplex grid
    TA-2: Fisher-weighted: search (w₁,...,w_N, α, γ) with Fisher weighting
    TA-3: family-weighted: search (w_{i,f}, α_f) per family
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    classify_parameter_family_fine,
    FINE_FAMILIES,
)
from daph_exfusion.merge.task_arithmetic import merge_task_arithmetic


# =============================================================================
# Grid generation
# =============================================================================


def generate_simplex_grid(
    n_experts: int,
    resolution: float = 0.1,
) -> List[Tuple[float, ...]]:
    """Generate all points on the probability simplex at given resolution.

    Generates all (λ₁,...,λ_N) such that:
        λᵢ ≥ 0
        Σλᵢ = 1
        each λᵢ is a multiple of `resolution`

    For 3 experts at 0.1: 66 points.
    For 3 experts at 0.05: 231 points.

    Args:
        n_experts: Number of experts.
        resolution: Grid spacing (e.g., 0.1, 0.05, 0.025).

    Returns:
        List of tuples, each summing to 1.0.
    """
    n_steps = int(round(1.0 / resolution))
    points: List[Tuple[float, ...]] = []

    def _generate(remaining: int, n_left: int, current: List[float]):
        if n_left == 1:
            current.append(remaining * resolution)
            points.append(tuple(current))
            current.pop()
            return

        for k in range(remaining + 1):
            current.append(k * resolution)
            _generate(remaining - k, n_left - 1, current)
            current.pop()

    _generate(n_steps, n_experts, [])
    return points


def generate_scale_grid(
    scales: Optional[Sequence[float]] = None,
) -> List[float]:
    """Generate the α grid."""
    if scales is None:
        return [0.25, 0.5, 0.75, 1.0, 1.25]
    return list(scales)


# =============================================================================
# Evaluation result
# =============================================================================


@dataclass
class EvaluationResult:
    """Result of evaluating a single merge configuration."""
    lambdas: Tuple[float, ...]
    scale: float
    # Metrics
    mean_retention: float = 0.0
    min_retention: float = 0.0
    general_regression: float = 0.0
    per_domain_retention: Dict[str, float] = field(default_factory=dict)
    # Constraint satisfaction
    feasible: bool = True
    # Raw score (e.g., validation NLL)
    raw_score: float = 0.0


@dataclass
class SearchResult:
    """Result of a grid search."""
    best: Optional[EvaluationResult] = None
    all_results: List[EvaluationResult] = field(default_factory=list)
    n_configurations: int = 0
    n_feasible: int = 0
    mode: str = "TA-1"
    # Grid parameters
    resolution: float = 0.1
    scales: List[float] = field(default_factory=list)
    # Constraints
    tau: float = 0.70
    delta: float = 0.25

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "resolution": self.resolution,
            "scales": self.scales,
            "n_configurations": self.n_configurations,
            "n_feasible": self.n_feasible,
            "tau": self.tau,
            "delta": self.delta,
            "best": {
                "lambdas": list(self.best.lambdas) if self.best else None,
                "scale": self.best.scale if self.best else None,
                "mean_retention": self.best.mean_retention if self.best else None,
                "min_retention": self.best.min_retention if self.best else None,
                "general_regression": self.best.general_regression if self.best else None,
                "per_domain_retention": self.best.per_domain_retention if self.best else None,
                "feasible": self.best.feasible if self.best else None,
            } if self.best else None,
            "all_results": [
                {
                    "lambdas": list(r.lambdas),
                    "scale": r.scale,
                    "mean_retention": r.mean_retention,
                    "min_retention": r.min_retention,
                    "general_regression": r.general_regression,
                    "feasible": r.feasible,
                }
                for r in self.all_results
            ],
        }


# =============================================================================
# Search functions
# =============================================================================


def evaluate_config(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    lambdas: Tuple[float, ...],
    scale: float,
    evaluator: Callable,
    device: str = "cpu",
) -> EvaluationResult:
    """Evaluate a single (λ, α) configuration.

    Args:
        base_model: Base model.
        experts: Specialist models.
        lambdas: Expert coefficients.
        scale: Global scale α.
        evaluator: Callable(merged_model) -> dict with keys:
            "mean_retention", "min_retention", "general_regression",
            "per_domain_retention"
        device: Device for computation.

    Returns:
        EvaluationResult with metrics.
    """
    config = MergeConfig(
        method=MergeMethod.TASK_ARITHMETIC,
        task_scale=scale,
        lambdas=lambdas,
    )
    result = merge_task_arithmetic(base_model, experts, config, device=device)
    metrics = evaluator(result.merged_model)

    return EvaluationResult(
        lambdas=lambdas,
        scale=scale,
        mean_retention=metrics.get("mean_retention", 0.0),
        min_retention=metrics.get("min_retention", 0.0),
        general_regression=metrics.get("general_regression", 0.0),
        per_domain_retention=metrics.get("per_domain_retention", {}),
        raw_score=metrics.get("raw_score", metrics.get("mean_retention", 0.0)),
    )


def check_constraints(
    result: EvaluationResult,
    tau: float = 0.70,
    delta: float = 0.25,
) -> bool:
    """Check if a result satisfies the constraints.

    R_min ≥ τ and G_regression ≤ δ
    """
    return result.min_retention >= tau and result.general_regression <= delta


def search_ta0(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    evaluator: Callable,
    scales: Optional[Sequence[float]] = None,
    tau: float = 0.70,
    delta: float = 0.25,
    device: str = "cpu",
) -> SearchResult:
    """TA-0: Uniform baseline. Search α only with λᵢ = 1/N.

    θ* = θ₀ + α (1/N) Σᵢ Δᵢ
    """
    n_experts = len(experts)
    uniform_lambdas = tuple(1.0 / n_experts for _ in range(n_experts))
    scale_grid = generate_scale_grid(scales)

    search_result = SearchResult(
        mode="TA-0",
        resolution=0.0,
        scales=scale_grid,
        tau=tau,
        delta=delta,
    )

    for scale in scale_grid:
        result = evaluate_config(base_model, experts, uniform_lambdas, scale, evaluator, device)
        result.feasible = check_constraints(result, tau, delta)
        search_result.all_results.append(result)
        search_result.n_configurations += 1
        if result.feasible:
            search_result.n_feasible += 1

    # Select best: highest mean_retention among feasible, or highest mean_retention overall
    feasible = [r for r in search_result.all_results if r.feasible]
    candidates = feasible if feasible else search_result.all_results
    search_result.best = max(candidates, key=lambda r: r.mean_retention) if candidates else None

    return search_result


def search_ta1(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    evaluator: Callable,
    resolution: float = 0.1,
    scales: Optional[Sequence[float]] = None,
    tau: float = 0.70,
    delta: float = 0.25,
    device: str = "cpu",
    refine_around_best: bool = True,
    refine_resolution: float = 0.025,
    refine_radius: float = 0.1,
) -> SearchResult:
    """TA-1: Weighted Task Arithmetic. Search (λ₁,...,λ_N, α) on simplex grid.

    θ* = θ₀ + α Σᵢ λᵢ Δᵢ

    Two-stage:
        Stage 1: Coarse grid at `resolution`
        Stage 2: Refine around best at `refine_resolution` within `refine_radius`

    Args:
        base_model: Base model.
        experts: Specialist models.
        evaluator: Callable(merged_model) -> dict.
        resolution: Grid spacing for stage 1 (e.g., 0.1).
        scales: α values to search.
        tau: Minimum specialist retention constraint.
        delta: Maximum general regression constraint.
        device: Device for computation.
        refine_around_best: Whether to do stage 2 refinement.
        refine_resolution: Grid spacing for stage 2.
        refine_radius: How far around the best to search in stage 2.

    Returns:
        SearchResult with the best configuration.
    """
    n_experts = len(experts)
    scale_grid = generate_scale_grid(scales)

    # Stage 1: Coarse grid
    lambda_grid = generate_simplex_grid(n_experts, resolution)

    search_result = SearchResult(
        mode="TA-1",
        resolution=resolution,
        scales=scale_grid,
        tau=tau,
        delta=delta,
    )

    for lambdas in lambda_grid:
        for scale in scale_grid:
            result = evaluate_config(base_model, experts, lambdas, scale, evaluator, device)
            result.feasible = check_constraints(result, tau, delta)
            search_result.all_results.append(result)
            search_result.n_configurations += 1
            if result.feasible:
                search_result.n_feasible += 1

    # Select best from stage 1
    feasible = [r for r in search_result.all_results if r.feasible]
    candidates = feasible if feasible else search_result.all_results
    stage1_best = max(candidates, key=lambda r: r.mean_retention) if candidates else None

    if not stage1_best or not refine_around_best:
        search_result.best = stage1_best
        return search_result

    # Stage 2: Refine around best
    refined_grid = _generate_refined_grid(
        stage1_best.lambdas, n_experts, refine_resolution, refine_radius
    )

    for lambdas in refined_grid:
        for scale in scale_grid:
            result = evaluate_config(base_model, experts, lambdas, scale, evaluator, device)
            result.feasible = check_constraints(result, tau, delta)
            search_result.all_results.append(result)
            search_result.n_configurations += 1
            if result.feasible:
                search_result.n_feasible += 1

    # Final selection
    feasible = [r for r in search_result.all_results if r.feasible]
    candidates = feasible if feasible else search_result.all_results
    search_result.best = max(candidates, key=lambda r: r.mean_retention) if candidates else None

    return search_result


def _generate_refined_grid(
    center: Tuple[float, ...],
    n_experts: int,
    resolution: float,
    radius: float,
) -> List[Tuple[float, ...]]:
    """Generate a refined grid around a center point within a radius."""
    points: List[Tuple[float, ...]] = []
    n_steps = int(round(radius / resolution))

    def _generate(remaining_dims: int, current: List[float]):
        if remaining_dims == 0:
            # Check sum constraint and non-negativity
            total = sum(current)
            if abs(total - 1.0) < resolution / 2:
                if all(c >= -resolution / 2 for c in current):
                    clamped = tuple(max(0.0, c) for c in current)
                    # Re-normalize
                    s = sum(clamped)
                    if s > 0:
                        clamped = tuple(c / s for c in clamped)
                    points.append(clamped)
            return

        dim_idx = n_experts - remaining_dims
        center_val = center[dim_idx] if dim_idx < len(center) else 1.0 / n_experts

        for offset in range(-n_steps, n_steps + 1):
            val = center_val + offset * resolution
            if val >= -resolution / 2:  # allow slight negative, clamp later
                current.append(val)
                _generate(remaining_dims - 1, current)
                current.pop()

    _generate(n_experts, [])
    return list(set(points))


def search_ta2(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    evaluator: Callable,
    curvature_bank: Dict[str, Dict[str, Tensor]],
    resolution: float = 0.1,
    scales: Optional[Sequence[float]] = None,
    gamma_grid: Optional[Sequence[float]] = None,
    tau: float = 0.70,
    delta: float = 0.25,
    device: str = "cpu",
) -> SearchResult:
    """TA-2: Fisher-weighted Task Arithmetic.

    θ*_k = θ_{0,k} + α Σᵢ wᵢ (F_{i,k}+ε)^γ Δ_{i,k} / (Σᵢ wᵢ (F_{i,k}+ε)^γ + ε)

    Search over (w₁,...,w_N, α, γ).
    """
    from daph_exfusion.merge.fisher_dense import merge_fisher_dense

    n_experts = len(experts)
    scale_grid = generate_scale_grid(scales)
    if gamma_grid is None:
        gamma_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    lambda_grid = generate_simplex_grid(n_experts, resolution)
    expert_names = [f"expert_{i}" for i in range(n_experts)]

    search_result = SearchResult(
        mode="TA-2",
        resolution=resolution,
        scales=scale_grid,
        tau=tau,
        delta=delta,
    )

    for lambdas in lambda_grid:
        for scale in scale_grid:
            for gamma in gamma_grid:
                config = MergeConfig(
                    method=MergeMethod.FISHER_DENSE,
                    task_scale=scale,
                    lambdas=lambdas,
                    fisher_gamma=gamma,
                )
                try:
                    result = merge_fisher_dense(
                        base_model, experts, config, curvature_bank, device=device
                    )
                    metrics = evaluator(result.merged_model)
                    eval_result = EvaluationResult(
                        lambdas=lambdas,
                        scale=scale,
                        mean_retention=metrics.get("mean_retention", 0.0),
                        min_retention=metrics.get("min_retention", 0.0),
                        general_regression=metrics.get("general_regression", 0.0),
                        per_domain_retention=metrics.get("per_domain_retention", {}),
                        feasible=True,
                    )
                    eval_result.feasible = check_constraints(eval_result, tau, delta)
                except Exception:
                    eval_result = EvaluationResult(
                        lambdas=lambdas, scale=scale, feasible=False,
                        mean_retention=0.0, min_retention=0.0,
                        general_regression=float("inf"),
                    )

                search_result.all_results.append(eval_result)
                search_result.n_configurations += 1
                if eval_result.feasible:
                    search_result.n_feasible += 1

    feasible = [r for r in search_result.all_results if r.feasible]
    candidates = feasible if feasible else search_result.all_results
    search_result.best = max(candidates, key=lambda r: r.mean_retention) if candidates else None

    return search_result


def search_ta3(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    evaluator: Callable,
    resolution: float = 0.25,
    scales: Optional[Sequence[float]] = None,
    tau: float = 0.70,
    delta: float = 0.25,
    device: str = "cpu",
    frozen_families: Optional[Sequence[str]] = None,
) -> SearchResult:
    """TA-3: Family-weighted Task Arithmetic.

    θ*_{f,k} = θ_{0,k} + α_f Σᵢ w_{i,f} Δ_{i,k}

    Each family gets its own (w₁,...,w_N, α) search.
    Families: attention, ssm, ffn, norm, embedding, lm_head, router, other.

    By default, norm/embedding/lm_head are frozen (α=0).
    """
    if frozen_families is None:
        frozen_families = ("norm", "embedding", "lm_head")

    n_experts = len(experts)
    scale_grid = generate_scale_grid(scales)
    uniform = tuple(1.0 / n_experts for _ in range(n_experts))

    # Classify parameters into families
    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()

    param_families: Dict[str, str] = {}
    for name, _ in base_cpu.named_parameters():
        param_families[name] = classify_parameter_family_fine(name)

    # For each non-frozen family, search (w, α)
    family_results: Dict[str, EvaluationResult] = {}
    search_result = SearchResult(
        mode="TA-3",
        resolution=resolution,
        scales=scale_grid,
        tau=tau,
        delta=delta,
    )

    # For frozen families, use uniform λ and α=0 (no change)
    for family in frozen_families:
        family_results[family] = EvaluationResult(
            lambdas=uniform, scale=0.0, feasible=True,
            mean_retention=1.0, min_retention=1.0, general_regression=0.0,
        )

    # For non-frozen families, do a grid search
    # We evaluate the FULL model with family-specific coefficients
    non_frozen = [f for f in FINE_FAMILIES if f not in frozen_families]
    lambda_grid = generate_simplex_grid(n_experts, resolution)

    # Start with uniform for all families
    best_family_lambdas: Dict[str, Tuple[float, ...]] = {
        f: uniform for f in FINE_FAMILIES
    }
    best_family_scales: Dict[str, float] = {
        f: 0.0 for f in frozen_families
    }
    # Initialize non-frozen with α=1.0
    for f in non_frozen:
        best_family_scales[f] = 1.0

    # Greedy per-family optimization
    for family in non_frozen:
        best_score = -float("inf")
        best_lambdas = uniform
        best_scale = 1.0

        for lambdas in lambda_grid:
            for scale in scale_grid:
                # Build family-specific config
                test_lambdas = dict(best_family_lambdas)
                test_lambdas[family] = lambdas
                test_scales = dict(best_family_scales)
                test_scales[family] = scale

                # Merge with family-specific coefficients
                result = _merge_family_weighted(
                    base_cpu, experts, test_lambdas, test_scales,
                    param_families, device,
                )
                metrics = evaluator(result.merged_model)
                eval_result = EvaluationResult(
                    lambdas=lambdas, scale=scale,
                    mean_retention=metrics.get("mean_retention", 0.0),
                    min_retention=metrics.get("min_retention", 0.0),
                    general_regression=metrics.get("general_regression", 0.0),
                    per_domain_retention=metrics.get("per_domain_retention", {}),
                )
                eval_result.feasible = check_constraints(eval_result, tau, delta)
                search_result.all_results.append(eval_result)
                search_result.n_configurations += 1
                if eval_result.feasible:
                    search_result.n_feasible += 1

                # Score: prefer feasible, then higher mean_retention
                score = eval_result.mean_retention
                if not eval_result.feasible:
                    score -= 1.0  # penalty for infeasible

                if score > best_score:
                    best_score = score
                    best_lambdas = lambdas
                    best_scale = scale

        best_family_lambdas[family] = best_lambdas
        best_family_scales[family] = best_scale

    # Final evaluation with all family coefficients
    final_result = _merge_family_weighted(
        base_cpu, experts, best_family_lambdas, best_family_scales,
        param_families, device,
    )
    final_metrics = evaluator(final_result.merged_model)
    final_min_ret = final_metrics.get("min_retention", 0.0)
    final_gen_reg = final_metrics.get("general_regression", 0.0)
    search_result.best = EvaluationResult(
        lambdas=best_family_lambdas.get("attention", uniform),  # representative
        scale=best_family_scales.get("attention", 1.0),
        mean_retention=final_metrics.get("mean_retention", 0.0),
        min_retention=final_min_ret,
        general_regression=final_gen_reg,
        per_domain_retention=final_metrics.get("per_domain_retention", {}),
        feasible=(final_min_ret >= tau and final_gen_reg <= delta),
    )

    # Store family-specific results in all_results metadata
    search_result.all_results.append(search_result.best)

    return search_result


def _merge_family_weighted(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    family_lambdas: Dict[str, Tuple[float, ...]],
    family_scales: Dict[str, float],
    param_families: Dict[str, str],
    device: str = "cpu",
) -> MergeResult:
    """Merge with family-specific coefficients.

    θ*_{f,k} = θ_{0,k} + α_f Σᵢ w_{i,f} Δ_{i,k}
    """
    n_experts = len(experts)
    task_vectors = extract_task_vectors(experts, base_model)

    merged = copy.deepcopy(base_model)
    merged_params = dict(merged.named_parameters())

    with torch.no_grad():
        for name, param in merged_params.items():
            family = param_families.get(name, "other")
            lambdas = family_lambdas.get(family, tuple(1.0 / n_experts for _ in range(n_experts)))
            scale = family_scales.get(family, 0.0)

            if scale == 0.0:
                continue  # frozen family

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
        method="family_task_arithmetic",
        operators=["FAMILY_TASK_ARITHMETIC"],
        task_scale=1.0,  # scale is per-family
        lambdas=list(family_lambdas.get("attention", ())),
        config_hash="",  # TODO: compute proper hash
    )

    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=MergeConfig(method=MergeMethod.TASK_ARITHMETIC),
        method="family_task_arithmetic",
    )


# =============================================================================
# Unified search entry point
# =============================================================================


def search_task_arithmetic(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    evaluator: Callable,
    mode: str = "TA-1",
    resolution: float = 0.1,
    scales: Optional[Sequence[float]] = None,
    tau: float = 0.70,
    delta: float = 0.25,
    device: str = "cpu",
    curvature_bank: Optional[Dict[str, Dict[str, Tensor]]] = None,
    **kwargs,
) -> SearchResult:
    """Unified search entry point for Task Arithmetic optimization.

    Args:
        base_model: Base model θ₀.
        experts: Specialist models.
        evaluator: Callable(merged_model) -> dict with mean_retention, min_retention,
                   general_regression, per_domain_retention.
        mode: Search mode — "TA-0", "TA-1", "TA-2", or "TA-3".
        resolution: Simplex grid resolution.
        scales: α values to search.
        tau: Minimum specialist retention constraint (default 0.70).
        delta: Maximum general regression constraint (default 0.25).
        device: Device for computation.
        curvature_bank: Required for TA-2.
        **kwargs: Additional mode-specific arguments.

    Returns:
        SearchResult with the best configuration and all evaluated configs.
    """
    if mode == "TA-0":
        return search_ta0(base_model, experts, evaluator, scales, tau, delta, device)
    elif mode == "TA-1":
        return search_ta1(base_model, experts, evaluator, resolution, scales,
                          tau, delta, device, **kwargs)
    elif mode == "TA-2":
        if curvature_bank is None:
            raise ValueError("TA-2 requires curvature_bank")
        return search_ta2(base_model, experts, evaluator, curvature_bank,
                          resolution, scales, tau=tau, delta=delta,
                          device=device, **kwargs)
    elif mode == "TA-3":
        return search_ta3(base_model, experts, evaluator, resolution, scales,
                          tau, delta, device, **kwargs)
    else:
        raise ValueError(f"Unknown search mode: {mode}. Use TA-0, TA-1, TA-2, or TA-3.")
