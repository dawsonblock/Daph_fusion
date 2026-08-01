"""Production search for optimized Task Arithmetic (TA-0 through TA-3).

The search hierarchy is deliberately small and falsifiable:

TA-0
    Uniform expert coefficients, search only global scale α.
TA-1
    Search expert coefficients λ on a simplex plus independent global α.
TA-2
    Search λ, α and Fisher exponent γ using Fisher-weighted Task Arithmetic.
TA-3
    Greedy architecture-family refinement with per-family λ and α.

Selection is constrained rather than based on a single unconstrained score:

    maximize mean specialist retention
    subject to min specialist retention >= tau
               general regression <= delta

Research-integrity rule: algorithm/configuration errors are never swallowed and
relabelled as poor candidates. Missing or invalid curvature therefore fails a
TA-2 search closed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from daph_exfusion.merge.fisher_task_arithmetic import merge_fisher_task_arithmetic
from daph_exfusion.merge.task_arithmetic import merge_task_arithmetic
from daph_exfusion.merge.types import (
    FINE_FAMILIES,
    MergeConfig,
    MergeMethod,
    MergeResult,
    OperatorTrace,
    classify_parameter_family_fine,
    validate_parameter_names,
)

DEFAULT_SCALES: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25)
DEFAULT_GAMMAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_FROZEN_FAMILIES: Tuple[str, ...] = ("norm", "embedding", "lm_head")


def _validate_experts(experts: Sequence[nn.Module]) -> int:
    n = len(experts)
    if n <= 0:
        raise ValueError("at least one expert is required")
    return n


def _validate_resolution(resolution: float) -> int:
    if not math.isfinite(resolution) or resolution <= 0 or resolution > 1:
        raise ValueError("resolution must be finite and in (0, 1]")
    n_steps = int(round(1.0 / resolution))
    if n_steps <= 0 or not math.isclose(n_steps * resolution, 1.0, abs_tol=1e-9):
        raise ValueError(
            "resolution must divide 1.0 exactly (for example 0.5, 0.25, 0.1, 0.05)"
        )
    return n_steps


def generate_simplex_grid(
    n_experts: int,
    resolution: float = 0.1,
) -> List[Tuple[float, ...]]:
    """Generate all non-negative simplex points at the requested resolution."""
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    n_steps = _validate_resolution(float(resolution))
    points: List[Tuple[float, ...]] = []

    def visit(remaining: int, dims_left: int, prefix: List[int]) -> None:
        if dims_left == 1:
            ints = prefix + [remaining]
            values = tuple(v / n_steps for v in ints)
            points.append(values)
            return
        for value in range(remaining + 1):
            visit(remaining - value, dims_left - 1, prefix + [value])

    visit(n_steps, n_experts, [])
    return points


def generate_scale_grid(scales: Optional[Sequence[float]] = None) -> List[float]:
    """Return a validated α grid."""
    values = list(DEFAULT_SCALES if scales is None else scales)
    if not values:
        raise ValueError("scale grid must not be empty")
    values = [float(v) for v in values]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("scale values must be finite and non-negative")
    return values


@dataclass
class EvaluationResult:
    """Metrics for one evaluated merge configuration."""

    lambdas: Tuple[float, ...]
    scale: float
    mean_retention: float = 0.0
    min_retention: float = 0.0
    general_regression: float = 0.0
    per_domain_retention: Dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    raw_score: float = 0.0
    gamma: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "lambdas": list(self.lambdas),
            "scale": self.scale,
            "gamma": self.gamma,
            "mean_retention": self.mean_retention,
            "min_retention": self.min_retention,
            "general_regression": self.general_regression,
            "per_domain_retention": dict(self.per_domain_retention),
            "feasible": self.feasible,
            "raw_score": self.raw_score,
        }


@dataclass
class SearchResult:
    """Complete auditable result of a TA search."""

    best: Optional[EvaluationResult] = None
    all_results: List[EvaluationResult] = field(default_factory=list)
    n_configurations: int = 0
    n_feasible: int = 0
    mode: str = "TA-1"
    resolution: float = 0.1
    scales: List[float] = field(default_factory=list)
    tau: float = 0.70
    delta: float = 0.25
    gamma_grid: List[float] = field(default_factory=list)
    # TA-3 must preserve the entire solution, not a representative family.
    family_lambdas: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    family_scales: Dict[str, float] = field(default_factory=dict)
    family_order: List[str] = field(default_factory=list)
    solution_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "resolution": self.resolution,
            "scales": list(self.scales),
            "gamma_grid": list(self.gamma_grid),
            "n_configurations": self.n_configurations,
            "n_feasible": self.n_feasible,
            "tau": self.tau,
            "delta": self.delta,
            "solution_hash": self.solution_hash,
            "family_order": list(self.family_order),
            "family_lambdas": {
                family: list(values)
                for family, values in self.family_lambdas.items()
            },
            "family_scales": dict(self.family_scales),
            "best": self.best.to_dict() if self.best else None,
            "all_results": [result.to_dict() for result in self.all_results],
        }


def _metrics_to_evaluation(
    metrics: dict,
    lambdas: Tuple[float, ...],
    scale: float,
    gamma: Optional[float] = None,
) -> EvaluationResult:
    required = ("mean_retention", "min_retention", "general_regression")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"evaluator missing required metrics: {missing}")

    values = [
        float(metrics["mean_retention"]),
        float(metrics["min_retention"]),
        float(metrics["general_regression"]),
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("evaluator returned non-finite retention/regression metrics")

    return EvaluationResult(
        lambdas=tuple(float(v) for v in lambdas),
        scale=float(scale),
        gamma=None if gamma is None else float(gamma),
        mean_retention=values[0],
        min_retention=values[1],
        general_regression=values[2],
        per_domain_retention={
            str(k): float(v)
            for k, v in metrics.get("per_domain_retention", {}).items()
        },
        raw_score=float(metrics.get("raw_score", values[0])),
    )


def check_constraints(
    result: EvaluationResult,
    tau: float = 0.70,
    delta: float = 0.25,
) -> bool:
    """Return whether R_min >= tau and general regression <= delta."""
    if not math.isfinite(tau) or not math.isfinite(delta):
        raise ValueError("tau and delta must be finite")
    return result.min_retention >= tau and result.general_regression <= delta


def _select_best(results: Sequence[EvaluationResult]) -> Optional[EvaluationResult]:
    if not results:
        return None
    feasible = [result for result in results if result.feasible]
    candidates = feasible if feasible else list(results)
    # Deterministic tie-break: mean retention, then worst-domain retention,
    # then lower general regression, then lower scale.
    return max(
        candidates,
        key=lambda r: (
            r.mean_retention,
            r.min_retention,
            -r.general_regression,
            -r.scale,
        ),
    )


def evaluate_config(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    lambdas: Tuple[float, ...],
    scale: float,
    evaluator: Callable,
    device: str = "cpu",
) -> EvaluationResult:
    """Evaluate one TA-0/TA-1 configuration."""
    config = MergeConfig(
        method=MergeMethod.TASK_ARITHMETIC,
        task_scale=float(scale),
        lambdas=tuple(float(v) for v in lambdas),
    )
    merged = merge_task_arithmetic(base_model, experts, config, device=device)
    return _metrics_to_evaluation(evaluator(merged.merged_model), lambdas, scale)


def search_ta0(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    evaluator: Callable,
    scales: Optional[Sequence[float]] = None,
    tau: float = 0.70,
    delta: float = 0.25,
    device: str = "cpu",
) -> SearchResult:
    """TA-0: uniform λ, search only α."""
    n_experts = _validate_experts(experts)
    validate_parameter_names(experts, base_model)
    uniform = tuple(1.0 / n_experts for _ in range(n_experts))
    scale_grid = generate_scale_grid(scales)
    output = SearchResult(
        mode="TA-0", resolution=0.0, scales=scale_grid, tau=tau, delta=delta
    )
    for scale in scale_grid:
        result = evaluate_config(base_model, experts, uniform, scale, evaluator, device)
        result.feasible = check_constraints(result, tau, delta)
        output.all_results.append(result)
    output.n_configurations = len(output.all_results)
    output.n_feasible = sum(result.feasible for result in output.all_results)
    output.best = _select_best(output.all_results)
    return output


def _generate_refined_grid(
    center: Tuple[float, ...],
    n_experts: int,
    resolution: float,
    radius: float,
) -> List[Tuple[float, ...]]:
    """Generate exact-simplex points near a coarse optimum."""
    _validate_resolution(resolution)
    if radius < 0 or not math.isfinite(radius):
        raise ValueError("refine_radius must be finite and non-negative")
    candidates = generate_simplex_grid(n_experts, resolution)
    return [
        point
        for point in candidates
        if max(abs(point[i] - center[i]) for i in range(n_experts)) <= radius + 1e-12
    ]


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
    """TA-1: simplex search over λ plus independent α."""
    n_experts = _validate_experts(experts)
    validate_parameter_names(experts, base_model)
    scale_grid = generate_scale_grid(scales)
    lambda_grid = generate_simplex_grid(n_experts, resolution)
    output = SearchResult(
        mode="TA-1", resolution=resolution, scales=scale_grid, tau=tau, delta=delta
    )

    def run(points: Sequence[Tuple[float, ...]]) -> None:
        for lambdas in points:
            for scale in scale_grid:
                result = evaluate_config(
                    base_model, experts, lambdas, scale, evaluator, device
                )
                result.feasible = check_constraints(result, tau, delta)
                output.all_results.append(result)

    run(lambda_grid)
    stage_one_best = _select_best(output.all_results)
    if refine_around_best and stage_one_best is not None:
        refined = _generate_refined_grid(
            stage_one_best.lambdas,
            n_experts,
            refine_resolution,
            refine_radius,
        )
        coarse_keys = {
            (tuple(round(v, 12) for v in r.lambdas), round(r.scale, 12))
            for r in output.all_results
        }
        for lambdas in refined:
            for scale in scale_grid:
                key = (tuple(round(v, 12) for v in lambdas), round(scale, 12))
                if key in coarse_keys:
                    continue
                result = evaluate_config(
                    base_model, experts, lambdas, scale, evaluator, device
                )
                result.feasible = check_constraints(result, tau, delta)
                output.all_results.append(result)

    output.n_configurations = len(output.all_results)
    output.n_feasible = sum(r.feasible for r in output.all_results)
    output.best = _select_best(output.all_results)
    return output


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
    """TA-2: search λ, α and γ with exact Fisher weighting.

    Curvature/configuration failures deliberately propagate. A missing Fisher
    tensor is not equivalent to a legitimately poor candidate.
    """
    n_experts = _validate_experts(experts)
    validate_parameter_names(experts, base_model)
    scale_grid = generate_scale_grid(scales)
    lambdas_grid = generate_simplex_grid(n_experts, resolution)
    gammas = list(DEFAULT_GAMMAS if gamma_grid is None else gamma_grid)
    if not gammas:
        raise ValueError("gamma_grid must not be empty")
    gammas = [float(gamma) for gamma in gammas]
    if not all(math.isfinite(gamma) and gamma >= 0 for gamma in gammas):
        raise ValueError("gamma values must be finite and non-negative")

    output = SearchResult(
        mode="TA-2",
        resolution=resolution,
        scales=scale_grid,
        gamma_grid=gammas,
        tau=tau,
        delta=delta,
    )
    for lambdas in lambdas_grid:
        for scale in scale_grid:
            for gamma in gammas:
                config = MergeConfig(
                    method=MergeMethod.FISHER_DENSE,
                    task_scale=scale,
                    lambdas=lambdas,
                    fisher_gamma=gamma,
                )
                merged = merge_fisher_task_arithmetic(
                    base_model,
                    experts,
                    config,
                    curvature_bank,
                    device=device,
                )
                result = _metrics_to_evaluation(
                    evaluator(merged.merged_model), lambdas, scale, gamma
                )
                result.feasible = check_constraints(result, tau, delta)
                output.all_results.append(result)

    output.n_configurations = len(output.all_results)
    output.n_feasible = sum(r.feasible for r in output.all_results)
    output.best = _select_best(output.all_results)
    return output


def _cpu_task_vectors_no_mutation(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
    base_params = {
        name: p.detach().float().cpu()
        for name, p in base_model.named_parameters()
    }
    task_vectors: List[Dict[str, Tensor]] = []
    for expert in experts:
        params = dict(expert.named_parameters())
        task_vectors.append({
            name: params[name].detach().float().cpu() - base_p
            for name, base_p in base_params.items()
        })
    return base_params, task_vectors


def _family_solution_hash(
    family_lambdas: Dict[str, Tuple[float, ...]],
    family_scales: Dict[str, float],
) -> str:
    payload = {
        "family_lambdas": {
            family: [float(v) for v in family_lambdas[family]]
            for family in sorted(family_lambdas)
        },
        "family_scales": {
            family: float(family_scales[family])
            for family in sorted(family_scales)
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _merge_family_weighted(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    family_lambdas: Dict[str, Tuple[float, ...]],
    family_scales: Dict[str, float],
    param_families: Dict[str, str],
    device: str = "cpu",
) -> MergeResult:
    """Build θ*_{f,k} = θ₀ + α_f Σᵢ w_{i,f} Δᵢ without input mutation."""
    n_experts = _validate_experts(experts)
    validate_parameter_names(experts, base_model)
    base_params, task_vectors = _cpu_task_vectors_no_mutation(base_model, experts)
    merged = copy.deepcopy(base_model).cpu()

    with torch.no_grad():
        for name, out_param in merged.named_parameters():
            family = param_families.get(name, "other")
            lambdas = family_lambdas.get(
                family, tuple(1.0 / n_experts for _ in range(n_experts))
            )
            if len(lambdas) != n_experts:
                raise ValueError(
                    f"family {family!r} has {len(lambdas)} lambdas for {n_experts} experts"
                )
            scale = float(family_scales.get(family, 0.0))
            if scale == 0.0:
                out_param.copy_(base_params[name])
                continue
            delta = torch.zeros_like(base_params[name])
            for index, task_vector in enumerate(task_vectors):
                delta.add_(task_vector[name], alpha=float(lambdas[index]))
            out_param.copy_(base_params[name] + scale * delta)

    merged.to(device)
    solution_hash = _family_solution_hash(family_lambdas, family_scales)
    config = MergeConfig(method=MergeMethod.TASK_ARITHMETIC)
    trace = OperatorTrace(
        method="family_task_arithmetic",
        operators=["FAMILY_TASK_ARITHMETIC"],
        task_scale=1.0,
        lambdas=[],
        config_hash=solution_hash,
    )
    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method="family_task_arithmetic",
    )


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
    """TA-3: greedy per-family λ/α refinement with complete provenance."""
    n_experts = _validate_experts(experts)
    validate_parameter_names(experts, base_model)
    scale_grid = generate_scale_grid(scales)
    lambda_grid = generate_simplex_grid(n_experts, resolution)
    uniform = tuple(1.0 / n_experts for _ in range(n_experts))
    frozen = tuple(DEFAULT_FROZEN_FAMILIES if frozen_families is None else frozen_families)
    unknown = sorted(set(frozen) - set(FINE_FAMILIES))
    if unknown:
        raise ValueError(f"unknown frozen families: {unknown}")

    param_families = {
        name: classify_parameter_family_fine(name)
        for name, _ in base_model.named_parameters()
    }
    present = [family for family in FINE_FAMILIES if family in set(param_families.values())]
    family_lambdas = {family: uniform for family in present}
    family_scales = {
        family: (0.0 if family in frozen else 1.0)
        for family in present
    }

    output = SearchResult(
        mode="TA-3",
        resolution=resolution,
        scales=scale_grid,
        tau=tau,
        delta=delta,
        family_order=list(present),
    )

    for family in [f for f in present if f not in frozen]:
        family_candidates: List[Tuple[EvaluationResult, Tuple[float, ...], float]] = []
        for lambdas in lambda_grid:
            for scale in scale_grid:
                candidate_lambdas = dict(family_lambdas)
                candidate_lambdas[family] = lambdas
                candidate_scales = dict(family_scales)
                candidate_scales[family] = scale
                merged = _merge_family_weighted(
                    base_model,
                    experts,
                    candidate_lambdas,
                    candidate_scales,
                    param_families,
                    device,
                )
                evaluation = _metrics_to_evaluation(
                    evaluator(merged.merged_model), lambdas, scale
                )
                evaluation.feasible = check_constraints(evaluation, tau, delta)
                output.all_results.append(evaluation)
                family_candidates.append((evaluation, lambdas, scale))

        best_eval = _select_best([item[0] for item in family_candidates])
        if best_eval is None:
            raise RuntimeError(f"TA-3 produced no candidates for family {family!r}")
        # Match by object identity because each evaluation is unique.
        selected = next(item for item in family_candidates if item[0] is best_eval)
        family_lambdas[family] = selected[1]
        family_scales[family] = selected[2]

    final_merge = _merge_family_weighted(
        base_model,
        experts,
        family_lambdas,
        family_scales,
        param_families,
        device,
    )
    final_metrics = evaluator(final_merge.merged_model)
    representative_family = next(
        (family for family in present if family not in frozen),
        present[0] if present else "other",
    )
    final_eval = _metrics_to_evaluation(
        final_metrics,
        family_lambdas.get(representative_family, uniform),
        family_scales.get(representative_family, 0.0),
    )
    final_eval.feasible = check_constraints(final_eval, tau, delta)
    output.all_results.append(final_eval)
    output.best = final_eval
    output.n_configurations = len(output.all_results)
    output.n_feasible = sum(r.feasible for r in output.all_results)
    output.family_lambdas = dict(family_lambdas)
    output.family_scales = dict(family_scales)
    output.solution_hash = _family_solution_hash(family_lambdas, family_scales)
    return output


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
    """Unified production TA search entry point."""
    normalized = mode.upper()
    if normalized == "TA-0":
        return search_ta0(base_model, experts, evaluator, scales, tau, delta, device)
    if normalized == "TA-1":
        return search_ta1(
            base_model,
            experts,
            evaluator,
            resolution,
            scales,
            tau,
            delta,
            device,
            **kwargs,
        )
    if normalized == "TA-2":
        if curvature_bank is None:
            raise ValueError("TA-2 requires curvature_bank")
        return search_ta2(
            base_model,
            experts,
            evaluator,
            curvature_bank,
            resolution,
            scales,
            tau=tau,
            delta=delta,
            device=device,
            **kwargs,
        )
    if normalized == "TA-3":
        return search_ta3(
            base_model,
            experts,
            evaluator,
            resolution,
            scales,
            tau,
            delta,
            device,
            **kwargs,
        )
    raise ValueError(f"unknown search mode {mode!r}; expected TA-0, TA-1, TA-2, or TA-3")
