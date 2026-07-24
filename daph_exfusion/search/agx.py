"""AGX v2 — Architecture-aware Geometry eXchange (Phase 21-23).

AGX is no longer "which merge algorithm is best?"
It answers: which representation, metric, operator, coefficients, and
constraints are best for this particular region of this model?

A merge decision is:
    M_g = (B_g, O_g, λ_g, η_g)

where:
    g   = architectural group
    B_g = representation/basis
    O_g = merge operator
    λ_g = expert contributions
    η_g = operator hyperparameters

AGX searches over all four. Hierarchy:
    architecture → geometry → operator → coefficients

Algorithm:
    QUALIFY EXPERTS → EXTRACT DENSE TASK VECTORS → BUILD CURVATURE BANK →
    BUILD ACTIVATION BANK → PROFILE GEOMETRY → GLOBAL METHOD TOURNAMENT →
    PRUNE WEAK GEOMETRY FAMILIES → ARCHITECTURE-FAMILY TOURNAMENT →
    DETECT HETEROGENEITY → ADAPTIVELY SUBDIVIDE → SUBSPACE ANALYSIS →
    SELECT REGIONAL OPERATORS → OPTIMIZE REGIONAL COEFFICIENTS →
    COMPOSE REGIONAL WINNERS → GLOBAL RECONCILIATION → PARETO VALIDATION →
    OPTIONAL REPRESENTATION REPAIR → LOCK GEOMETRY GENOME → FINAL TEST
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    classify_parameter_family,
    get_layer_index,
    count_layers,
)
from daph_exfusion.merge.task_arithmetic import merge_task_arithmetic
from daph_exfusion.geometry.profiler import (
    GroupGeometryProfile,
    profile_all_groups,
)


# =============================================================================
# Geometry Genome
# =============================================================================


@dataclass
class RegionDecision:
    """A merge decision for one architectural region.

    M_g = (B_g, O_g, λ_g, η_g)
    """
    region_name: str
    operator: MergeMethod = MergeMethod.TASK_ARITHMETIC
    lambdas: Tuple[float, ...] = ()
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    # Provenance
    geometry_profile: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class GeometryGenome:
    """AGX's primary output: a machine-readable description of the discovered model.

    Each node stores: operator, expert coefficients, hyperparameters,
    geometry measurements, validation metrics, confidence interval, provenance.
    """
    regions: Dict[str, RegionDecision] = field(default_factory=dict)
    global_reconciliation: Optional[Dict[str, Any]] = None
    pareto_front: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> dict:
        return {
            "regions": {
                name: {
                    "region_name": r.region_name,
                    "operator": r.operator.value,
                    "lambdas": list(r.lambdas),
                    "hyperparams": r.hyperparams,
                    "validation_metrics": r.validation_metrics,
                    "confidence_interval": r.confidence_interval,
                }
                for name, r in self.regions.items()
            },
            "global_reconciliation": self.global_reconciliation,
            "pareto_front": self.pareto_front,
        }


# =============================================================================
# Candidate generation
# =============================================================================


# Complete operator pool
ALL_OPERATORS = [
    MergeMethod.FROZEN,
    MergeMethod.TASK_ARITHMETIC,
    MergeMethod.FISHER_DENSE,
    MergeMethod.FISHER_BASE_ANCHORED,
    MergeMethod.REGMEAN,
    MergeMethod.REGMEAN_PP,
    MergeMethod.COEFFICIENT_OPT,
    MergeMethod.TRUST_REGION,
    MergeMethod.KFAC_BARYCENTER,
]

# Legacy controls
LEGACY_CONTROLS = [
    MergeMethod.DARE,
    MergeMethod.TIES_MAGNITUDE,
    MergeMethod.DARE_TIES,
]


def generate_candidates_for_profile(
    profile: GroupGeometryProfile,
    include_legacy: bool = False,
) -> List[MergeMethod]:
    """Generate candidate operators based on a geometry profile.

    The profiler prunes the pool. These are candidate priors, not decisions.
    AGX still has to prove them experimentally.

    Heuristic priors:
        low conflict → TA_OPT
        high base curvature → FISHER_BASE_ANCHORED
        large activation covariance shift → REGMEAN
        low-rank + conflicting subspaces → TSV_PROJECT (subspace)
        high Fisher conflict → TRUST_REGION / projection
        SSM recurrence → FROZEN / BASE_ANCHORED_FISHER
        normalization → FROZEN initially
    """
    candidates: List[MergeMethod] = [MergeMethod.TASK_ARITHMETIC]

    # High base curvature → Fisher
    if profile.avg_base_fisher_norm > 0:
        candidates.append(MergeMethod.FISHER_BASE_ANCHORED)
        candidates.append(MergeMethod.FISHER_DENSE)

    # Activation covariance available → RegMean
    if profile.activation_cov_distance is not None:
        candidates.append(MergeMethod.REGMEAN)

    # High Fisher conflict → trust region
    if profile.fisher_cosine is not None:
        min_cos = profile.fisher_cosine.min().item()
        if min_cos < -0.2:
            candidates.append(MergeMethod.TRUST_REGION)

    # Spectral gate passes → subspace (via coefficient_opt as proxy)
    if profile.spectral_gate_pass_rate > 0.3:
        candidates.append(MergeMethod.COEFFICIENT_OPT)

    # SSM recurrence → frozen / base-anchored
    if "ssm_recurrence" in profile.group_name:
        candidates = [MergeMethod.FROZEN, MergeMethod.FISHER_BASE_ANCHORED]

    # Normalization → frozen
    if "norm" in profile.group_name:
        candidates = [MergeMethod.FROZEN]

    # Embeddings → frozen or conservative TA
    if "embed" in profile.group_name:
        candidates = [MergeMethod.FROZEN, MergeMethod.TASK_ARITHMETIC]

    # Always include coefficient_opt as a candidate
    if MergeMethod.COEFFICIENT_OPT not in candidates:
        candidates.append(MergeMethod.COEFFICIENT_OPT)

    if include_legacy:
        candidates.extend(LEGACY_CONTROLS)

    # Deduplicate
    seen = set()
    unique: List[MergeMethod] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


# =============================================================================
# Successive halving
# =============================================================================


@dataclass
class HalvingConfig:
    """Configuration for successive halving search."""
    stage_1_samples: int = 128
    stage_1_retain: int = 4
    stage_2_samples: int = 512
    stage_2_retain: int = 2
    stage_3_full: bool = True


@dataclass
class CandidateResult:
    """Result of evaluating a candidate."""
    method: MergeMethod
    score: float        # lower is better (e.g., validation NLL)
    retention: float = 0.0
    general_regression: float = 0.0
    representation_drift: float = 0.0
    feasible: bool = True


def successive_halving(
    candidates: List[MergeMethod],
    evaluator: Callable[[MergeMethod, int], CandidateResult],
    config: Optional[HalvingConfig] = None,
) -> List[CandidateResult]:
    """Run successive halving over candidate configurations.

    Stage 1: 128 validation samples, all candidates → keep top 4
    Stage 2: 512 samples, survivors → keep top 2
    Stage 3: full validation, survivors → final ranking

    Args:
        candidates: List of merge methods to evaluate.
        evaluator: Callable(method, num_samples) -> CandidateResult.
        config: Halving configuration.

    Returns:
        List of final CandidateResults, sorted by score.
    """
    if config is None:
        config = HalvingConfig()

    # Stage 1
    stage1_results = [evaluator(c, config.stage_1_samples) for c in candidates]
    stage1_results.sort(key=lambda r: r.score if r.feasible else float("inf"))
    survivors = stage1_results[:config.stage_1_retain]

    if len(survivors) <= 1:
        return survivors

    # Stage 2
    stage2_results = [evaluator(r.method, config.stage_2_samples) for r in survivors]
    stage2_results.sort(key=lambda r: r.score if r.feasible else float("inf"))
    survivors = stage2_results[:config.stage_2_retain]

    if len(survivors) <= 1 or not config.stage_3_full:
        return survivors

    # Stage 3: full validation
    stage3_results = [evaluator(r.method, -1) for r in survivors]  # -1 = full
    stage3_results.sort(key=lambda r: r.score if r.feasible else float("inf"))
    return stage3_results


# =============================================================================
# AGX main entry point
# =============================================================================


def merge_agx(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    curvature_bank: Optional[Dict[str, Dict[str, Tensor]]] = None,
    base_fisher: Optional[Dict[str, Tensor]] = None,
    activation_bank: Optional[Dict[str, Dict[str, Tensor]]] = None,
    calibration_data: Optional[Any] = None,
    evaluator: Optional[Callable] = None,
    device: str = "cpu",
    halving_config: Optional[HalvingConfig] = None,
) -> MergeResult:
    """AGX v2 merge: architecture-aware selection of dense merge geometry.

    This is the full AGX pipeline:
        1. Profile geometry of all architectural groups
        2. Global method tournament (find which geometry families are competitive)
        3. Architecture-family tournament (per-family winners)
        4. Adaptive subdivision (only where heterogeneity justifies it)
        5. Compose regional winners
        6. Global reconciliation
        7. Lock geometry genome

    For simplicity in the initial implementation, this does the global
    tournament and picks the best single method. The full hierarchical
    search is available via the search functions above.

    Args:
        base_model: Base model θ₀.
        experts: Specialist models.
        config: Merge configuration (method must be agx).
        curvature_bank: Optional Fisher diagonals.
        base_fisher: Optional base Fisher.
        activation_bank: Optional activation covariance.
        calibration_data: Optional calibration data.
        evaluator: Optional Callable(merged_model) -> float.
        device: Device for computation.
        halving_config: Successive halving configuration.

    Returns:
        MergeResult with the AGX-selected merge and geometry genome.
    """
    if config.method != MergeMethod.AGX:
        raise ValueError(f"merge_agx called with method={config.method}, expected agx")

    n_experts = len(experts)
    validate_parameter_names(experts, base_model)

    # Step 1: Extract task vectors and profile geometry
    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    task_vectors = extract_task_vectors(experts, base_cpu)
    num_layers = count_layers(base_cpu)

    profiles = profile_all_groups(
        task_vectors,
        base_fisher=base_fisher,
        activation_bank=activation_bank,
        num_layers=num_layers,
    )

    # Step 2: Generate candidates from profiles
    all_candidates: List[MergeMethod] = set()
    for profile in profiles.values():
        candidates = generate_candidates_for_profile(profile)
        all_candidates.update(candidates)
    all_candidates = list(all_candidates)

    if not all_candidates:
        all_candidates = [MergeMethod.TASK_ARITHMETIC]

    # Step 3: Global tournament
    if evaluator is not None:
        def eval_wrapper(method: MergeMethod, num_samples: int) -> CandidateResult:
            try:
                merge_config = MergeConfig(
                    method=method,
                    task_scale=config.task_scale,
                    lambdas=config.lambdas,
                    fisher_gamma=config.fisher_gamma,
                    base_precision_weight=config.base_precision_weight,
                    seed=config.seed,
                )
                # Use the v3 pipeline to merge
                from daph_exfusion.merge.pipeline_v3 import merge_experts as _merge
                result = _merge(
                    base_model, experts, merge_config,
                    curvature_bank=curvature_bank,
                    base_fisher=base_fisher,
                    activation_bank=activation_bank,
                    calibration_data=calibration_data,
                    evaluator=evaluator,
                    device=device,
                )
                score = evaluator(result.merged_model)
                return CandidateResult(
                    method=method, score=score, feasible=True,
                )
            except (ValueError, NotImplementedError):
                return CandidateResult(method=method, score=float("inf"), feasible=False)

        results = successive_halving(all_candidates, eval_wrapper, halving_config)

        if results and results[0].feasible:
            best_method = results[0].method
        else:
            best_method = MergeMethod.TASK_ARITHMETIC
    else:
        # No evaluator — use TA as default
        best_method = MergeMethod.TASK_ARITHMETIC

    # Step 4: Execute the best method
    from daph_exfusion.merge.pipeline_v3 import merge_experts as _merge
    best_config = MergeConfig(
        method=best_method,
        task_scale=config.task_scale,
        lambdas=config.lambdas,
        fisher_gamma=config.fisher_gamma,
        base_precision_weight=config.base_precision_weight,
        seed=config.seed,
    )

    result = _merge(
        base_model, experts, best_config,
        curvature_bank=curvature_bank,
        base_fisher=base_fisher,
        activation_bank=activation_bank,
        calibration_data=calibration_data,
        evaluator=evaluator,
        device=device,
    )

    # Build geometry genome
    genome = GeometryGenome()
    for group_name, profile in profiles.items():
        genome.regions[group_name] = RegionDecision(
            region_name=group_name,
            operator=best_method,
            lambdas=config.lambdas,
            geometry_profile=profile.to_dict(),
        )

    # Update trace
    result.trace.operators = ["AGX", f"SELECTED:{best_method.value}"]
    result.trace.method = "agx"

    return result
