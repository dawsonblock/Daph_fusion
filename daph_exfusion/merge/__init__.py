"""Canonical merge pipeline for DAPH ExFusion v3.

Single entry point for all model merging. Every experiment, AGX candidate,
CLI, and integration test must go through ``merge_experts`` so that algorithm
names and implementations never diverge.

Production merge modes (TA-0 through TA-3):
    TA-0 (uniform)         — θ* = θ₀ + α (1/N) Σᵢ Δᵢ
    TA-1 (weighted)        — θ* = θ₀ + α Σᵢ λᵢ Δᵢ
    TA-2 (Fisher-weighted) — θ*_k = θ₀ + α Σᵢ wᵢ F_{i,k}^γ Δ_{i,k} / (Σᵢ wᵢ F_{i,k}^γ + ε)
    TA-3 (family-weighted) — θ*_{f,k} = θ₀ + α_f Σᵢ w_{i,f} Δ_{i,k}

Legacy baselines: dare, ties_magnitude, ties_majority, dare_ties
Experimental: regmean, regmean_pp, kfac, surgery, subspace, trust_region,
              coefficient_opt, agx (see daph_exfusion/experimental/)
"""
# v3 canonical types
from daph_exfusion.merge.types import (
    ExpertSpec,
    MergeConfig as MergeConfigV3,
    MergeMethod,
    MergeResult as MergeResultV3,
    OperatorTrace,
    CoefficientGranularity,
    CoefficientParameterization,
    FisherStabilization,
    RegMeanMode,
    MissingCurvatureError,
    extract_task_vectors,
    validate_parameter_names,
    classify_parameter_family,
    classify_parameter_family_fine,
    get_layer_index,
    count_layers,
    validate_ssm_stability,
    FINE_FAMILIES,
)
# v3 pipeline (single entry point)
from daph_exfusion.merge.pipeline_v3 import merge_experts as merge_experts_v3

# Production search
from daph_exfusion.merge.task_search import (
    search_task_arithmetic,
    search_ta0,
    search_ta1,
    search_ta2,
    search_ta3,
    generate_simplex_grid,
    generate_scale_grid,
    EvaluationResult,
    SearchResult,
)

# Backward-compatible v2.5 API (legacy tests depend on these)
from daph_exfusion.merge.pipeline import (
    ExpertMergeState,
    MergeConfig,
    MergeResult,
    merge_experts,
)

__all__ = [
    # v3 types
    "ExpertSpec",
    "MergeConfigV3",
    "MergeMethod",
    "MergeResultV3",
    "OperatorTrace",
    "CoefficientGranularity",
    "CoefficientParameterization",
    "FisherStabilization",
    "RegMeanMode",
    "MissingCurvatureError",
    "extract_task_vectors",
    "validate_parameter_names",
    "classify_parameter_family",
    "classify_parameter_family_fine",
    "get_layer_index",
    "count_layers",
    "validate_ssm_stability",
    "FINE_FAMILIES",
    # v3 pipeline
    "merge_experts_v3",
    # production search
    "search_task_arithmetic",
    "search_ta0",
    "search_ta1",
    "search_ta2",
    "search_ta3",
    "generate_simplex_grid",
    "generate_scale_grid",
    "EvaluationResult",
    "SearchResult",
    # v2.5 backward-compatible
    "ExpertMergeState",
    "MergeConfig",
    "MergeResult",
    "merge_experts",
]
