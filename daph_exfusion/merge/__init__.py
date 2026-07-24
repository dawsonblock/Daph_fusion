"""Canonical merge pipeline for DAPH ExFusion v3.

Single entry point for all model merging. Every experiment, AGX candidate,
CLI, and integration test must go through ``merge_experts`` so that algorithm
names and implementations never diverge.

v3 dense merge trunk:
    task_arithmetic → fisher_dense → fisher_base_anchored → regmean →
    regmean_pp → coefficient_opt → trust_region → kfac_barycenter → agx

Legacy benchmark methods (controlled baselines):
    dare, ties_magnitude, ties_majority, dare_ties, emr, model_stock, slerp
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
    extract_task_vectors,
    validate_parameter_names,
    classify_parameter_family,
    get_layer_index,
    count_layers,
    validate_ssm_stability,
)
# v3 pipeline (single entry point for dense methods)
from daph_exfusion.merge.pipeline_v3 import merge_experts as merge_experts_v3

# Backward-compatible v2.5 API (legacy tests depend on these)
from daph_exfusion.merge.pipeline import (
    ExpertMergeState,
    MergeConfig,
    MergeResult,
    merge_experts,
)

__all__ = [
    # v3
    "ExpertSpec",
    "MergeConfigV3",
    "MergeMethod",
    "MergeResultV3",
    "OperatorTrace",
    "CoefficientGranularity",
    "CoefficientParameterization",
    "FisherStabilization",
    "RegMeanMode",
    "merge_experts_v3",
    "extract_task_vectors",
    "validate_parameter_names",
    "classify_parameter_family",
    "get_layer_index",
    "count_layers",
    "validate_ssm_stability",
    # v2.5 backward-compatible
    "ExpertMergeState",
    "MergeConfig",
    "MergeResult",
    "merge_experts",
]
