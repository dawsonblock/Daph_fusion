"""Canonical merge surface for DAPH ExFusion v3.

Production merging is Task Arithmetic first:
    TA-0 uniform
    TA-1 expert-weighted
    TA-2 Fisher-weighted
    TA-3 architecture-family weighted

Legacy sparse algorithms are controlled baselines. Experimental algorithms stay
isolated under ``daph_exfusion.experimental``.
"""
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
    classify_parameter_family_fine,
    get_layer_index,
    count_layers,
    validate_ssm_stability,
    FINE_FAMILIES,
)
from daph_exfusion.merge.fisher_dense import MissingCurvatureError
from daph_exfusion.merge.pipeline_v3 import merge_experts as merge_experts_v3
from daph_exfusion.merge.fisher_task_arithmetic import merge_fisher_task_arithmetic
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

# Backward-compatible v2.5 API. Kept only because legacy tests and research
# comparisons still import these names.
from daph_exfusion.merge.pipeline import (
    ExpertMergeState,
    MergeConfig,
    MergeResult,
    merge_experts,
)

__all__ = [
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
    "merge_experts_v3",
    "merge_fisher_task_arithmetic",
    "search_task_arithmetic",
    "search_ta0",
    "search_ta1",
    "search_ta2",
    "search_ta3",
    "generate_simplex_grid",
    "generate_scale_grid",
    "EvaluationResult",
    "SearchResult",
    "ExpertMergeState",
    "MergeConfig",
    "MergeResult",
    "merge_experts",
]
