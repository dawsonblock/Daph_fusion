"""
Comprehensive Merge Baseline Suite (Phase 10).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from daph_exfusion.geometry.normalization import normalize_layerwise_task_vectors
from daph_hybrid_exfusion_v2_3 import (
    MergeMode,
    extract_task_vectors,
    merge_expert_family,
)


def run_baseline_merge(
    base_model: nn.Module,
    experts: List[nn.Module],
    merge_method: str,
    lambdas: List[float],
    calibration_batch: Optional[Any] = None,
) -> nn.Module:
    target = copy.deepcopy(base_model)
    weights = torch.tensor(lambdas, dtype=torch.float32)

    if merge_method == "parameter_average":
        merge_expert_family(
            experts,
            base_model,
            weights,
            policies={"merge_mode": MergeMode.PARAMETER_AVERAGE.value},
            apply_to=target,
        )
    elif merge_method == "raw_task_arithmetic":
        merge_expert_family(
            experts,
            base_model,
            weights,
            policies={"merge_mode": MergeMode.TASK_ARITHMETIC.value},
            apply_to=target,
        )
    elif merge_method == "layer_normalized_arithmetic":
        task_vecs = [extract_task_vectors(e, base_model) for e in experts]
        norm_task_vecs = normalize_layerwise_task_vectors(task_vecs, base_model)
        # Apply normalized task vectors
        with torch.no_grad():
            for name, param in target.named_parameters():
                if param.requires_grad and name in norm_task_vecs[0]:
                    delta_sum = sum(
                        w * norm_task_vecs[i][name] for i, w in enumerate(lambdas)
                    )
                    param.add_(delta_sum.to(param.dtype))
    elif merge_method == "ties":
        merge_expert_family(
            experts,
            base_model,
            weights,
            policies={"dare_base_p": 0.0, "ties_trim_ratio": 0.2},
            apply_to=target,
        )
    elif merge_method == "dare_ties":
        merge_expert_family(
            experts,
            base_model,
            weights,
            policies={"dare_base_p": 0.2, "ties_trim_ratio": 0.2},
            apply_to=target,
        )
    elif merge_method == "fisher":
        merge_expert_family(
            experts,
            base_model,
            weights,
            calibration_batch=calibration_batch,
            policies={"ties_fisher_blend": 1.0},
            apply_to=target,
        )
    elif merge_method == "exfusion_v2_3":
        merge_expert_family(
            experts,
            base_model,
            weights,
            calibration_batch=calibration_batch,
            policies={
                "dare_base_p": 0.2,
                "ties_trim_ratio": 0.2,
                "ties_fisher_blend": 0.5,
            },
            apply_to=target,
        )
    else:
        raise ValueError(f"Unknown merge method: {merge_method}")

    return target
