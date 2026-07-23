"""
Hierarchical geometry analysis across global, layer, and block levels (Phases 6-8).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from daph_exfusion.geometry.descriptors import (
    compute_cosine_similarity,
    compute_l2_norm,
    compute_rms_norm,
    compute_sign_conflict_ratio,
    compute_support_overlap,
)


def compute_global_geometry(
    task_vectors: List[Dict[str, Tensor]],
) -> Dict[str, Any]:
    flat_vectors = [
        torch.cat([p.flatten() for p in tv.values()]) for tv in task_vectors
    ]

    norms = [compute_l2_norm(v) for v in flat_vectors]
    rms_norms = [compute_rms_norm(v) for v in flat_vectors]

    pairwise_cosine = {}
    pairwise_sign = {}
    pairwise_support = {}

    num_experts = len(task_vectors)
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            key = f"{i}_vs_{j}"
            pairwise_cosine[key] = compute_cosine_similarity(
                flat_vectors[i], flat_vectors[j]
            )
            pairwise_sign[key] = compute_sign_conflict_ratio(
                flat_vectors[i], flat_vectors[j]
            )
            pairwise_support[key] = compute_support_overlap(
                flat_vectors[i], flat_vectors[j]
            )

    max_norm = max(norms) if norms else 1.0
    min_norm = min(norms) if norms else 1.0
    norm_ratio = max_norm / max(min_norm, 1e-8)

    return {
        "l2_norms": norms,
        "rms_norms": rms_norms,
        "pairwise_cosine": pairwise_cosine,
        "pairwise_sign_conflict": pairwise_sign,
        "pairwise_support_overlap": pairwise_support,
        "norm_ratio": norm_ratio,
    }
