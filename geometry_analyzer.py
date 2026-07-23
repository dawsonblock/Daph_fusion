from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from daph_hybrid_exfusion_v2_3 import extract_task_vectors


def compute_cosine_similarity(v1: Tensor, v2: Tensor) -> float:
    """Computes cosine similarity between two 1D task vector tensors."""
    dot = torch.sum(v1 * v2).item()
    n1 = torch.norm(v1).item()
    n2 = torch.norm(v2).item()
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def compute_sign_conflict_ratio(v1: Tensor, v2: Tensor) -> float:
    """
    Computes sign conflict ratio:
      C_ab = |{j: sign(v1_j) != sign(v2_j)}| / |{j: v1_j != 0 and v2_j != 0}|
    """
    nonzero_mask = (v1 != 0) & (v2 != 0)
    nonzero_count = int(nonzero_mask.sum().item())
    if nonzero_count == 0:
        return 0.0

    sign_conflict = torch.sign(v1[nonzero_mask]) != torch.sign(v2[nonzero_mask])
    return float(sign_conflict.sum().item()) / nonzero_count


def compute_support_overlap(v1: Tensor, v2: Tensor) -> float:
    """
    Computes support overlap:
      O_ab = |supp(v1) cap supp(v2)| / |supp(v1) cup supp(v2)|
    """
    supp1 = v1 != 0
    supp2 = v2 != 0
    intersection = int((supp1 & supp2).sum().item())
    union = int((supp1 | supp2).sum().item())
    if union == 0:
        return 0.0
    return float(intersection) / union


class TaskVectorGeometryAnalyzer:
    """
    Phase 7 Task Vector Geometry Instrumentation Tool.
    Measures cosine similarity, sign conflict, support overlap, and norm ratio
    across parameter families and expert pairs.
    """

    @staticmethod
    def analyze_expert_pair_geometry(
        tv1: Dict[str, Tensor],
        tv2: Dict[str, Tensor],
    ) -> Dict[str, Any]:
        results = {}

        # 1. Global flattened task vectors
        flat1 = torch.cat([p.flatten() for p in tv1.values()])
        flat2 = torch.cat([p.flatten() for p in tv2.values()])

        results["global"] = {
            "cosine_similarity": compute_cosine_similarity(flat1, flat2),
            "sign_conflict_ratio": compute_sign_conflict_ratio(flat1, flat2),
            "support_overlap": compute_support_overlap(flat1, flat2),
            "norm_ratio": float(
                torch.norm(flat1).item() / max(torch.norm(flat2).item(), 1e-8)
            ),
        }

        # 2. Per-parameter family breakdown
        per_param = {}
        for name in tv1:
            if name in tv2:
                p1 = tv1[name].flatten()
                p2 = tv2[name].flatten()
                per_param[name] = {
                    "cosine_similarity": compute_cosine_similarity(p1, p2),
                    "sign_conflict_ratio": compute_sign_conflict_ratio(p1, p2),
                    "support_overlap": compute_support_overlap(p1, p2),
                    "norm_ratio": float(
                        torch.norm(p1).item() / max(torch.norm(p2).item(), 1e-8)
                    ),
                }
        results["per_parameter"] = per_param
        return results
