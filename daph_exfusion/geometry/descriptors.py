"""
Task vector geometry descriptors (Phase 6).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def compute_l2_norm(v: Tensor) -> float:
    return float(torch.norm(v, p=2).item())


def compute_rms_norm(v: Tensor) -> float:
    if v.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(v.square())).item())


def compute_cosine_similarity(v1: Tensor, v2: Tensor) -> float:
    dot = torch.sum(v1 * v2).item()
    n1 = torch.norm(v1).item()
    n2 = torch.norm(v2).item()
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def compute_sign_conflict_ratio(v1: Tensor, v2: Tensor) -> float:
    nonzero_mask = (v1 != 0) & (v2 != 0)
    nonzero_count = int(nonzero_mask.sum().item())
    if nonzero_count == 0:
        return 0.0
    conflicts = torch.sign(v1[nonzero_mask]) != torch.sign(v2[nonzero_mask])
    return float(conflicts.sum().item()) / nonzero_count


def compute_support_overlap(v1: Tensor, v2: Tensor) -> float:
    supp1 = v1 != 0
    supp2 = v2 != 0
    intersection = int((supp1 & supp2).sum().item())
    union = int((supp1 | supp2).sum().item())
    if union == 0:
        return 0.0
    return float(intersection) / union
