"""
Representation-space CKA and Drift Analysis (Phase 9, Phase 23).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def compute_linear_cka(x: Tensor, y: Tensor) -> float:
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    similarity = torch.norm(torch.matmul(x.t(), y)) ** 2
    normalization = torch.norm(torch.matmul(x.t(), x)) * torch.norm(
        torch.matmul(y.t(), y)
    )

    if normalization == 0:
        return 0.0
    return float((similarity / (normalization + 1e-8)).item())
