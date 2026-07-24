"""Legacy TIES operator (v2.5 sparse merge trunk).

TIES: Trim → Elect → Merge (cross-expert sign election).

In v3 it is retained only as a controlled baseline. It is NOT part of the
mainline dense merge trunk.
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor


def _ties_trim(deltas: List[Tensor], trim_fraction: float) -> List[Tensor]:
    """Trim: zero out the smallest-magnitude trim_fraction of each delta."""
    trim = min(max(trim_fraction, 0.0), 0.99)
    if trim <= 0.0:
        return list(deltas)
    trimmed = []
    for d in deltas:
        flat = d.abs().flatten()
        k = int(flat.numel() * trim)
        if k > 0 and k < flat.numel():
            threshold = torch.kthvalue(flat, k).values
            d = torch.where(d.abs() > threshold, d, torch.zeros_like(d))
        trimmed.append(d)
    return trimmed


def op_ties(
    deltas: List[Tensor],
    trim_fraction: float = 0.2,
    sign_mode: str = "magnitude",
) -> Tensor:
    """TIES: Trim → Elect → Merge."""
    if not deltas:
        raise ValueError("TIES requires at least one delta")

    N = len(deltas)
    trimmed = _ties_trim(deltas, trim_fraction)
    stacked = torch.stack(trimmed, dim=0)

    if N == 1:
        return stacked[0]

    if sign_mode == "majority":
        signs = stacked.sign()
        vote_sum = signs.sum(dim=0)
        elected_sign = torch.where(vote_sum > 0, 1.0,
                          torch.where(vote_sum < 0, -1.0, 0.0))
    else:
        total_pos = (stacked * (stacked > 0).to(stacked.dtype)).sum(dim=0)
        total_neg = (-stacked * (stacked < 0).to(stacked.dtype)).sum(dim=0)
        elected_sign = torch.where(total_pos >= total_neg, 1.0, -1.0)

    matching = (stacked * elected_sign.unsqueeze(0) > 0).to(stacked.dtype)
    count = matching.sum(dim=0).clamp(min=1)
    merged = (stacked * matching).sum(dim=0) / count

    return merged
