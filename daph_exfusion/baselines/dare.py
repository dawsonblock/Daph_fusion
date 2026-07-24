"""Legacy DARE operator (v2.5 sparse merge trunk).

DARE: T(Δ) = (M ⊙ Δ) / (1-p),  M ~ Bernoulli(1-p)

This is a sparsification method. In v3 it is retained only as a controlled
baseline for comparison against dense merge methods. It is NOT part of the
mainline dense merge trunk.

See: daph_exfusion/merge/legacy/README.md
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def op_dare(
    delta: Tensor,
    drop_probability: float = 0.2,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-8,
) -> Tensor:
    """DARE: T(Δ) = (M ⊙ Δ) / (1-p).

    The rescaling by 1/(1-p) ensures E[T(Δ)] ≈ Δ.
    """
    p = min(max(drop_probability, 0.0), 0.99)
    if p <= 0.0:
        return delta
    keep_mask = (
        torch.rand(delta.shape, generator=generator, device=delta.device) >= p
    ).to(delta.dtype)
    scale = 1.0 / max(1.0 - p, eps)
    return delta * keep_mask * scale
