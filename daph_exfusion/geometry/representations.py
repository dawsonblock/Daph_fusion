"""
Representation-space CKA and Drift Analysis (Phase 9, Phase 23).

CKA (Centered Kernel Alignment) measures representational similarity between
two activation tensors. The correct observation layout for [B, L, H] hidden
states is TOKEN-observation: each (batch, position) pair is one observation,
yielding a [B*L, H] matrix. The previous implementation used
`x.view(x.size(0), -1)`, which collapses the entire batch into a single
observation -- wrong for B=1 (one row, mean=0, CKA undefined) and wrong in
general (it measures batch-level similarity, not token-level).

This module also:
  - respects padding via an optional attention_mask
  - returns MetricResult(valid=False) for degenerate cases instead of fake 0.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class MetricResult:
    """Result of a representation-space metric (CKA, drift, etc.).

    `valid=False` means the metric could not be computed meaningfully
    (e.g. fewer than 2 observations after masking). Callers MUST check
    `valid` before using `value`; consuming an invalid result's `value`
    (which is None) is a bug.
    """

    valid: bool
    value: Optional[float] = None
    reason: Optional[str] = None


def _flatten_to_token_observations(
    x: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """Reshape [B, L, H] -> [B*L, H] (token-observation layout).

    If attention_mask [B, L] is provided, only active (mask>0) positions are
    kept. If x is already 2-D [N, H], it is returned as-is (after optional
    masking via a 1-D mask).
    """
    if x.dim() == 2:
        if attention_mask is not None:
            mask = attention_mask.reshape(-1).bool()
            if mask.shape[0] != x.shape[0]:
                raise ValueError(
                    f"1-D mask length {mask.shape[0]} != observations {x.shape[0]}"
                )
            return x[mask]
        return x

    if x.dim() != 3:
        raise ValueError(
            f"Expected 2-D [N,H] or 3-D [B,L,H]; got shape {tuple(x.shape)}"
        )

    B, L, H = x.shape
    if attention_mask is not None:
        if attention_mask.shape != (B, L):
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} != {(B, L)}"
            )
        mask = attention_mask.reshape(-1).bool()
        return x.reshape(-1, H)[mask]

    return x.reshape(-1, H)


def compute_linear_cka(
    x: Tensor,
    y: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> MetricResult:
    """Linear CKA between two activation tensors.

    Observation layout: TOKEN-level. For [B, L, H] inputs, each (batch,
    position) is one observation, yielding [B*L, H]. This is the correct
    layout for measuring representational similarity of hidden states.

    Padding: if attention_mask [B, L] is provided, only active positions are
    used as observations.

    Degenerate cases: if fewer than 2 observations remain after masking, or
    if either tensor has zero variance, returns MetricResult(valid=False)
    rather than a misleading 0.0.

    Returns:
        MetricResult with valid=True and value in [0, 1] when computable,
        or valid=False with a reason string otherwise.
    """
    x_obs = _flatten_to_token_observations(x, attention_mask)
    y_obs = _flatten_to_token_observations(y, attention_mask)

    if x_obs.shape[0] != y_obs.shape[0]:
        raise ValueError(
            f"x and y must have the same number of observations; "
            f"got {x_obs.shape[0]} vs {y_obs.shape[0]}"
        )

    n = x_obs.shape[0]
    if n < 2:
        return MetricResult(valid=False, reason="insufficient_observations")

    # Center
    x_c = x_obs - x_obs.mean(dim=0, keepdim=True)
    y_c = y_obs - y_obs.mean(dim=0, keepdim=True)

    xtx = torch.linalg.norm(x_c.t() @ x_c)
    yty = torch.linalg.norm(y_c.t() @ y_c)
    normalization = xtx * yty

    if normalization.item() == 0.0:
        return MetricResult(valid=False, reason="zero_variance")

    similarity = torch.linalg.norm(x_c.t() @ y_c) ** 2
    value = float((similarity / (normalization + 1e-8)).item())
    # CKA is theoretically in [0, 1]; clamp tiny numerical overshoot.
    value = max(0.0, min(1.0, value))
    return MetricResult(valid=True, value=value)


# Backward-compatible helper for callers that want a raw float and accept
# that degenerate cases yield None. Prefer compute_linear_cka directly.
def compute_linear_cka_value(
    x: Tensor,
    y: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Optional[float]:
    """Return CKA value or None if invalid."""
    result = compute_linear_cka(x, y, attention_mask)
    return result.value
