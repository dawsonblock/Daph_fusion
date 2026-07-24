"""AGX operator mathematical contracts (Phase 11).

Each operator has a precise mathematical definition. No operator is a stub.

Operators:
  RAW:        T(Δ) = Δ
  NORMALIZED: T(Δ) = Δ * s_l / (||Δ||_F + ε)
  DARE:       T(Δ) = (M ⊙ Δ) / (1-p),  M ~ Bernoulli(1-p)
  DELTA_DROPOUT: T(Δ) = M ⊙ Δ  (no rescaling; distinct from DARE)
  TIES:       Cross-expert: Trim → Elect → Merge (NOT a single-expert op)
  FISHER:     Layerwise Fisher-weighted merge: w_{i,l} = F_{i,l}^γ / Σ_j F_{j,l}^γ
  PROJECT:    Interference removal: Δ' = Δ - U U^T Δ  (idempotent)

TIES and FISHER are CROSS-EXPERT operators: they operate on the full set
of expert deltas for a layer, not a single delta. They must be invoked
through `transform_expert_set`, not `_transform_delta`.

Single-expert operators (RAW, NORMALIZED, DARE, DELTA_DROPOUT, PROJECT)
can be applied per-expert via `transform_single_delta`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Single-expert operators
# =============================================================================


def op_raw(delta: Tensor) -> Tensor:
    """RAW: T(Δ) = Δ (identity)."""
    return delta


def op_normalized(delta: Tensor, target_scale: float = 1.0, eps: float = 1e-8) -> Tensor:
    """NORMALIZED: T(Δ) = Δ * s_l / (||Δ||_F + ε).

    Scales the delta to have Frobenius norm = target_scale.
    """
    norm = delta.norm(p="fro")
    if norm.item() < eps:
        return delta
    return delta * (target_scale / (norm + eps))


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


def op_delta_dropout(
    delta: Tensor,
    drop_probability: float = 0.2,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Delta dropout (NOT DARE): T(Δ) = M ⊙ Δ (no rescaling)."""
    p = min(max(drop_probability, 0.0), 0.99)
    if p <= 0.0:
        return delta
    keep_mask = (
        torch.rand(delta.shape, generator=generator, device=delta.device) >= p
    ).to(delta.dtype)
    return delta * keep_mask


def op_project(
    delta: Tensor,
    conflict_subspace: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """PROJECT: interference removal.

    Δ' = Δ - U U^T Δ

    where U is the conflict subspace (orthonormal columns). This removes
    the component of Δ that lies in the conflict subspace.

    If conflict_subspace is None, returns Δ unchanged (no projection).
    The subspace must be provided by the caller (computed from cross-expert
    conflict analysis). This is idempotent: projecting twice gives the same
    result (since U^T Δ' = U^T Δ - U^T U U^T Δ = U^T Δ - U^T Δ = 0).
    """
    if conflict_subspace is None:
        return delta
    # conflict_subspace: [..., k, d] orthonormal columns
    # delta: [..., d] flattened or [..., a, b]
    original_shape = delta.shape
    if delta.dim() > 2:
        delta_flat = delta.reshape(-1, delta.shape[-1])
    else:
        delta_flat = delta

    # U: [d, k] (columns are orthonormal basis vectors)
    U = conflict_subspace.to(delta_flat.device, delta_flat.dtype)
    # projection = U U^T delta
    proj = U @ (U.t() @ delta_flat.t())  # [d, N]
    proj = proj.t()  # [N, d]
    result = delta_flat - proj

    return result.reshape(original_shape)


# =============================================================================
# Cross-expert operators
# =============================================================================


def op_ties(
    deltas: List[Tensor],
    trim_fraction: float = 0.2,
) -> Tensor:
    """TIES: Trim → Elect → Merge.

    1. Trim: zero out the smallest-magnitude trim_fraction of each delta
    2. Elect: for each parameter, choose the sign with greater total magnitude
    3. Merge: average only the deltas whose sign matches the elected sign

    Args:
        deltas: List of N tensors, each [..., a, b] (same shape)
        trim_fraction: Fraction of smallest-magnitude elements to trim (0-1)

    Returns:
        Merged tensor of the same shape as each input delta.
    """
    if not deltas:
        raise ValueError("TIES requires at least one delta")

    N = len(deltas)
    stacked = torch.stack(deltas, dim=0)  # [N, ...]

    # 1. Trim: zero out smallest-magnitude elements per delta
    # (applies even for N=1; election/merge are trivial for N=1)
    trim = min(max(trim_fraction, 0.0), 0.99)
    if trim > 0.0:
        trimmed = []
        for i in range(N):
            d = deltas[i]
            flat = d.abs().flatten()
            k = int(flat.numel() * trim)
            if k > 0 and k < flat.numel():
                threshold = torch.kthvalue(flat, k).values
                d = torch.where(d.abs() > threshold, d, torch.zeros_like(d))
            trimmed.append(d)
        stacked = torch.stack(trimmed, dim=0)

    if N == 1:
        return stacked[0]

    # 2. Elect: sign with greater total magnitude
    total_pos = (stacked * (stacked > 0).to(stacked.dtype)).sum(dim=0)
    total_neg = (-stacked * (stacked < 0).to(stacked.dtype)).sum(dim=0)
    elected_sign = torch.where(total_pos >= total_neg, 1.0, -1.0)

    # 3. Merge: average only matching-sign entries
    matching = (stacked * elected_sign.unsqueeze(0) > 0).to(stacked.dtype)
    count = matching.sum(dim=0).clamp(min=1)
    merged = (stacked * matching).sum(dim=0) / count

    return merged


def op_fisher_weighted(
    deltas: List[Tensor],
    fisher_diagonals: List[Tensor],
    gamma: float = 0.5,
    eps: float = 1e-8,
) -> Tensor:
    """Fisher-weighted layerwise merge:

        w_{i,l,k} = F_{i,l,k}^γ / (Σ_j F_{j,l,k}^γ + ε)
        Δ*_l = Σ_i w_{i,l} ⊙ Δ_{i,l}

    Higher Fisher (more important parameter) gets higher weight.
    """
    if not deltas:
        raise ValueError("Fisher merge requires at least one delta")
    if len(deltas) != len(fisher_diagonals):
        raise ValueError(
            f"deltas count {len(deltas)} != fisher count {len(fisher_diagonals)}"
        )
    if len(deltas) == 1:
        return deltas[0]

    N = len(deltas)
    # Compute weights: w_i = F_i^γ / Σ_j F_j^γ
    powered = [f.clamp(min=0).pow(gamma) for f in fisher_diagonals]
    total = sum(powered) + eps
    weights = [p / total for p in powered]

    # Weighted sum
    result = sum(w * d for w, d in zip(weights, deltas))
    return result


# =============================================================================
# Normalization operators (Phase 10)
# =============================================================================


def normalize_raw(delta: Tensor) -> Tensor:
    """RAW normalization: no change."""
    return delta


def normalize_unit_frobenius(delta: Tensor, eps: float = 1e-8) -> Tensor:
    """UNIT_FROBENIUS: scale to unit Frobenius norm."""
    norm = delta.norm(p="fro")
    if norm.item() < eps:
        return delta
    return delta / (norm + eps)


def normalize_base_relative(
    delta: Tensor,
    base_param: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """BASE_RELATIVE: r_{i,l} = ||Δ_{i,l}||_F / (||θ_{0,l}||_F + ε).

    Returns a SCALAR descriptor (the relative magnitude), not a rescaled
    delta. Used as a feature for AGX, not as a direct transform.
    """
    delta_norm = delta.norm(p="fro")
    base_norm = base_param.norm(p="fro")
    return delta_norm / (base_norm + eps)


def normalize_median_expert_relative(
    delta: Tensor,
    all_delta_norms: List[float],
    eps: float = 1e-8,
) -> Tensor:
    """MEDIAN_EXPERT_RELATIVE: q_{i,l} = ||Δ_{i,l}||_F / (median_j ||Δ_{j,l}||_F + ε).

    Returns a SCALAR descriptor. Used as a feature for AGX.
    """
    if not all_delta_norms:
        return delta
    median_norm = torch.tensor(all_delta_norms).median().item()
    delta_norm = delta.norm(p="fro")
    return delta_norm / (median_norm + eps)


def normalize_clipped_norm(
    delta: Tensor,
    max_norm: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """CLIPPED_NORM: scale down if norm exceeds max_norm, else unchanged."""
    norm = delta.norm(p="fro")
    if norm.item() <= max_norm + eps:
        return delta
    return delta * (max_norm / (norm + eps))


# =============================================================================
# Dispatch functions
# =============================================================================


SINGLE_EXPERT_OPS = {"RAW", "NORMALIZED", "DARE", "DELTA_DROPOUT", "PROJECT"}
CROSS_EXPERT_OPS = {"TIES", "FISHER"}


def transform_single_delta(
    delta: Tensor,
    operator: str,
    generator: Optional[torch.Generator] = None,
    dare_drop: float = 0.2,
    target_scale: float = 1.0,
    conflict_subspace: Optional[Tensor] = None,
) -> Tensor:
    """Apply a single-expert operator to one delta."""
    op = operator.upper()
    if op == "RAW":
        return op_raw(delta)
    if op == "NORMALIZED":
        return op_normalized(delta, target_scale=target_scale)
    if op == "DARE":
        return op_dare(delta, drop_probability=dare_drop, generator=generator)
    if op == "DELTA_DROPOUT":
        return op_delta_dropout(delta, drop_probability=dare_drop, generator=generator)
    if op == "PROJECT":
        return op_project(delta, conflict_subspace=conflict_subspace)
    raise ValueError(
        f"Single-expert operator '{operator}' not recognized. "
        f"Cross-expert operators (TIES, FISHER) must use transform_expert_set()."
    )


def transform_expert_set(
    deltas: List[Tensor],
    operator: str,
    fisher_diagonals: Optional[List[Tensor]] = None,
    trim_fraction: float = 0.2,
    fisher_gamma: float = 0.5,
    generator: Optional[torch.Generator] = None,
    dare_drop: float = 0.2,
    target_scale: float = 1.0,
    conflict_subspace: Optional[Tensor] = None,
) -> Tensor:
    """Apply a cross-expert operator to the full set of deltas for a layer.

    For single-expert operators (RAW, NORMALIZED, DARE, PROJECT), this
    applies the operator to each delta independently and returns the sum.
    For cross-expert operators (TIES, FISHER), it applies the joint
    operation and returns a single merged delta.
    """
    op = operator.upper()
    if op == "TIES":
        return op_ties(deltas, trim_fraction=trim_fraction)
    if op == "FISHER":
        if fisher_diagonals is None:
            raise ValueError("FISHER operator requires fisher_diagonals")
        return op_fisher_weighted(deltas, fisher_diagonals, gamma=fisher_gamma)
    # Single-expert ops: apply per-delta and sum
    transformed = [
        transform_single_delta(
            d, op, generator=generator, dare_drop=dare_drop,
            target_scale=target_scale, conflict_subspace=conflict_subspace,
        )
        for d in deltas
    ]
    return sum(transformed)
