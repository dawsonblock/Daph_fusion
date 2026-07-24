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
    """TIES: Trim → Elect → Merge.

    1. Trim: zero out the smallest-magnitude trim_fraction of each delta
    2. Elect: choose the sign per-coordinate via the specified election mode
    3. Merge: average only the deltas whose sign matches the elected sign

    Args:
        deltas: List of N tensors, each [..., a, b] (same shape)
        trim_fraction: Fraction of smallest-magnitude elements to trim (0-1).
                       trim_fraction=0.2 means remove the bottom 20%.
        sign_mode: "magnitude" (default) — sign with greater total magnitude wins.
                   "majority" — sign held by more experts wins (pure count).

    Returns:
        Merged tensor of the same shape as each input delta.
    """
    if not deltas:
        raise ValueError("TIES requires at least one delta")

    N = len(deltas)
    trimmed = _ties_trim(deltas, trim_fraction)
    stacked = torch.stack(trimmed, dim=0)  # [N, ...]

    if N == 1:
        return stacked[0]

    # 2. Elect
    if sign_mode == "majority":
        # Pure sign counting: s_k = sign(Σ_i sign(Δ_{i,k}))
        signs = stacked.sign()  # +1, -1, or 0
        vote_sum = signs.sum(dim=0)
        elected_sign = torch.where(vote_sum > 0, 1.0,
                          torch.where(vote_sum < 0, -1.0, 0.0))
    else:
        # Magnitude-based: sign with greater total accumulated magnitude
        total_pos = (stacked * (stacked > 0).to(stacked.dtype)).sum(dim=0)
        total_neg = (-stacked * (stacked < 0).to(stacked.dtype)).sum(dim=0)
        elected_sign = torch.where(total_pos >= total_neg, 1.0, -1.0)

    # 3. Merge: average only matching-sign entries (disjoint merge)
    matching = (stacked * elected_sign.unsqueeze(0) > 0).to(stacked.dtype)
    count = matching.sum(dim=0).clamp(min=1)
    merged = (stacked * matching).sum(dim=0) / count

    return merged


def op_ties_fisher(
    deltas: List[Tensor],
    fisher_diagonals: List[Tensor],
    trim_fraction: float = 0.2,
    fisher_gamma: float = 0.5,
    sign_mode: str = "magnitude",
    eps: float = 1e-8,
) -> Tensor:
    """TIES → Fisher-weighted disjoint merge (ExFusion-F core).

    1. Trim each delta independently
    2. Elect sign across experts (magnitude or majority mode)
    3. Among agreeing experts, weight by Fisher curvature: w_i = F_i^γ / Σ_j F_j^γ

    This is the correct composition: TIES resolves conflict, Fisher weights
    importance among the agreeing survivors.
    """
    if not deltas:
        raise ValueError("TIES-Fisher requires at least one delta")
    if len(deltas) != len(fisher_diagonals):
        raise ValueError(
            f"deltas count {len(deltas)} != fisher count {len(fisher_diagonals)}"
        )

    N = len(deltas)
    trimmed = _ties_trim(deltas, trim_fraction)
    stacked = torch.stack(trimmed, dim=0)

    if N == 1:
        return stacked[0]

    # 2. Elect sign
    if sign_mode == "majority":
        signs = stacked.sign()
        vote_sum = signs.sum(dim=0)
        elected_sign = torch.where(vote_sum > 0, 1.0,
                          torch.where(vote_sum < 0, -1.0, 0.0))
    else:
        total_pos = (stacked * (stacked > 0).to(stacked.dtype)).sum(dim=0)
        total_neg = (-stacked * (stacked < 0).to(stacked.dtype)).sum(dim=0)
        elected_sign = torch.where(total_pos >= total_neg, 1.0, -1.0)

    # 3. Fisher-weighted disjoint merge
    matching = (stacked * elected_sign.unsqueeze(0) > 0).to(stacked.dtype)  # [N, ...]
    fisher_stack = torch.stack(
        [f.clamp(min=0).pow(fisher_gamma) for f in fisher_diagonals], dim=0
    )  # [N, ...]
    weighted = stacked * fisher_stack  # [N, ...]
    num = (weighted * matching).sum(dim=0)
    den = (fisher_stack * matching).sum(dim=0) + eps
    return num / den


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
CROSS_EXPERT_OPS = {"TIES", "TIES_MAGNITUDE", "TIES_MAJORITY", "FISHER", "TIES_FISHER", "DARE_TIES", "DARE_TIES_FISHER"}


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
        f"Cross-expert operators (TIES, FISHER, TIES_FISHER, etc.) must use "
        f"transform_expert_set()."
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
    sign_mode: str = "magnitude",
) -> Tensor:
    """Apply a cross-expert operator to the full set of deltas for a layer.

    For single-expert operators (RAW, NORMALIZED, DARE, PROJECT), this
    applies the operator to each delta independently and returns the sum.
    For cross-expert operators (TIES, FISHER, TIES_FISHER, DARE_TIES,
    DARE_TIES_FISHER), it applies the joint operation and returns a single
    merged delta.
    """
    op = operator.upper()
    if op in ("TIES", "TIES_MAGNITUDE"):
        return op_ties(deltas, trim_fraction=trim_fraction, sign_mode="magnitude")
    if op == "TIES_MAJORITY":
        return op_ties(deltas, trim_fraction=trim_fraction, sign_mode="majority")
    if op == "FISHER":
        if fisher_diagonals is None:
            raise ValueError("FISHER operator requires fisher_diagonals")
        return op_fisher_weighted(deltas, fisher_diagonals, gamma=fisher_gamma)
    if op == "TIES_FISHER":
        if fisher_diagonals is None:
            raise ValueError("TIES_FISHER operator requires fisher_diagonals")
        return op_ties_fisher(
            deltas, fisher_diagonals,
            trim_fraction=trim_fraction, fisher_gamma=fisher_gamma,
            sign_mode=sign_mode,
        )
    if op == "DARE_TIES":
        gen = generator or torch.Generator()
        dare_deltas = [op_dare(d, drop_probability=dare_drop, generator=gen) for d in deltas]
        return op_ties(dare_deltas, trim_fraction=trim_fraction, sign_mode=sign_mode)
    if op == "DARE_TIES_FISHER":
        if fisher_diagonals is None:
            raise ValueError("DARE_TIES_FISHER operator requires fisher_diagonals")
        gen = generator or torch.Generator()
        dare_deltas = [op_dare(d, drop_probability=dare_drop, generator=gen) for d in deltas]
        dare_fishers = [f for f in fisher_diagonals]
        return op_ties_fisher(
            dare_deltas, dare_fishers,
            trim_fraction=trim_fraction, fisher_gamma=fisher_gamma,
            sign_mode=sign_mode,
        )
    # Single-expert ops: apply per-delta and sum
    transformed = [
        transform_single_delta(
            d, op, generator=generator, dare_drop=dare_drop,
            target_scale=target_scale, conflict_subspace=conflict_subspace,
        )
        for d in deltas
    ]
    return sum(transformed)
