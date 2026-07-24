"""Spectral diagnostics for task vectors (Phase 17).

For each matrix delta ΔW = UΣVᵀ, compute:
    - rank for 50%, 80%, 90%, 95% explained energy
    - effective rank (entropy-based)
    - spectral entropy

This is a GATE: subspace merging is only enabled when r₉₀ < 0.1 * min(m, n).
If most layers require high rank, the subspace branch is killed.

Do not spend weeks doing SVD because it sounds elegant. Measure first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class SpectralDiagnostics:
    """Spectral analysis of a task vector delta."""
    rank_50: int       # rank for 50% energy
    rank_80: int       # rank for 80% energy
    rank_90: int       # rank for 90% energy
    rank_95: int       # rank for 95% energy
    effective_rank: float  # entropy-based effective rank
    spectral_entropy: float
    total_energy: float
    max_singular_value: float
    min_singular_value: float
    num_singular_values: int
    matrix_shape: tuple

    @property
    def min_dim(self) -> int:
        return min(self.matrix_shape)

    @property
    def spectral_gate_passes(self) -> bool:
        """True if r90 < 0.1 * min(m, n) — subspace merging is justified."""
        return self.rank_90 < 0.1 * self.min_dim


def compute_spectral_diagnostics(delta: Tensor) -> SpectralDiagnostics:
    """Compute spectral diagnostics for a task vector delta.

    ΔW = UΣVᵀ

    Explained energy: E(r) = Σ_{j=1}^{r} σⱼ² / Σ_j σⱼ²

    Effective rank: exp(H) where H = -Σ pⱼ log(pⱼ), pⱼ = σⱼ/Σσ
    """
    if delta.dim() != 2:
        # For non-2D tensors, flatten to 2D
        if delta.dim() == 1:
            delta_2d = delta.unsqueeze(0)
        else:
            delta_2d = delta.reshape(-1, delta.shape[-1])
    else:
        delta_2d = delta

    # SVD
    try:
        s = torch.linalg.svdvals(delta_2d.float())
    except Exception:
        # Fallback for numerical issues
        s = torch.linalg.svdvals(delta_2d.double()).float()

    s = s.clamp(min=0)  # Ensure non-negative
    total_energy = (s ** 2).sum().item()

    if total_energy < 1e-12:
        return SpectralDiagnostics(
            rank_50=0, rank_80=0, rank_90=0, rank_95=0,
            effective_rank=0.0, spectral_entropy=0.0,
            total_energy=0.0, max_singular_value=0.0,
            min_singular_value=0.0, num_singular_values=s.numel(),
            matrix_shape=tuple(delta_2d.shape),
        )

    # Cumulative energy
    energy = (s ** 2) / total_energy
    cumsum_energy = torch.cumsum(energy, dim=0)

    # Find ranks for energy thresholds
    def find_rank(threshold: float) -> int:
        idx = (cumsum_energy >= threshold).nonzero()
        return idx[0].item() + 1 if len(idx) > 0 else s.numel()

    rank_50 = find_rank(0.50)
    rank_80 = find_rank(0.80)
    rank_90 = find_rank(0.90)
    rank_95 = find_rank(0.95)

    # Effective rank (entropy-based)
    probs = energy.clamp(min=1e-12)
    spectral_entropy = -(probs * probs.log()).sum().item()
    effective_rank = torch.exp(torch.tensor(spectral_entropy)).item()

    return SpectralDiagnostics(
        rank_50=rank_50,
        rank_80=rank_80,
        rank_90=rank_90,
        rank_95=rank_95,
        effective_rank=effective_rank,
        spectral_entropy=spectral_entropy,
        total_energy=total_energy,
        max_singular_value=s[0].item() if s.numel() > 0 else 0.0,
        min_singular_value=s[-1].item() if s.numel() > 0 else 0.0,
        num_singular_values=s.numel(),
        matrix_shape=tuple(delta_2d.shape),
    )


def spectral_gate_passes(delta: Tensor, threshold: float = 0.1) -> bool:
    """Check if the spectral gate passes for subspace merging.

    Promote subspace merging only when r₉₀ < threshold * min(m, n).
    """
    diag = compute_spectral_diagnostics(delta)
    return diag.rank_90 < threshold * diag.min_dim


def batch_spectral_diagnostics(
    task_vectors: List[Dict[str, Tensor]],
) -> Dict[str, List[SpectralDiagnostics]]:
    """Compute spectral diagnostics for all parameters across all experts.

    Returns:
        Dict mapping param_name -> list of SpectralDiagnostics (one per expert).
    """
    if not task_vectors:
        return {}

    param_names = task_vectors[0].keys()
    results: Dict[str, List[SpectralDiagnostics]] = {}

    for name in param_names:
        expert_diags = []
        for tv in task_vectors:
            if name in tv and tv[name].dim() >= 2:
                expert_diags.append(compute_spectral_diagnostics(tv[name]))
        if expert_diags:
            results[name] = expert_diags

    return results
