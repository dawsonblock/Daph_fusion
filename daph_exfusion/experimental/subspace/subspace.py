"""TSV/Subspace Merge module for ExFusion v3.

Subspace-aware merging with a spectral gate.

For each matrix delta ΔW, compute SVD: ΔW = UΣVᵀ.  The spectral gate
determines whether the delta is low-rank enough to benefit from
subspace-aware merging.  If the gate passes, conflicting subspaces
between experts are projected out before merging.  If the gate fails,
the parameter falls back to Task Arithmetic.

Mathematical definitions:
    SVD:                    ΔW = UΣVᵀ,  σ₁ ≥ σ₂ ≥ ... ≥ σ_r > 0
    Explained energy:       E(r) = Σ_{j=1}^{r} σⱼ² / Σ_j σⱼ²
    Effective rank:         r_eff = exp(H),  H = -Σ_j p_j log p_j,
                            p_j = σⱼ² / Σ_j σⱼ²
    Spectral entropy:       H = -Σ_j p_j log p_j
    Spectral gate:          passes  ⟺  r₉₀ < τ · min(m, n),
                            where r₉₀ = min{ r : E(r) ≥ 0.90 }
    Principal angles:       θ_k = arccos(σ_k(U_iᵀ U_j))
    Conflict detection:     direction k conflicts  ⟺  cos(θ_k) > ρ
    Subspace projection:    ΔW_i' = (I - C_i C_iᵀ) ΔW_i
    Subspace merge:         θ* = θ₀ + α Σᵢ λᵢ ΔW_i'
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from daph_exfusion.merge.types import (
    MergeConfig,
    MergeMethod,
    MergeResult,
    OperatorTrace,
    extract_task_vectors,
    validate_parameter_names,
)


# =============================================================================
# Spectral diagnostics
# =============================================================================


def compute_spectral_diagnostics(delta: Tensor) -> dict:
    """Compute spectral diagnostics for a matrix delta via SVD.

    ΔW = UΣVᵀ,  σ₁ ≥ σ₂ ≥ ... ≥ σ_r > 0

    Explained energy:
        E(r) = Σ_{j=1}^{r} σⱼ² / Σ_j σⱼ²

    Effective rank (Shannon entropy of the normalized energy spectrum):
        r_eff = exp(H),  H = -Σ_j p_j log p_j,  p_j = σⱼ² / Σ_j σⱼ²

    Args:
        delta: Matrix ΔW of shape [m, n].

    Returns:
        Dict with keys:
            rank_50 — smallest r with E(r) ≥ 0.50
            rank_80 — smallest r with E(r) ≥ 0.80
            rank_90 — smallest r with E(r) ≥ 0.90
            rank_95 — smallest r with E(r) ≥ 0.95
            effective_rank — exp(spectral_entropy)
            spectral_entropy — H = -Σ_j p_j log p_j
            singular_values — Tensor of singular values σⱼ
    """
    if delta.dim() != 2:
        raise ValueError(
            f"compute_spectral_diagnostics expects a 2D matrix, "
            f"got shape {tuple(delta.shape)}"
        )

    delta_f = delta.detach().float()
    singular_values = torch.linalg.svdvals(delta_f)
    energy = singular_values ** 2
    total_energy = energy.sum().item()

    if total_energy == 0.0:
        return {
            "rank_50": 0,
            "rank_80": 0,
            "rank_90": 0,
            "rank_95": 0,
            "effective_rank": 0.0,
            "spectral_entropy": 0.0,
            "singular_values": singular_values,
        }

    normalized_cumulative = torch.cumsum(energy, dim=0) / total_energy

    def _rank_at(threshold: float) -> int:
        mask = normalized_cumulative >= threshold
        if mask.any():
            return int(mask.nonzero(as_tuple=True)[0][0].item()) + 1
        return len(singular_values)

    p = energy / total_energy
    positive = p > 0
    spectral_entropy = -torch.sum(p[positive] * torch.log(p[positive])).item()
    effective_rank = math.exp(spectral_entropy)

    return {
        "rank_50": _rank_at(0.50),
        "rank_80": _rank_at(0.80),
        "rank_90": _rank_at(0.90),
        "rank_95": _rank_at(0.95),
        "effective_rank": effective_rank,
        "spectral_entropy": spectral_entropy,
        "singular_values": singular_values,
    }


# =============================================================================
# Spectral gate
# =============================================================================


def spectral_gate_passes(delta: Tensor, threshold: float = 0.1) -> bool:
    """Check whether the spectral gate passes for subspace merging.

    The gate passes  ⟺  r₉₀ < τ · min(m, n),  where:
        r₉₀ = min{ r : E(r) ≥ 0.90 }
        E(r) = Σ_{j=1}^{r} σⱼ² / Σ_j σⱼ²
        τ = threshold

    A passing gate indicates the delta is sufficiently low-rank that
    subspace-aware merging is beneficial.

    Args:
        delta: Matrix ΔW of shape [m, n].
        threshold: Gate threshold τ (default 0.1).

    Returns:
        True if r₉₀ < threshold * min(m, n).
    """
    if delta.dim() != 2:
        return False

    m, n = delta.shape
    min_dim = min(m, n)
    if min_dim == 0:
        return False

    diagnostics = compute_spectral_diagnostics(delta)
    r90 = diagnostics["rank_90"]

    return r90 < threshold * min_dim


# =============================================================================
# Subspace merge helpers
# =============================================================================


def _rank_at_energy(singular_values: Tensor, energy_threshold: float) -> int:
    """Smallest r such that E(r) = Σ_{j=1}^{r} σⱼ² / Σ_j σⱼ² ≥ energy_threshold."""
    energy = singular_values ** 2
    total = energy.sum().item()
    if total == 0.0:
        return 0
    normalized_cumulative = torch.cumsum(energy, dim=0) / total
    mask = normalized_cumulative >= energy_threshold
    if mask.any():
        return int(mask.nonzero(as_tuple=True)[0][0].item()) + 1
    return len(singular_values)


def _collect_conflicting_directions(
    svds: List[Tuple[Tensor, Tensor, Tensor]],
    ranks: List[int],
    projection_strength: float,
) -> List[Optional[Tensor]]:
    """Identify and collect conflicting subspace directions for each expert.

    For each pair (i, j), computes principal angles between the top-r
    left singular subspaces via SVD of U_iᵀ U_j.  Directions where
    cos(θ_k) > projection_strength are flagged as conflicting.

    Args:
        svds: List of (U, S, Vh) tuples from torch.linalg.svd.
        ranks: List of truncation ranks r_i per expert.
        projection_strength: Cosine threshold ρ for conflict detection.

    Returns:
        List of tensors (or None) of conflicting directions per expert,
        each of shape [m, k_i].
    """
    n = len(svds)
    conflicting: List[List[Tensor]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            r_i, r_j = ranks[i], ranks[j]
            if r_i == 0 or r_j == 0:
                continue

            U_i = svds[i][0][:, :r_i]
            U_j = svds[j][0][:, :r_j]

            M = U_i.t() @ U_j
            P, cos_angles, Qt = torch.linalg.svd(M, full_matrices=False)

            mask = cos_angles > projection_strength
            if not mask.any():
                continue

            dirs_i = U_i @ P[:, mask]
            dirs_j = U_j @ Qt[mask, :].t()

            conflicting[i].append(dirs_i)
            conflicting[j].append(dirs_j)

    result: List[Optional[Tensor]] = []
    for dirs in conflicting:
        if dirs:
            result.append(torch.cat(dirs, dim=1))
        else:
            result.append(None)
    return result


def _project_out_conflicts(delta: Tensor, conflict_dirs: Optional[Tensor]) -> Tensor:
    """Project out the conflicting subspace from a delta.

    ΔW' = (I - C Cᵀ) ΔW

    where C is an orthonormal basis for the conflicting directions.

    Args:
        delta: Matrix ΔW of shape [m, n].
        conflict_dirs: Matrix of conflicting directions [m, k] or None.

    Returns:
        Projected delta ΔW' with conflicting components removed.
    """
    if conflict_dirs is None or conflict_dirs.shape[1] == 0:
        return delta

    C, _ = torch.linalg.qr(conflict_dirs, mode="reduced")
    return delta - C @ (C.t() @ delta)


def _subspace_merge_param(
    deltas: List[Tensor],
    lambdas: List[float],
    energy_threshold: float,
    projection_strength: float,
) -> Tensor:
    """Merge a single parameter's deltas using subspace projection.

    Steps:
        1. Compute SVD of each ΔW_i = U_i Σ_i V_iᵀ
        2. Determine r_i = rank at energy_threshold:  E(r_i) ≥ energy_threshold
        3. Identify conflicting subspaces (small principal angles):
           cos(θ_k) = σ_k(U_iᵀ U_j) > projection_strength
        4. Project out:  ΔW_i' = (I - C_i C_iᵀ) ΔW_i
        5. Merge:  Σᵢ λᵢ ΔW_i'

    Args:
        deltas: List of raw (unscaled) delta matrices, each [m, n].
        lambdas: Per-expert weights λᵢ.
        energy_threshold: Energy threshold for rank selection.
        projection_strength: Cosine threshold ρ for conflict detection.

    Returns:
        Merged delta for this parameter, shape [m, n].
    """
    n = len(deltas)
    if n == 1:
        return deltas[0] * lambdas[0]

    svds: List[Tuple[Tensor, Tensor, Tensor]] = []
    ranks: List[int] = []

    for delta in deltas:
        U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
        svds.append((U, S, Vh))
        ranks.append(_rank_at_energy(S, energy_threshold))

    conflict_dirs = _collect_conflicting_directions(
        svds, ranks, projection_strength
    )

    merged_delta = torch.zeros_like(deltas[0])
    for i, delta in enumerate(deltas):
        projected = _project_out_conflicts(delta, conflict_dirs[i])
        merged_delta += lambdas[i] * projected

    return merged_delta


# =============================================================================
# Main entry point
# =============================================================================


def merge_subspace(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    device: str = "cpu",
) -> MergeResult:
    """Execute TSV/Subspace merge with a spectral gate.

    For each 2D parameter:
        - Compute SVD diagnostics (rank at 50/80/90/95% energy,
          effective rank, spectral entropy).
        - Spectral gate: pass iff r₉₀ < 0.1 · min(m, n).
        - If gate passes for all experts: project out conflicting
          subspaces and merge in subspace:
              θ* = θ₀ + α Σᵢ λᵢ ΔW_i'
        - If gate fails: fall back to Task Arithmetic:
              θ* = θ₀ + α Σᵢ λᵢ ΔW_i

    For 1D parameters (biases, norms): always use Task Arithmetic.

    Hyperparameters are read from ``config.legacy_sparse``:
        - 'svd_energy_threshold' (default 0.9): energy level for rank
          selection in subspace projection.
        - 'projection_strength' (default 0.9): cosine threshold ρ for
          conflict detection between expert subspaces.

    Args:
        base_model: Base model θ₀.
        experts: Specialist models θᵢ.
        config: Merge configuration.
        device: Device for computation.

    Returns:
        MergeResult with merged model and operator trace recording
        SVD_DIAGNOSTIC (always), SUBSPACE_PROJECT (if any parameter
        used subspace merging), and TASK_ARITHMETIC_FALLBACK (if any
        parameter fell back to Task Arithmetic).
    """
    n_experts = len(experts)
    validate_parameter_names(experts, base_model)

    legacy = config.legacy_sparse or {}
    energy_threshold = float(legacy.get("svd_energy_threshold", 0.9))
    projection_strength = float(legacy.get("projection_strength", 0.9))
    gate_threshold = 0.1

    if config.lambdas:
        lambdas = list(config.lambdas)
        if len(lambdas) != n_experts:
            raise ValueError(
                f"lambdas length {len(lambdas)} != n_experts {n_experts}"
            )
    else:
        lambdas = [1.0 / n_experts] * n_experts

    scale = config.task_scale

    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    task_vectors = extract_task_vectors(experts, base_cpu)
    base_params = dict(base_cpu.named_parameters())

    merged = copy.deepcopy(base_cpu)
    merged_params = dict(merged.named_parameters())

    operators: List[str] = ["SVD_DIAGNOSTIC"]
    subspace_count = 0
    fallback_count = 0

    with torch.no_grad():
        for name, param in merged_params.items():
            raw_deltas: List[Tensor] = []
            for tv in task_vectors:
                if name in tv:
                    raw_deltas.append(tv[name])

            if not raw_deltas:
                continue

            base_param = base_params[name].detach().float()

            is_2d = all(d.dim() == 2 for d in raw_deltas)
            if is_2d:
                gate_results = [
                    spectral_gate_passes(d, gate_threshold)
                    for d in raw_deltas
                ]
                if all(gate_results):
                    merged_delta = _subspace_merge_param(
                        raw_deltas,
                        lambdas,
                        energy_threshold,
                        projection_strength,
                    )
                    param.copy_(base_param + merged_delta * scale)
                    subspace_count += 1
                else:
                    merged_delta = sum(
                        lam * d for lam, d in zip(lambdas, raw_deltas)
                    )
                    param.copy_(base_param + merged_delta * scale)
                    fallback_count += 1
            else:
                merged_delta = sum(
                    lam * d for lam, d in zip(lambdas, raw_deltas)
                )
                param.copy_(base_param + merged_delta * scale)
                fallback_count += 1

    if subspace_count > 0:
        operators.append("SUBSPACE_PROJECT")
    if fallback_count > 0:
        operators.append("TASK_ARITHMETIC_FALLBACK")

    merged.to(device)

    trace = OperatorTrace(
        method="subspace",
        operators=operators,
        task_scale=scale,
        lambdas=lambdas,
        fisher_used=False,
        activation_covariance_used=False,
        dare_used=False,
        ties_used=False,
        config_hash=config.config_hash(),
    )

    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method="subspace",
    )
