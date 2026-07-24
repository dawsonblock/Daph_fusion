"""Geometry Profiler (AGX Phase 3).

Before AGX merges anything, profile every candidate architectural group.

For each group g, calculate a GeometryProfile:
    G_g = {N, C, C_F, S, P, A, R, K}

where:
    N:   task-vector norms
    C:   Euclidean cosine matrix
    C_F: Fisher cosine matrix
    S:   sign-conflict statistics
    P:   principal angles / subspace overlap
    A:   activation covariance distances
    R:   spectral / effective ranks
    K:   base curvature / sensitivity

Also collect functional sensitivity:
    D_g^act = E_x[‖h_g^merge(x) - h_g^reference(x)‖²]

AGX knows what kind of problem exists before selecting a solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from daph_exfusion.geometry.interactions import (
    InteractionMatrix,
    compute_interaction_matrix,
)
from daph_exfusion.geometry.spectral import (
    SpectralDiagnostics,
    batch_spectral_diagnostics,
)
from daph_exfusion.merge.types import (
    classify_parameter_family,
    get_layer_index,
    count_layers,
)


@dataclass
class GroupGeometryProfile:
    """Geometry profile for an architectural group.

    Contains all diagnostics AGX needs to select a merge operator.
    """
    group_name: str
    param_names: List[str] = field(default_factory=list)

    # N: task-vector norms per expert
    norms: Dict[str, List[float]] = field(default_factory=dict)  # expert -> [norm_per_param]

    # C: Euclidean cosine matrix (averaged over params in group)
    euclidean_cosine: Optional[Tensor] = None  # [N_experts, N_experts]

    # C_F: Fisher cosine matrix (averaged over params in group)
    fisher_cosine: Optional[Tensor] = None  # [N_experts, N_experts]

    # S: sign-conflict statistics
    sign_conflict_rate: float = 0.0  # fraction of coords with sign disagreement

    # P: principal angles (average over params)
    avg_principal_angle: float = 0.0

    # A: activation covariance distances (if available)
    activation_cov_distance: Optional[float] = None

    # R: spectral statistics (averaged over params and experts)
    avg_rank_90: float = 0.0
    avg_effective_rank: float = 0.0
    spectral_gate_pass_rate: float = 0.0  # fraction of params passing the gate

    # K: base curvature / sensitivity
    avg_base_fisher_norm: float = 0.0

    # Functional sensitivity
    functional_sensitivity: Optional[float] = None

    # Summary diagnostics
    n_experts: int = 0
    n_params: int = 0

    def to_dict(self) -> dict:
        return {
            "group_name": self.group_name,
            "n_experts": self.n_experts,
            "n_params": self.n_params,
            "sign_conflict_rate": self.sign_conflict_rate,
            "avg_principal_angle": self.avg_principal_angle,
            "avg_rank_90": self.avg_rank_90,
            "avg_effective_rank": self.avg_effective_rank,
            "spectral_gate_pass_rate": self.spectral_gate_pass_rate,
            "avg_base_fisher_norm": self.avg_base_fisher_norm,
            "activation_cov_distance": self.activation_cov_distance,
            "functional_sensitivity": self.functional_sensitivity,
            "euclidean_cosine": self.euclidean_cosine.tolist() if self.euclidean_cosine is not None else None,
            "fisher_cosine": self.fisher_cosine.tolist() if self.fisher_cosine is not None else None,
            "norms": {k: v for k, v in self.norms.items()},
        }


def compute_sign_conflict_rate(deltas: List[Tensor]) -> float:
    """Compute the fraction of coordinates with sign disagreement across experts."""
    if len(deltas) < 2:
        return 0.0

    total_coords = 0
    conflict_coords = 0

    for d in deltas:
        flat = d.flatten()
        total_coords += flat.numel()

    # For each coordinate, check if signs disagree
    stacked = torch.stack([d.flatten() for d in deltas], dim=0)  # [N, ...]
    signs = stacked.sign()
    # Conflict if not all signs are the same (excluding zeros)
    nonzero_mask = (stacked.abs() > 1e-10).any(dim=0)
    if nonzero_mask.sum() == 0:
        return 0.0

    sign_std = signs[:, nonzero_mask].std(dim=0)
    conflict_coords = (sign_std > 0).sum().item()
    total_nonzero = nonzero_mask.sum().item()

    return conflict_coords / total_nonzero if total_nonzero > 0 else 0.0


def compute_principal_angles(deltas: List[Tensor]) -> float:
    """Compute average principal angle between expert delta subspaces.

    For 2D weight matrices, compute the SVD and find the principal angles
    between the top-k singular vectors.
    """
    if len(deltas) < 2:
        return 0.0

    angles = []
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            d_i = deltas[i]
            d_j = deltas[j]

            if d_i.dim() < 2 or d_j.dim() < 2:
                continue

            # SVD to get right singular vectors
            try:
                _, _, v_i = torch.linalg.svd(d_i.float(), full_matrices=False)
                _, _, v_j = torch.linalg.svd(d_j.float(), full_matrices=False)

                k = min(v_i.shape[0], v_j.shape[0], 10)  # top-k
                v_i_k = v_i[:k]
                v_j_k = v_j[:k]

                # Principal angles via SVD of V_i^T V_j
                cos_angles = torch.linalg.svdvals(v_i_k @ v_j_k.t())
                cos_angles = cos_angles.clamp(-1, 1)
                angles_radians = torch.arccos(cos_angles)
                angles.append(angles_radians.mean().item())
            except Exception:
                continue

    return sum(angles) / len(angles) if angles else 0.0


def profile_group(
    group_name: str,
    param_names: List[str],
    task_vectors: List[Dict[str, Tensor]],
    base_fisher: Optional[Dict[str, Tensor]] = None,
    activation_bank: Optional[Dict[str, Dict[str, Tensor]]] = None,
) -> GroupGeometryProfile:
    """Profile a single architectural group.

    Args:
        group_name: Name of the group (e.g., "early_attention").
        param_names: Parameter names belonging to this group.
        task_vectors: List of N task vector dicts.
        base_fisher: Optional base model Fisher diagonal.
        activation_bank: Optional activation covariance bank.

    Returns:
        GroupGeometryProfile with all diagnostics.
    """
    n_experts = len(task_vectors)
    profile = GroupGeometryProfile(
        group_name=group_name,
        param_names=param_names,
        n_experts=n_experts,
        n_params=len(param_names),
    )

    # N: norms
    for i, tv in enumerate(task_vectors):
        expert_key = f"expert_{i}"
        profile.norms[expert_key] = []
        for name in param_names:
            if name in tv:
                profile.norms[expert_key].append(tv[name].float().norm().item())

    # Collect deltas for this group
    group_deltas: Dict[str, List[Tensor]] = {}  # param_name -> [delta_per_expert]
    for name in param_names:
        deltas = []
        for tv in task_vectors:
            if name in tv:
                deltas.append(tv[name].float())
        if deltas:
            group_deltas[name] = deltas

    # S: sign conflict
    all_conflict_rates = []
    for name, deltas in group_deltas.items():
        all_conflict_rates.append(compute_sign_conflict_rate(deltas))
    profile.sign_conflict_rate = sum(all_conflict_rates) / len(all_conflict_rates) if all_conflict_rates else 0.0

    # P: principal angles
    all_angles = []
    for name, deltas in group_deltas.items():
        all_angles.append(compute_principal_angles(deltas))
    profile.avg_principal_angle = sum(all_angles) / len(all_angles) if all_angles else 0.0

    # R: spectral diagnostics
    all_rank_90 = []
    all_eff_rank = []
    gate_passes = 0
    gate_total = 0

    for name, deltas in group_deltas.items():
        for d in deltas:
            if d.dim() >= 2:
                from daph_exfusion.geometry.spectral import compute_spectral_diagnostics
                diag = compute_spectral_diagnostics(d)
                all_rank_90.append(diag.rank_90)
                all_eff_rank.append(diag.effective_rank)
                gate_total += 1
                if diag.spectral_gate_passes:
                    gate_passes += 1

    if all_rank_90:
        profile.avg_rank_90 = sum(all_rank_90) / len(all_rank_90)
        profile.avg_effective_rank = sum(all_eff_rank) / len(all_eff_rank)
    if gate_total > 0:
        profile.spectral_gate_pass_rate = gate_passes / gate_total

    # C and C_F: interaction matrices (computed on group-level task vectors)
    group_tv = [{name: tv[name] for name in group_deltas if name in tv} for tv in task_vectors]
    if base_fisher:
        # Filter to params that have base Fisher
        fisher_tv = []
        for tv in group_tv:
            filtered = {name: d for name, d in tv.items() if name in base_fisher}
            fisher_tv.append(filtered)
        if all(fisher_tv):
            interaction = compute_interaction_matrix(fisher_tv, base_fisher)
            profile.fisher_cosine = interaction.C_fisher
            # K: base curvature
            fisher_norms = []
            for name in group_deltas:
                if name in base_fisher:
                    fisher_norms.append(base_fisher[name].float().norm().item())
            if fisher_norms:
                profile.avg_base_fisher_norm = sum(fisher_norms) / len(fisher_norms)

    # Euclidean cosine
    from daph_exfusion.geometry.interactions import compute_euclidean_cosine
    if group_tv and all(group_tv):
        profile.euclidean_cosine = compute_euclidean_cosine(group_tv)

    # A: activation covariance distances
    if activation_bank:
        cov_distances = []
        expert_names = list(activation_bank.keys())
        for i in range(len(expert_names)):
            for j in range(i + 1, len(expert_names)):
                e1, e2 = expert_names[i], expert_names[j]
                for name in param_names:
                    c1 = activation_bank[e1].get(name)
                    c2 = activation_bank[e2].get(name)
                    if c1 is not None and c2 is not None:
                        dist = (c1.float() - c2.float()).norm().item()
                        cov_distances.append(dist)
        if cov_distances:
            profile.activation_cov_distance = sum(cov_distances) / len(cov_distances)

    return profile


def profile_all_groups(
    task_vectors: List[Dict[str, Tensor]],
    base_fisher: Optional[Dict[str, Tensor]] = None,
    activation_bank: Optional[Dict[str, Dict[str, Tensor]]] = None,
    num_layers: int = -1,
) -> Dict[str, GroupGeometryProfile]:
    """Profile all architectural groups.

    Groups: embeddings, early/middle/late_attention, early/middle/late_ffn,
    ssm_projections, ssm_recurrence, normalization, lm_head, other.

    Args:
        task_vectors: List of N task vector dicts.
        base_fisher: Optional base Fisher.
        activation_bank: Optional activation bank.
        num_layers: Number of layers (for early/middle/late splitting).

    Returns:
        Dict mapping group_name -> GroupGeometryProfile.
    """
    if num_layers < 0:
        # Infer from task vectors
        max_layer = -1
        for tv in task_vectors:
            for name in tv:
                idx = get_layer_index(name)
                if idx > max_layer:
                    max_layer = idx
        num_layers = max_layer + 1 if max_layer >= 0 else 1

    # Classify all parameters into groups
    group_params: Dict[str, List[str]] = {}
    for name in task_vectors[0]:
        layer_idx = get_layer_index(name)
        family = classify_parameter_family(name, layer_idx, num_layers)
        if family not in group_params:
            group_params[family] = []
        group_params[family].append(name)

    # Profile each group
    profiles: Dict[str, GroupGeometryProfile] = {}
    for group_name, params in group_params.items():
        profiles[group_name] = profile_group(
            group_name=group_name,
            param_names=params,
            task_vectors=task_vectors,
            base_fisher=base_fisher,
            activation_bank=activation_bank,
        )

    return profiles
