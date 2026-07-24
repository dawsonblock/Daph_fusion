"""Fisher interaction matrix and curvature cosine (Phase 8, 25).

For expert task vectors Δᵢ, compute the Fisher interaction matrix:

    G_ij = Δᵢᵀ F₀ Δⱼ

And the normalized curvature cosine:

    C^F_ij = Δᵢᵀ F₀ Δⱼ / (√(Δᵢᵀ F₀ Δᵢ) √(Δⱼᵀ F₀ Δⱼ))

This collapses a billions-dimensional curvature constraint into an N×N
matrix. For three experts, G ∈ R^{3×3}.

AGX uses this to distinguish:
    Euclidean aligned + Fisher aligned     → easy merge
    Euclidean conflict + Fisher weak       → conflict probably unimportant
    Euclidean aligned + Fisher conflict    → dangerous hidden interaction
    Fisher orthogonal                      → potentially coexist safely
    Strong Fisher conflict                 → projection / suppression / base anchoring

This is much more relevant to dense LLM geometry than raw sign disagreement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class InteractionMatrix:
    """Fisher interaction matrix and derived diagnostics."""
    G: Tensor                         # N×N Fisher interaction matrix
    C_fisher: Tensor                  # N×N curvature cosine matrix
    C_euclidean: Tensor               # N×N Euclidean cosine matrix
    norms_fisher: Tensor              # N vector of Fisher-weighted norms
    norms_euclidean: Tensor           # N vector of Euclidean norms
    n_experts: int

    def interaction_type(self, i: int, j: int) -> str:
        """Classify the interaction between experts i and j."""
        c_f = self.C_fisher[i, j].item()
        c_e = self.C_euclidean[i, j].item()

        if c_f > 0.7 and c_e > 0.7:
            return "aligned_aligned"       # easy merge
        elif c_f < -0.3 and abs(c_e) < 0.3:
            return "conflict_weak"         # conflict probably unimportant
        elif c_f < -0.3 and c_e > 0.5:
            return "aligned_conflict"      # dangerous hidden interaction
        elif abs(c_f) < 0.3:
            return "fisher_orthogonal"     # potentially coexist safely
        elif c_f < -0.5:
            return "strong_fisher_conflict"  # projection / suppression
        else:
            return "moderate_interaction"


def compute_fisher_interaction_matrix(
    task_vectors: List[Dict[str, Tensor]],
    base_fisher: Dict[str, Tensor],
) -> Tensor:
    """Compute the Fisher interaction matrix G.

    G_ij = Σ_k Δ_{i,k}ᵀ F_{0,k} Δ_{j,k}

    where the sum is over all parameters k.

    Args:
        task_vectors: List of N task vector dicts.
        base_fisher: Dict mapping param_name -> base Fisher diagonal.

    Returns:
        G: N×N tensor.
    """
    n = len(task_vectors)
    G = torch.zeros(n, n, dtype=torch.float32)

    for name in task_vectors[0]:
        if name not in base_fisher:
            continue

        f = base_fisher[name].float()
        deltas = []
        for tv in task_vectors:
            if name in tv:
                deltas.append(tv[name].float().flatten())
            else:
                deltas.append(None)

        for i in range(n):
            if deltas[i] is None:
                continue
            for j in range(n):
                if deltas[j] is None:
                    continue
                # Δᵢᵀ F₀ Δⱼ = Σ (Δᵢ * F₀ * Δⱼ)
                G[i, j] += (deltas[i] * f.flatten() * deltas[j]).sum()

    return G


def compute_curvature_cosine(
    task_vectors: List[Dict[str, Tensor]],
    base_fisher: Dict[str, Tensor],
) -> Tensor:
    """Compute the curvature cosine matrix C^F.

    C^F_ij = Δᵢᵀ F₀ Δⱼ / (√(Δᵢᵀ F₀ Δᵢ) √(Δⱼᵀ F₀ Δⱼ))

    Args:
        task_vectors: List of N task vector dicts.
        base_fisher: Dict mapping param_name -> base Fisher diagonal.

    Returns:
        C^F: N×N tensor with values in [-1, 1].
    """
    G = compute_fisher_interaction_matrix(task_vectors, base_fisher)
    n = G.shape[0]

    # Diagonal = Fisher-weighted norms squared
    diag = G.diagonal().clamp(min=1e-12)
    norms = diag.sqrt()

    # C^F_ij = G_ij / (norm_i * norm_j)
    C = G / (norms.unsqueeze(0) * norms.unsqueeze(1) + 1e-12)

    # Clamp to [-1, 1] for numerical stability
    return C.clamp(-1.0, 1.0)


def compute_euclidean_cosine(
    task_vectors: List[Dict[str, Tensor]],
) -> Tensor:
    """Compute the Euclidean cosine matrix.

    C_ij = Δᵢ · Δⱼ / (‖Δᵢ‖ ‖Δⱼ‖)

    Args:
        task_vectors: List of N task vector dicts.

    Returns:
        C: N×N tensor with values in [-1, 1].
    """
    n = len(task_vectors)
    # Flatten all params into one vector per expert
    flat_deltas = []
    for tv in task_vectors:
        flat = torch.cat([v.float().flatten() for v in tv.values()])
        flat_deltas.append(flat)

    C = torch.zeros(n, n, dtype=torch.float32)
    norms = torch.stack([d.norm() for d in flat_deltas])

    for i in range(n):
        for j in range(n):
            C[i, j] = torch.dot(flat_deltas[i], flat_deltas[j]) / (norms[i] * norms[j] + 1e-12)

    return C.clamp(-1.0, 1.0)


def compute_interaction_matrix(
    task_vectors: List[Dict[str, Tensor]],
    base_fisher: Dict[str, Tensor],
) -> InteractionMatrix:
    """Compute the full interaction matrix with all diagnostics.

    Returns an InteractionMatrix containing:
        - Fisher interaction matrix G
        - Curvature cosine C^F
        - Euclidean cosine C
        - Fisher-weighted and Euclidean norms
    """
    G = compute_fisher_interaction_matrix(task_vectors, base_fisher)
    C_fisher = compute_curvature_cosine(task_vectors, base_fisher)
    C_euclidean = compute_euclidean_cosine(task_vectors)

    norms_fisher = G.diagonal().clamp(min=1e-12).sqrt()
    flat_deltas = [torch.cat([v.float().flatten() for v in tv.values()]) for tv in task_vectors]
    norms_euclidean = torch.stack([d.norm() for d in flat_deltas])

    return InteractionMatrix(
        G=G,
        C_fisher=C_fisher,
        C_euclidean=C_euclidean,
        norms_fisher=norms_fisher,
        norms_euclidean=norms_euclidean,
        n_experts=len(task_vectors),
    )
