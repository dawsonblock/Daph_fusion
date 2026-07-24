"""K-FAC structured merge module.

For a linear layer y = Wx where W ∈ R^{out×in}, the Fisher information
matrix F has the Kronecker-factored approximation:

    F ≈ G ⊗ A

where A = E[aa^T] ∈ R^{in×in} is the input activation covariance
(a is the column input vector) and G = E[gg^T] ∈ R^{out×out} is the
output gradient covariance (g = ∂L/∂y is the per-sample gradient).

The merge minimizes total local quadratic damage across experts:

    For each expert i:
        L_i(W) ≈ ½ vec(W - W_i)^T (G_i ⊗ A_i) vec(W - W_i)
               = ½ tr(A_i (W - W_i)^T G_i (W - W_i))

    Solve: W* = argmin_W Σ_i λ_i ‖W - W_i‖²_{G_i ⊗ A_i} + ρ‖W‖²

Taking the derivative w.r.t. W and setting to zero (A_i, G_i symmetric):

    Σ_i λ_i G_i W A_i + ρ W = Σ_i λ_i G_i W_i A_i

Progressive approximations:
    K1: Activation-only (G = I) — overlaps with RegMean
        W* = (Σ_i λ_i W_i A_i)(Σ_i λ_i A_i + ρI)^{-1}

    K2: Diagonal: diag(A) ⊗ diag(G)
        W*_{jk} = Σ_i λ_i g_{i,j} a_{i,k} W_{i,jk}
                  / (Σ_i λ_i g_{i,j} a_{i,k} + ρ)

    K3: Block K-FAC
        Split A and G into block-diagonal form. For each (output_block,
        input_block) pair, solve the Sylvester sub-problem:
            (Σ_i λ_i A_{i,p} ⊗ G_{i,r} + ρI) vec(W_{rp})
                = Σ_i λ_i vec(G_{i,r} W_{i,rp} A_{i,p})

    K4: Low-rank factors
        Compute a shared low-rank eigenbasis from the weighted average of
        A and G. Project weights and factors into the reduced space, solve
        the small Sylvester system, and project back. Null-space components
        use the weighted mean as a residual.
"""
from __future__ import annotations

import copy
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


_KFAC_EXCLUDE_PATTERNS = (
    "norm", "ln", "rms", "layernorm", "bias",
    "embed", "wte", "position", "pos_embed",
    "lm_head", "output",
    "a_log", "dt_proj",
    ".d",
)


def _is_kfac_eligible(param_name: str, param: Tensor) -> bool:
    """Check if a parameter is eligible for K-FAC merging.

    K-FAC applies to 2D weight matrices of linear layers only.
    Biases, norms, embeddings, and SSM recurrence parameters are excluded.
    """
    if param.dim() != 2:
        return False
    name_lower = param_name.lower()
    for pattern in _KFAC_EXCLUDE_PATTERNS:
        if pattern in name_lower:
            return False
    if "weight" in name_lower:
        return True
    return False


def _normalize_approximation(approx: str) -> str:
    """Normalize the kfac_approximation string to K1/K2/K3/K4."""
    upper = approx.upper().strip()
    if upper in ("K1", "K2", "K3", "K4"):
        return upper
    if upper in ("DIAGONAL", "DIAG"):
        return "K2"
    if upper in ("ACTIVATION", "ACTIVATION_ONLY", "A_ONLY"):
        return "K1"
    if upper in ("BLOCK", "BLOCK_KFAC"):
        return "K3"
    if upper in ("LOW_RANK", "LOWRANK"):
        return "K4"
    return "K2"


# =============================================================================
# K1: Activation-only (G = I)
# =============================================================================


def _kfac_k1_solve(
    weights: List[Tensor],
    a_factors: List[Tensor],
    lambdas: List[float],
    ridge: float,
) -> Tensor:
    """K1: Activation-only approximation (G = I).

    W* = (Σ_i λ_i W_i A_i)(Σ_i λ_i A_i + ρI)^{-1}

    This is mathematically equivalent to RegMean with C_i = A_i.

    Args:
        weights: List of weight matrices [out, in].
        a_factors: List of input covariance matrices [in, in].
        lambdas: Per-expert weights.
        ridge: Regularization ρ.

    Returns:
        Merged weight matrix [out, in].
    """
    dtype = weights[0].dtype
    device = weights[0].device

    numerator = sum(
        lam * w.float() @ a.float()
        for lam, w, a in zip(lambdas, weights, a_factors)
    )

    denominator = sum(
        lam * a.float() for lam, a in zip(lambdas, a_factors)
    )
    in_dim = denominator.shape[0]
    denominator = denominator + ridge * torch.eye(
        in_dim, dtype=denominator.dtype, device=device
    )

    w_star = torch.linalg.solve(denominator, numerator.t()).t()
    return w_star.to(dtype)


# =============================================================================
# K2: Diagonal approximation
# =============================================================================


def _kfac_k2_solve(
    weights: List[Tensor],
    a_factors: List[Tensor],
    g_factors: List[Tensor],
    lambdas: List[float],
    ridge: float,
) -> Tensor:
    """K2: Diagonal approximation diag(A) ⊗ diag(G).

    The Fisher for element (j, k) is approximated as G_{jj} * A_{kk}.
    The solution is element-wise:

        W*_{jk} = Σ_i λ_i g_{i,j} a_{i,k} W_{i,jk}
                  / (Σ_i λ_i g_{i,j} a_{i,k} + ρ)

    Args:
        weights: List of weight matrices [out, in].
        a_factors: List of input covariance matrices [in, in].
        g_factors: List of output gradient covariance matrices [out, out].
        lambdas: Per-expert weights.
        ridge: Regularization ρ.

    Returns:
        Merged weight matrix [out, in].
    """
    dtype = weights[0].dtype
    device = weights[0].device

    a_diags = [
        a.float().diagonal() if a.dim() == 2 else a.float()
        for a in a_factors
    ]
    g_diags = [
        g.float().diagonal() if g.dim() == 2 else g.float()
        for g in g_factors
    ]

    out_dim, in_dim = weights[0].shape

    numerator = torch.zeros(out_dim, in_dim, dtype=torch.float32, device=device)
    denominator = torch.zeros(out_dim, in_dim, dtype=torch.float32, device=device)

    for lam, w, a_diag, g_diag in zip(lambdas, weights, a_diags, g_diags):
        fisher_element = torch.outer(g_diag, a_diag)
        numerator += lam * fisher_element * w.float()
        denominator += lam * fisher_element

    denominator += ridge
    w_star = numerator / denominator
    return w_star.to(dtype)


# =============================================================================
# K3: Block K-FAC
# =============================================================================


def _kfac_k3_solve(
    weights: List[Tensor],
    a_factors: List[Tensor],
    g_factors: List[Tensor],
    lambdas: List[float],
    ridge: float,
    block_size: int,
) -> Tensor:
    """K3: Block K-FAC approximation.

    A and G are assumed block-diagonal with block size `block_size`.
    For each (output_block, input_block) pair, the Sylvester sub-problem
    is solved directly:

        (Σ_i λ_i A_{i,p} ⊗ G_{i,r} + ρI) vec(W_{rp})
            = Σ_i λ_i vec(G_{i,r} W_{i,rp} A_{i,p})

    Args:
        weights: List of weight matrices [out, in].
        a_factors: List of input covariance matrices [in, in].
        g_factors: List of output gradient covariance matrices [out, out].
        lambdas: Per-expert weights.
        ridge: Regularization ρ.
        block_size: Block size for block-diagonal approximation.

    Returns:
        Merged weight matrix [out, in].
    """
    dtype = weights[0].dtype
    device = weights[0].device
    out_dim, in_dim = weights[0].shape

    num_out_blocks = (out_dim + block_size - 1) // block_size
    num_in_blocks = (in_dim + block_size - 1) // block_size

    result = torch.zeros(out_dim, in_dim, dtype=torch.float32, device=device)

    for ob in range(num_out_blocks):
        o_start = ob * block_size
        o_end = min(o_start + block_size, out_dim)
        o_size = o_end - o_start

        for ib in range(num_in_blocks):
            i_start = ib * block_size
            i_end = min(i_start + block_size, in_dim)
            i_size = i_end - i_start

            w_blocks = [
                w.float()[o_start:o_end, i_start:i_end] for w in weights
            ]
            a_blocks = [
                a.float()[i_start:i_end, i_start:i_end] for a in a_factors
            ]
            g_blocks = [
                g.float()[o_start:o_end, o_start:o_end] for g in g_factors
            ]

            n = o_size * i_size
            lhs = ridge * torch.eye(n, dtype=torch.float32, device=device)
            rhs = torch.zeros(n, dtype=torch.float32, device=device)

            for lam, w_b, a_b, g_b in zip(lambdas, w_blocks, a_blocks, g_blocks):
                kron = torch.kron(a_b, g_b)
                lhs += lam * kron
                rhs += lam * (g_b @ w_b @ a_b).flatten()

            vec_w = torch.linalg.solve(lhs, rhs)
            result[o_start:o_end, i_start:i_end] = vec_w.reshape(o_size, i_size)

    return result.to(dtype)


# =============================================================================
# K4: Low-rank factors
# =============================================================================


def _kfac_k4_solve(
    weights: List[Tensor],
    a_factors: List[Tensor],
    g_factors: List[Tensor],
    lambdas: List[float],
    ridge: float,
    low_rank: int,
) -> Tensor:
    """K4: Low-rank factor approximation.

    Computes a shared low-rank eigenbasis from the weighted averages of A
    and G across experts. Projects weights and factors into the reduced
    space, solves the small Sylvester system, and projects back.

    The null-space component (directions not captured by the top-k
    eigenvectors) uses the weighted mean of the experts' weights as a
    residual, since curvature is negligible in those directions.

    Args:
        weights: List of weight matrices [out, in].
        a_factors: List of input covariance matrices [in, in].
        g_factors: List of output gradient covariance matrices [out, out].
        lambdas: Per-expert weights.
        ridge: Regularization ρ.
        low_rank: Number of eigenvectors to retain (k).

    Returns:
        Merged weight matrix [out, in].
    """
    dtype = weights[0].dtype
    device = weights[0].device
    out_dim, in_dim = weights[0].shape

    a_avg = sum(lam * a.float() for lam, a in zip(lambdas, a_factors))
    g_avg = sum(lam * g.float() for lam, g in zip(lambdas, g_factors))

    a_eigvals, a_eigvecs = torch.linalg.eigh(a_avg)
    g_eigvals, g_eigvecs = torch.linalg.eigh(g_avg)

    k_a = min(low_rank, in_dim)
    k_g = min(low_rank, out_dim)

    u_a = a_eigvecs[:, -k_a:]
    u_g = g_eigvecs[:, -k_g:]

    tilde_weights = [u_g.t() @ w.float() @ u_a for w in weights]
    tilde_a = [u_a.t() @ a.float() @ u_a for a in a_factors]
    tilde_g = [u_g.t() @ g.float() @ u_g for g in g_factors]

    n = k_g * k_a
    lhs = ridge * torch.eye(n, dtype=torch.float32, device=device)
    rhs = torch.zeros(n, dtype=torch.float32, device=device)

    for lam, tw, ta, tg in zip(lambdas, tilde_weights, tilde_a, tilde_g):
        kron = torch.kron(ta, tg)
        lhs += lam * kron
        rhs += lam * (tg @ tw @ ta).flatten()

    tilde_w_vec = torch.linalg.solve(lhs, rhs)
    tilde_w = tilde_w_vec.reshape(k_g, k_a)

    w_star = u_g @ tilde_w @ u_a.t()

    w_mean = sum(lam * w.float() for lam, w in zip(lambdas, weights))
    captured = u_g @ (u_g.t() @ w_mean) @ u_a @ u_a.t()
    w_star = w_star + (w_mean - captured)

    return w_star.to(dtype)


# =============================================================================
# Main K-FAC merge entry point
# =============================================================================


def merge_kfac(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    kfac_bank: Dict[str, Dict[str, Tuple[Tensor, Tensor]]],
    device: str = "cpu",
) -> MergeResult:
    """K-FAC structured merge.

    W* = argmin_W Σ_i λ_i ‖W - W_i‖²_{G_i ⊗ A_i} + ρ‖W‖²

    Uses the Kronecker-factored Fisher approximation F ≈ G ⊗ A to weight
    each expert's contribution. The approximation level is controlled by
    config.kfac_approximation ("K1"–"K4").

    For K-FAC-eligible parameters (linear weights), uses the Kronecker
    factors from kfac_bank. For non-eligible parameters, falls back to
    Task Arithmetic: θ* = θ_0 + α Σ_i λ_i Δ_i.

    Args:
        base_model: Base model θ_0.
        experts: Specialist models.
        config: Merge configuration (method must be kfac_barycenter).
        kfac_bank: Dict mapping expert_name -> {param_name -> (A_factor, G_factor)}
            where A_factor ∈ R^{in×in} and G_factor ∈ R^{out×out}.
        device: Device for computation.

    Returns:
        MergeResult with operator trace.
    """
    if config.method != MergeMethod.KFAC_BARYCENTER:
        raise ValueError(
            f"merge_kfac called with method={config.method}, "
            f"expected kfac_barycenter"
        )

    n_experts = len(experts)
    validate_parameter_names(experts, base_model)

    if config.lambdas:
        lambdas = list(config.lambdas)
    else:
        lambdas = [1.0 / n_experts] * n_experts

    ridge = config.regmean_ridge
    scale = config.task_scale
    approximation = _normalize_approximation(config.kfac_approximation)

    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    base_params = dict(base_cpu.named_parameters())
    task_vectors = extract_task_vectors(experts, base_cpu)

    expert_names = [f"expert_{i}" for i in range(n_experts)]

    merged = copy.deepcopy(base_cpu)
    merged_params = dict(merged.named_parameters())

    kfac_count = 0
    ta_count = 0

    with torch.no_grad():
        for name, param in merged_params.items():
            deltas = []
            for i, tv in enumerate(task_vectors):
                if name in tv:
                    deltas.append(tv[name] * lambdas[i])

            if not deltas:
                continue

            base_param = base_params[name].detach().float()

            use_kfac = _is_kfac_eligible(name, base_param)
            factors_available = False
            if use_kfac:
                factors_available = all(
                    ename in kfac_bank and name in kfac_bank[ename]
                    for ename in expert_names
                )

            if use_kfac and factors_available:
                expert_weights = [
                    base_param + tv[name] for tv in task_vectors
                ]
                a_factors = []
                g_factors = []
                for ename in expert_names:
                    a_f, g_f = kfac_bank[ename][name]
                    a_factors.append(a_f.float())
                    g_factors.append(g_f.float())

                if approximation == "K1":
                    merged_weight = _kfac_k1_solve(
                        expert_weights, a_factors, lambdas, ridge
                    )
                elif approximation == "K2":
                    merged_weight = _kfac_k2_solve(
                        expert_weights, a_factors, g_factors,
                        lambdas, ridge
                    )
                elif approximation == "K3":
                    merged_weight = _kfac_k3_solve(
                        expert_weights, a_factors, g_factors,
                        lambdas, ridge, config.regmean_block_size
                    )
                elif approximation == "K4":
                    merged_weight = _kfac_k4_solve(
                        expert_weights, a_factors, g_factors,
                        lambdas, ridge, config.regmean_low_rank
                    )
                else:
                    merged_weight = _kfac_k2_solve(
                        expert_weights, a_factors, g_factors,
                        lambdas, ridge
                    )

                merged_delta = (merged_weight - base_param) * scale
                param.copy_(base_param + merged_delta)
                kfac_count += 1
            else:
                merged_delta = sum(deltas)
                param.copy_(base_param + merged_delta * scale)
                ta_count += 1

    merged.to(device)

    trace = OperatorTrace(
        method="kfac_barycenter",
        operators=["KFAC_MERGE"],
        kfac_used=True,
        fisher_used=True,
        fisher_estimator="exact_per_sample",
        activation_covariance_used=True,
        dare_used=False,
        ties_used=False,
        task_scale=scale,
        lambdas=lambdas,
        config_hash=config.config_hash(),
    )

    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method="kfac_barycenter",
    )
