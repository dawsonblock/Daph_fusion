"""RegMean merge module (Phase 10 — activation-space merging).

For a linear layer y = Wx, RegMean solves:

    min_W  Σᵢ E[‖Wxᵢ - Wᵢxᵢ‖²]

Using Cᵢ = E[xᵢxᵢᵀ], the optimum is:

    W* = (Σᵢ WᵢCᵢ)(Σᵢ Cᵢ + ρI)⁻¹

Never explicitly compute C⁻¹. Use torch.linalg.solve or Cholesky when SPD.

Four covariance modes:
    FULL:      C ∈ R^{d×d}       (research reference)
    BLOCK:     block-diagonal    (split hidden dim into blocks)
    DIAGONAL:  C = diag(c)       (cheap approximation)
    LOW_RANK:  C ≈ UΛUᵀ + σ²I   (Woodbury for inverse)

RegMean applies to linear weight matrices only:
    attention: q_proj, k_proj, v_proj, o_proj
    FFN:       gate_proj, up_proj, down_proj
    Mamba:     in_proj, x_proj, dt_proj, out_proj

Do NOT RegMean: LayerNorm/RMSNorm, biases, embeddings, A_log, D.
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
    RegMeanMode,
    extract_task_vectors,
    validate_parameter_names,
)


# =============================================================================
# RegMean-eligible parameter detection
# =============================================================================


# Parameters that should NOT be RegMean'd
_REGMEAN_EXCLUDE_PATTERNS = (
    "norm", "ln", "rms", "layernorm", "bias",
    "embed", "wte", "position", "pos_embed",
    "lm_head", "output",
    "a_log", "dt_proj",  # SSM recurrence — needs special treatment
    ".d",  # SSM D parameter
)

# Linear projection parameters that ARE RegMean-eligible
_REGMEAN_INCLUDE_PATTERNS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj", "x_proj", "out_proj",
    "fc", "intermediate", "mlp", "ffn",
)


def is_regmean_eligible(param_name: str, param: Tensor) -> bool:
    """Check if a parameter is eligible for RegMean merging.

    RegMean applies to 2D weight matrices of linear layers only.
    Biases, norms, embeddings, and SSM recurrence parameters are excluded.
    """
    if param.dim() != 2:
        return False

    name_lower = param_name.lower()

    # Check exclusions first
    for pattern in _REGMEAN_EXCLUDE_PATTERNS:
        if pattern in name_lower:
            return False

    # Check inclusions
    for pattern in _REGMEAN_INCLUDE_PATTERNS:
        if pattern in name_lower and "weight" in name_lower:
            return True

    # Default: if it's a 2D weight and not excluded, allow it
    if "weight" in name_lower:
        return True

    return False


# =============================================================================
# RegMean core solver
# =============================================================================


def regmean_solve(
    weights: List[Tensor],       # [W₁, W₂, ...] each [out, in]
    covariances: List[Tensor],   # [C₁, C₂, ...] each [in, in] or [in] (diag)
    ridge: float = 1e-4,
    mode: RegMeanMode = RegMeanMode.DIAGONAL,
    block_size: int = 256,
    low_rank: int = 64,
) -> Tensor:
    """Solve RegMean: W* = (Σᵢ WᵢCᵢ)(Σᵢ Cᵢ + ρI)⁻¹.

    Args:
        weights: List of weight matrices [out, in].
        covariances: List of covariance matrices matching input dimension.
        ridge: Regularization ρ.
        mode: Covariance approximation mode.
        block_size: Block size for block-diagonal mode.
        low_rank: Rank for low-rank approximation.

    Returns:
        Merged weight matrix [out, in].
    """
    if not weights:
        raise ValueError("RegMean requires at least one weight")
    if len(weights) != len(covariances):
        raise ValueError(f"weights count {len(weights)} != covariances count {len(covariances)}")

    out_dim, in_dim = weights[0].shape

    if mode == RegMeanMode.DIAGONAL:
        return _regmean_diagonal(weights, covariances, ridge)
    elif mode == RegMeanMode.FULL:
        return _regmean_full(weights, covariances, ridge)
    elif mode == RegMeanMode.BLOCK:
        return _regmean_block(weights, covariances, ridge, block_size)
    elif mode == RegMeanMode.LOW_RANK:
        return _regmean_low_rank(weights, covariances, ridge, low_rank)
    else:
        raise ValueError(f"Unknown RegMean mode: {mode}")


def _regmean_full(
    weights: List[Tensor],
    covariances: List[Tensor],
    ridge: float,
) -> Tensor:
    """Full RegMean: W* = (Σᵢ WᵢCᵢ)(Σᵢ Cᵢ + ρI)⁻¹.

    Uses torch.linalg.solve instead of explicit inverse.
    """
    # Σᵢ WᵢCᵢ  [out, in]
    numerator = sum(w @ c for w, c in zip(weights, covariances))

    # Σᵢ Cᵢ + ρI  [in, in]
    total_cov = sum(c for c in covariances)
    total_cov = total_cov + ridge * torch.eye(
        total_cov.shape[0], dtype=total_cov.dtype, device=total_cov.device
    )

    # Solve: W* = numerator @ total_cov⁻¹
    # Using solve: total_covᵀ @ W*ᵀ = numeratorᵀ
    # W* = (solve(total_covᵀ, numeratorᵢᵀ))ᵀ
    # But total_cov is symmetric, so total_covᵀ = total_cov
    wt_star = torch.linalg.solve(total_cov, numerator.t()).t()
    return wt_star


def _regmean_diagonal(
    weights: List[Tensor],
    covariances: List[Tensor],
    ridge: float,
) -> Tensor:
    """Diagonal RegMean: C = diag(c).

    W* = (Σᵢ Wᵢ diag(cᵢ)) / (Σᵢ diag(cᵢ) + ρI)
       = (Σᵢ Wᵢ ⊙ cᵢ) / (Σᵢ cᵢ + ρ)   (broadcast over output dim)
    """
    # Convert covariances to diagonal vectors if needed
    diag_covs = []
    for c in covariances:
        if c.dim() == 2:
            diag_covs.append(c.diagonal())
        else:
            diag_covs.append(c)

    # Σᵢ Wᵢ ⊙ cᵢ  [out, in]  (cᵢ broadcasts across output dim)
    numerator = sum(w * c.unsqueeze(0) for w, c in zip(weights, diag_covs))

    # Σᵢ cᵢ + ρ  [in]
    denominator = sum(c for c in diag_covs) + ridge

    return numerator / denominator.unsqueeze(0)


def _regmean_block(
    weights: List[Tensor],
    covariances: List[Tensor],
    ridge: float,
    block_size: int,
) -> Tensor:
    """Block-diagonal RegMean: split input dimension into blocks."""
    out_dim, in_dim = weights[0].shape
    num_blocks = (in_dim + block_size - 1) // block_size

    result = torch.zeros_like(weights[0])

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, in_dim)

        # Extract blocks
        w_blocks = [w[:, start:end] for w in weights]
        c_blocks = []
        for c in covariances:
            if c.dim() == 2:
                c_blocks.append(c[start:end, start:end])
            else:
                c_blocks.append(c[start:end])

        # Solve sub-problem in full mode
        block_result = _regmean_full(w_blocks, c_blocks, ridge)
        result[:, start:end] = block_result

    return result


def _regmean_low_rank(
    weights: List[Tensor],
    covariances: List[Tensor],
    ridge: float,
    low_rank: int,
) -> Tensor:
    """Low-rank RegMean: C ≈ UΛUᵀ + σ²I.

    Uses Woodbury identity for efficient inverse:
    (A + UCV)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹
    """
    out_dim, in_dim = weights[0].shape

    # For each covariance, decompose into low-rank + diagonal
    total_diag = torch.zeros(in_dim, dtype=weights[0].dtype, device=weights[0].device)
    total_low_rank_num = torch.zeros(out_dim, in_dim, dtype=weights[0].dtype, device=weights[0].device)

    us = []
    lambdas = []

    for w, c in zip(weights, covariances):
        if c.dim() == 2:
            # SVD-based low-rank approximation
            u, s, vh = torch.linalg.svd(c.float(), full_matrices=False)
            k = min(low_rank, s.shape[0])
            u_k = u[:, :k]
            s_k = s[:k]

            # Diagonal residual (average of remaining eigenvalues)
            if s.shape[0] > k:
                sigma2 = s[k:].mean().item() if s[k:].numel() > 0 else 0.0
            else:
                sigma2 = 0.0

            total_diag += sigma2
            us.append(u_k)
            lambdas.append(s_k)

            # Wᵢ @ Cᵢ ≈ Wᵢ @ (UΛUᵀ + σ²I) = WᵢUΛUᵀ + σ²Wᵢ
            total_low_rank_num += (w @ u_k) * s_k.unsqueeze(0) @ u_k.t() + sigma2 * w
        else:
            # Diagonal covariance — just add
            total_diag += c
            total_low_rank_num += w * c.unsqueeze(0)

    # Total covariance: Σᵢ (UᵢΛᵢUᵢᵀ + σ²ᵢI) + ρI
    # = (Σᵢ σ²ᵢ + ρ)I + Σᵢ UᵢΛᵢUᵢᵀ
    diag_part = total_diag + ridge

    # For simplicity, if low-rank components exist, use Woodbury
    # Otherwise fall back to diagonal
    if not us:
        return total_low_rank_num / diag_part.unsqueeze(0)

    # Combine all low-rank factors
    U_all = torch.cat(us, dim=1)  # [in, total_k]
    Lambda_all = torch.cat(lambdas)  # [total_k]

    # (D + UΛUᵀ)⁻¹ = D⁻¹ - D⁻¹U(Λ⁻¹ + UᵀD⁻¹U)⁻¹UᵀD⁻¹
    D_inv = 1.0 / diag_part  # [in]
    D_inv_U = D_inv.unsqueeze(0) * U_all.t()  # [k, in]
    inner = torch.diag(1.0 / Lambda_all) + U_all.t() @ D_inv_U  # [k, k]
    inner_inv = torch.linalg.inv(inner)

    # W* = numerator @ (D + UΛUᵀ)⁻¹
    # = numerator @ (D⁻¹ - D⁻¹U inner⁻¹ Uᵀ D⁻¹)
    # = numerator @ D⁻¹ - (numerator @ D⁻¹U) inner⁻¹ (Uᵀ D⁻¹)
    num_D_inv = total_low_rank_num * D_inv.unsqueeze(0)  # [out, in]
    num_D_inv_U = num_D_inv @ U_all  # [out, k]
    correction = num_D_inv_U @ inner_inv @ D_inv_U  # [out, in]
    wt_star = num_D_inv - correction

    return wt_star


# =============================================================================
# RegMean merge entry point
# =============================================================================


def merge_regmean(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    activation_bank: Dict[str, Dict[str, Tensor]],
    device: str = "cpu",
) -> MergeResult:
    """RegMean merge.

    W* = (Σᵢ WᵢCᵢ)(Σᵢ Cᵢ + ρI)⁻¹

    For RegMean-eligible parameters (linear weights), uses activation
    covariance to solve the RegMean optimization.
    For non-eligible parameters, falls back to Task Arithmetic.

    Args:
        base_model: Base model θ₀.
        experts: Specialist models.
        config: Merge configuration (method must be regmean).
        activation_bank: Dict mapping expert_name -> {param_name: covariance}.
        device: Device for computation.

    Returns:
        MergeResult with operator trace.
    """
    if config.method != MergeMethod.REGMEAN:
        raise ValueError(
            f"merge_regmean called with method={config.method}, "
            f"expected regmean"
        )

    n_experts = len(experts)
    validate_parameter_names(experts, base_model)

    if config.lambdas:
        lambdas = list(config.lambdas)
    else:
        lambdas = [1.0 / n_experts] * n_experts

    ridge = config.regmean_ridge
    scale = config.task_scale

    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    base_params = dict(base_cpu.named_parameters())
    task_vectors = extract_task_vectors(experts, base_cpu)

    expert_names = [f"expert_{i}" for i in range(n_experts)]

    merged = copy.deepcopy(base_cpu)
    merged_params = dict(merged.named_parameters())

    regmean_count = 0
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

            # Check if RegMean-eligible and covariance available
            use_regmean = is_regmean_eligible(name, base_param)
            cov_available = all(
                f"expert_{i}" in activation_bank and name in activation_bank[f"expert_{i}"]
                for i in range(n_experts)
            ) if use_regmean else False

            if use_regmean and cov_available:
                # RegMean: solve W* = (Σᵢ WᵢCᵢ)(ΣᵢCᵢ + ρI)⁻¹
                # We work in delta space: Wᵢ = W₀ + Δᵢ
                # W* = (Σᵢ (W₀ + Δᵢ)Cᵢ)(ΣᵢCᵢ + ρI)⁻¹
                #    = W₀ + (Σᵢ ΔᵢCᵢ)(ΣᵢCᵢ + ρI)⁻¹
                expert_weights = [base_param + d for d in deltas]
                covariances = []
                for i in range(n_experts):
                    c = activation_bank[f"expert_{i}"][name].float()
                    covariances.append(c)

                merged_weight = regmean_solve(
                    expert_weights, covariances,
                    ridge=ridge,
                    mode=config.regmean_mode,
                    block_size=config.regmean_block_size,
                    low_rank=config.regmean_low_rank,
                )
                # Apply scale (for consistency with other methods)
                merged_delta = (merged_weight - base_param) * scale
                param.copy_(base_param + merged_delta)
                regmean_count += 1
            else:
                # Fallback: Task Arithmetic for non-eligible params
                merged_delta = sum(deltas)
                param.copy_(base_param + merged_delta * scale)
                ta_count += 1

    merged.to(device)

    trace = OperatorTrace(
        method="regmean",
        operators=["REGMEAN", "TASK_ARITHMETIC_FALLBACK"],
        activation_covariance_used=True,
        fisher_used=False,
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
        method="regmean",
    )
