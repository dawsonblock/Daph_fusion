"""Dense Fisher merge module (Phases 5-9).

Implements:
1. Exact empirical Fisher (micro_batch=1, per-sample gradient squaring)
2. Dense Fisher merge: θ*_k = Σᵢ λᵢ F_{i,k}^γ θ_{i,k} / (Σᵢ λᵢ F_{i,k}^γ + ε)
3. Base-anchored Fisher merge: includes base Fisher F₀ with weight λ₀
4. Fisher stabilization modes (floor, log, clip, power)

Empirical Fisher (rigorous):
    F_k = (1/N) Σₙ (∂Lₙ/∂θ_k)²

Exact reference mode (micro_batch_size=1):
    For each sample:
        model.zero_grad(set_to_none=True)
        out = model(input_ids, attention_mask, labels)
        loss = out.loss
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach().float().square()
    Then divide by sample count.

Do NOT call grad(batch_mean_loss).square() "exact Fisher".
Label it "microbatch_gradient_square" if retained.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from daph_exfusion.merge.types import (
    FisherStabilization,
    MergeConfig,
    MergeMethod,
    MergeResult,
    OperatorTrace,
    extract_task_vectors,
    validate_parameter_names,
)


# =============================================================================
# Exact empirical Fisher (micro_batch=1)
# =============================================================================


@dataclass
class FisherStats:
    """Numerical statistics for a Fisher tensor (corruption detection)."""
    min_val: float
    max_val: float
    mean_val: float
    median_val: float
    fraction_zero: float
    fraction_nonfinite: float
    valid: bool


def compute_fisher_stats(fisher_tensor: Tensor) -> FisherStats:
    """Compute statistics for a Fisher tensor to detect corruption."""
    flat = fisher_tensor.flatten().float()
    finite_mask = torch.isfinite(flat)
    fraction_nonfinite = 1.0 - finite_mask.float().mean().item()
    finite_vals = flat[finite_mask]
    if finite_vals.numel() == 0:
        return FisherStats(
            min_val=0.0, max_val=0.0, mean_val=0.0, median_val=0.0,
            fraction_zero=1.0, fraction_nonfinite=fraction_nonfinite,
            valid=False,
        )
    return FisherStats(
        min_val=finite_vals.min().item(),
        max_val=finite_vals.max().item(),
        mean_val=finite_vals.mean().item(),
        median_val=finite_vals.median().item(),
        fraction_zero=(finite_vals == 0).float().mean().item(),
        fraction_nonfinite=fraction_nonfinite,
        valid=fraction_nonfinite < 0.01,
    )


def build_exact_fisher(
    model: nn.Module,
    dataset: Any,
    forward_fn: Optional[Any] = None,
    loss_fn: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
    max_samples: Optional[int] = None,
    ignore_padding: bool = True,
) -> Tuple[Dict[str, Tensor], Dict[str, FisherStats]]:
    """Build exact empirical Fisher diagonal (micro_batch_size=1).

    F_k = (1/N) Σₙ (∂Lₙ/∂θ_k)²

    This is the rigorous reference implementation. Each sample is processed
    individually (batch_size=1) so that the squared gradient is the per-sample
    gradient squared, NOT the square of a batch-mean gradient.

    Args:
        model: The model to compute Fisher for.
        dataset: Calibration data. Must be iterable yielding batches.
                 Each batch is a dict with 'input_ids', 'attention_mask', 'labels'.
        forward_fn: Optional custom forward function. If None, uses model(input_ids, ...).
        loss_fn: Optional custom loss function. If None, uses model's built-in loss.
        device: Device for computation.
        max_samples: Maximum number of samples to process.
        ignore_padding: If True, set labels[attention_mask==0] = -100.

    Returns:
        (fisher_dict, stats_dict) where fisher_dict maps param_name -> FP32 Fisher tensor.
    """
    model.to(device)
    model.eval()

    # Initialize Fisher accumulators in FP32
    fisher: Dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.detach(), device="cpu", dtype=torch.float32)

    sample_count = 0

    for batch in dataset:
        if max_samples is not None and sample_count >= max_samples:
            break

        # Handle different batch formats
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels", input_ids.clone())
        else:
            input_ids = batch
            attention_mask = None
            labels = input_ids.clone()

        # Process each sample individually (micro_batch=1)
        batch_size = input_ids.shape[0] if input_ids.dim() > 1 else 1
        for s in range(batch_size):
            if max_samples is not None and sample_count >= max_samples:
                break

            single_input = input_ids[s:s+1].to(device) if input_ids.dim() > 1 else input_ids.unsqueeze(0).to(device)
            single_labels = labels[s:s+1].to(device) if labels.dim() > 1 else labels.unsqueeze(0).to(device)
            single_mask = None
            if attention_mask is not None:
                single_mask = attention_mask[s:s+1].to(device) if attention_mask.dim() > 1 else attention_mask.unsqueeze(0).to(device)

            if ignore_padding and single_mask is not None:
                single_labels = single_labels.clone()
                single_labels[single_mask == 0] = -100

            model.zero_grad(set_to_none=True)

            if forward_fn is not None:
                out = forward_fn(model, single_input, attention_mask=single_mask)
            else:
                out = model(
                    input_ids=single_input,
                    attention_mask=single_mask,
                    labels=single_labels,
                )

            if loss_fn is not None:
                loss = loss_fn(out, single_labels)
            elif hasattr(out, "loss"):
                loss = out.loss
            else:
                # Manual cross-entropy with shift
                logits = out.logits if hasattr(out, "logits") else out
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = single_labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )

            loss.backward()

            # Accumulate squared gradients in FP32
            for name, param in model.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += param.grad.detach().float().square().cpu()

            sample_count += 1

    # Divide by sample count
    if sample_count > 0:
        for name in fisher:
            fisher[name] /= sample_count

    # Compute stats
    stats = {name: compute_fisher_stats(f) for name, f in fisher.items()}

    return fisher, stats


# =============================================================================
# Fisher stabilization
# =============================================================================


def stabilize_fisher(
    fisher: Tensor,
    mode: FisherStabilization,
    floor_eps: float = 1e-8,
    log_alpha: float = 1.0,
    clip_quantile: float = 0.999,
) -> Tensor:
    """Apply stabilization to a Fisher diagonal tensor.

    Modes:
        NONE:       F' = F
        FLOOR:      F' = max(F, ε)
        LOG:        F' = log(1 + αF)
        CLIP:       F' = min(F, Q_{q}(F))
        POWER:      F' = F^γ (applied during merge, not here)
    """
    if mode == FisherStabilization.NONE:
        return fisher
    elif mode == FisherStabilization.FLOOR:
        return fisher.clamp(min=floor_eps)
    elif mode == FisherStabilization.LOG_COMPRESS:
        return torch.log1p(log_alpha * fisher.clamp(min=0))
    elif mode == FisherStabilization.QUANTILE_CLIP:
        flat = fisher.flatten()
        if flat.numel() > 0:
            q_val = torch.quantile(flat.float(), clip_quantile).item()
            return fisher.clamp(max=q_val)
        return fisher
    elif mode == FisherStabilization.POWER:
        # Power is applied during merge via gamma exponent
        return fisher
    return fisher


# =============================================================================
# Dense Fisher merge
# =============================================================================


def merge_fisher_dense(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    curvature_bank: Dict[str, Dict[str, Tensor]],
    device: str = "cpu",
) -> MergeResult:
    """Dense Fisher merge (expert-only precision weighting).

    θ*_k = Σᵢ λᵢ F_{i,k}^γ θ_{i,k} / (Σᵢ λᵢ F_{i,k}^γ + ε)

    Args:
        base_model: Base model θ₀.
        experts: Specialist models.
        config: Merge configuration (method must be fisher_dense).
        curvature_bank: Dict mapping expert_name -> {param_name: Fisher diagonal}.
        device: Device for computation.

    Returns:
        MergeResult with operator trace.
    """
    if config.method != MergeMethod.FISHER_DENSE:
        raise ValueError(
            f"merge_fisher_dense called with method={config.method}, "
            f"expected fisher_dense"
        )

    n_experts = len(experts)
    validate_parameter_names(experts, base_model)

    if config.lambdas:
        lambdas = list(config.lambdas)
    else:
        lambdas = [1.0 / n_experts] * n_experts

    gamma = config.fisher_gamma
    eps = config.fisher_floor_eps
    scale = config.task_scale

    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    task_vectors = extract_task_vectors(experts, base_cpu)

    expert_names = [f"expert_{i}" for i in range(n_experts)]

    merged = copy.deepcopy(base_cpu)
    merged_params = dict(merged.named_parameters())

    with torch.no_grad():
        for name, param in merged_params.items():
            deltas = []
            fishers = []
            for i, tv in enumerate(task_vectors):
                if name in tv:
                    deltas.append(tv[name])
                    expert_key = expert_names[i]
                    if expert_key in curvature_bank and name in curvature_bank[expert_key]:
                        f = curvature_bank[expert_key][name].float()
                        f = stabilize_fisher(f, config.fisher_stabilization,
                                             floor_eps=eps,
                                             log_alpha=config.fisher_log_alpha,
                                             clip_quantile=config.fisher_clip_quantile)
                        fishers.append(f.pow(gamma) * lambdas[i])
                    else:
                        fishers.append(torch.ones_like(tv[name]) * lambdas[i])

            if not deltas:
                continue

            # Dense Fisher: weighted average of expert parameters
            # θ*_k = Σᵢ λᵢ F_{i,k}^γ θ_{i,k} / (Σᵢ λᵢ F_{i,k}^γ + ε)
            # In delta form: Δ*_k = Σᵢ wᵢ Δᵢ where wᵢ = Fᵢ^γ / Σ Fⱼ^γ
            total_fisher = sum(fishers) + eps
            weighted_delta = sum(f * d for f, d in zip(fishers, deltas)) / total_fisher
            param.copy_(param.detach().float() + weighted_delta * scale)

    merged.to(device)

    trace = OperatorTrace(
        method="fisher_dense",
        operators=["EMPIRICAL_FISHER", "DENSE_PRECISION_MERGE"],
        fisher_used=True,
        fisher_estimator="exact_per_sample",
        task_scale=scale,
        fisher_gamma=gamma,
        lambdas=lambdas,
        config_hash=config.config_hash(),
    )

    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method="fisher_dense",
    )


# =============================================================================
# Base-anchored Fisher merge
# =============================================================================


def merge_fisher_base_anchored(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    curvature_bank: Dict[str, Dict[str, Tensor]],
    base_fisher: Dict[str, Tensor],
    device: str = "cpu",
) -> MergeResult:
    """Base-anchored Fisher merge.

    θ*_k = (λ₀ F_{0,k}^γ θ_{0,k} + Σᵢ λᵢ F_{i,k}^γ θ_{i,k})
           / (λ₀ F_{0,k}^γ + Σᵢ λᵢ F_{i,k}^γ + ε)

    This explicitly encodes: retain the pretrained model where it is confident.

    Args:
        base_model: Base model θ₀.
        experts: Specialist models.
        config: Merge configuration (method must be fisher_base_anchored).
        curvature_bank: Dict mapping expert_name -> {param_name: Fisher diagonal}.
        base_fisher: Dict mapping param_name -> base model Fisher diagonal.
        device: Device for computation.

    Returns:
        MergeResult with operator trace.
    """
    if config.method != MergeMethod.FISHER_BASE_ANCHORED:
        raise ValueError(
            f"merge_fisher_base_anchored called with method={config.method}, "
            f"expected fisher_base_anchored"
        )

    n_experts = len(experts)
    validate_parameter_names(experts, base_model)

    if config.lambdas:
        lambdas = list(config.lambdas)
    else:
        lambdas = [1.0 / n_experts] * n_experts

    lambda_0 = config.base_precision_weight
    gamma = config.fisher_gamma
    eps = config.fisher_floor_eps
    scale = config.task_scale

    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    base_params = dict(base_cpu.named_parameters())
    task_vectors = extract_task_vectors(experts, base_cpu)

    expert_names = [f"expert_{i}" for i in range(n_experts)]

    merged = copy.deepcopy(base_cpu)
    merged_params = dict(merged.named_parameters())

    with torch.no_grad():
        for name, param in merged_params.items():
            deltas = []
            expert_fishers = []
            for i, tv in enumerate(task_vectors):
                if name in tv:
                    deltas.append(tv[name])
                    expert_key = expert_names[i]
                    if expert_key in curvature_bank and name in curvature_bank[expert_key]:
                        f = curvature_bank[expert_key][name].float()
                        f = stabilize_fisher(f, config.fisher_stabilization,
                                             floor_eps=eps,
                                             log_alpha=config.fisher_log_alpha,
                                             clip_quantile=config.fisher_clip_quantile)
                        expert_fishers.append(f.pow(gamma) * lambdas[i])
                    else:
                        expert_fishers.append(torch.ones_like(tv[name]) * lambdas[i])

            if not deltas:
                continue

            # Base Fisher
            base_f = None
            if name in base_fisher:
                base_f = base_fisher[name].float()
                base_f = stabilize_fisher(base_f, config.fisher_stabilization,
                                          floor_eps=eps,
                                          log_alpha=config.fisher_log_alpha,
                                          clip_quantile=config.fisher_clip_quantile)
                base_f = base_f.pow(gamma) * lambda_0
            else:
                base_f = torch.ones_like(param.detach().float()) * lambda_0

            base_param = base_params[name].detach().float()

            # θ*_k = (λ₀ F₀^γ θ₀ + Σᵢ λᵢ Fᵢ^γ θᵢ) / (λ₀ F₀^γ + Σᵢ λᵢ Fᵢ^γ + ε)
            # In delta form:
            # θ* = θ₀ + (Σᵢ λᵢ Fᵢ^γ Δᵢ) / (λ₀ F₀^γ + Σᵢ λᵢ Fᵢ^γ + ε)
            total_weight = base_f + sum(expert_fishers) + eps
            weighted_delta = sum(f * d for f, d in zip(expert_fishers, deltas)) / total_weight
            param.copy_(base_param + weighted_delta * scale)

    merged.to(device)

    trace = OperatorTrace(
        method="fisher_base_anchored",
        operators=["EMPIRICAL_FISHER", "BASE_ANCHOR", "DENSE_PRECISION_MERGE"],
        fisher_used=True,
        fisher_estimator="exact_per_sample",
        task_scale=scale,
        fisher_gamma=gamma,
        base_precision_weight=lambda_0,
        lambdas=lambdas,
        config_hash=config.config_hash(),
    )

    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method="fisher_base_anchored",
    )
