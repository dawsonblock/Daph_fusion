"""Production Fisher-weighted Task Arithmetic (TA-2).

This module keeps Fisher as a refinement of Task Arithmetic rather than a
separate merge philosophy:

    Δ*_k = α · [Σ_i λ_i (F_{i,k}+ε)^γ Δ_{i,k}]
                  / [Σ_i λ_i (F_{i,k}+ε)^γ + ε]
    θ* = θ₀ + Δ*

Properties:
- exact same expert-level λ coefficients used by TA-1;
- global task scale α remains independent of λ;
- fail-closed curvature lookup by default;
- caller models are never moved or mutated.
"""
from __future__ import annotations

import copy
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from daph_exfusion.merge.fisher_dense import MissingCurvatureError, stabilize_fisher
from daph_exfusion.merge.types import (
    MergeConfig,
    MergeMethod,
    MergeResult,
    OperatorTrace,
    validate_parameter_names,
)


def _validated_lambdas(config: MergeConfig, n_experts: int) -> list[float]:
    if n_experts <= 0:
        raise ValueError("Fisher Task Arithmetic requires at least one expert")
    if not config.lambdas:
        return [1.0 / n_experts] * n_experts
    if len(config.lambdas) != n_experts:
        raise ValueError(
            f"lambdas length {len(config.lambdas)} != n_experts {n_experts}"
        )
    values = [float(v) for v in config.lambdas]
    if not torch.isfinite(torch.tensor(values)).all().item():
        raise ValueError("lambdas must be finite")
    if any(v < 0 for v in values):
        raise ValueError("production Fisher Task Arithmetic requires non-negative lambdas")
    if sum(values) <= 0:
        raise ValueError("at least one lambda must be positive")
    return values


def _cpu_params(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: p.detach().float().cpu()
        for name, p in model.named_parameters()
    }


def _curvature_for(
    curvature_bank: Dict[str, Dict[str, Tensor]],
    expert_key: str,
    param_name: str,
    reference: Tensor,
    config: MergeConfig,
) -> Tensor:
    bank = curvature_bank.get(expert_key)
    if bank is None or param_name not in bank:
        if not config.allow_missing_fisher:
            raise MissingCurvatureError(expert=expert_key, parameter=param_name)
        raw = torch.ones_like(reference)
    else:
        raw = bank[param_name].detach().float().cpu()
        if raw.shape != reference.shape:
            raise ValueError(
                f"Fisher shape mismatch for {expert_key}:{param_name}: "
                f"{tuple(raw.shape)} != {tuple(reference.shape)}"
            )
        if not torch.isfinite(raw).all().item() or (raw < 0).any().item():
            raise ValueError(
                f"Invalid Fisher values for {expert_key}:{param_name}; "
                "curvature must be finite and non-negative"
            )

    stabilized = stabilize_fisher(
        raw,
        config.fisher_stabilization,
        floor_eps=config.fisher_floor_eps,
        log_alpha=config.fisher_log_alpha,
        clip_quantile=config.fisher_clip_quantile,
    )
    return stabilized.pow(float(config.fisher_gamma))


def merge_fisher_task_arithmetic(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    curvature_bank: Dict[str, Dict[str, Tensor]],
    device: str = "cpu",
) -> MergeResult:
    """Execute production TA-2 without mutating caller modules."""
    if config.method != MergeMethod.FISHER_DENSE:
        raise ValueError(
            f"merge_fisher_task_arithmetic called with method={config.method}, "
            "expected fisher_dense"
        )

    validate_parameter_names(experts, base_model)
    n_experts = len(experts)
    lambdas = _validated_lambdas(config, n_experts)
    scale = float(config.task_scale)
    gamma = float(config.fisher_gamma)
    eps = float(config.fisher_floor_eps)
    if eps <= 0:
        raise ValueError("fisher_floor_eps must be > 0")
    if not torch.isfinite(torch.tensor([scale, gamma, eps])).all().item():
        raise ValueError("task_scale, fisher_gamma, and fisher_floor_eps must be finite")

    base_params = _cpu_params(base_model)
    expert_params = [_cpu_params(expert) for expert in experts]
    merged = copy.deepcopy(base_model).cpu()

    with torch.no_grad():
        for name, out_param in merged.named_parameters():
            base_p = base_params[name]
            numerator = torch.zeros_like(base_p)
            denominator = torch.zeros_like(base_p)

            for i, e_params in enumerate(expert_params):
                delta = e_params[name] - base_p
                precision = _curvature_for(
                    curvature_bank,
                    f"expert_{i}",
                    name,
                    base_p,
                    config,
                )
                weight = lambdas[i] * precision
                numerator.add_(weight * delta)
                denominator.add_(weight)

            merged_delta = numerator / denominator.add(eps)
            out_param.copy_(base_p + scale * merged_delta)

    merged.to(device)
    trace = OperatorTrace(
        method="fisher_dense",
        operators=["EMPIRICAL_FISHER", "FISHER_TASK_ARITHMETIC"],
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
