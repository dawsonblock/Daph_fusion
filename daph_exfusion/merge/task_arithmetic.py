"""Canonical dense Task Arithmetic merge for ExFusion v3.

Definition:
    Δᵢ = θᵢ - θ₀
    θ* = θ₀ + α Σᵢ λᵢ Δᵢ

Production variants:
    TA-0: λᵢ = 1/N, search α only.
    TA-1: search λ on the simplex and α independently.

The merge operator is intentionally simple and side-effect free: it never moves
or mutates the caller's base/expert modules. Search lives in ``task_search``.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from daph_exfusion.merge.types import (
    MergeConfig,
    MergeMethod,
    MergeResult,
    OperatorTrace,
    validate_parameter_names,
)


def _cpu_task_vectors(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
) -> Tuple[Dict[str, Tensor], list[Dict[str, Tensor]]]:
    """Extract FP32 CPU task vectors without changing caller device placement."""
    base_params = {
        name: param.detach().float().cpu()
        for name, param in base_model.named_parameters()
    }
    task_vectors: list[Dict[str, Tensor]] = []
    for expert in experts:
        expert_params = dict(expert.named_parameters())
        task_vectors.append({
            name: expert_params[name].detach().float().cpu() - base_param
            for name, base_param in base_params.items()
        })
    return base_params, task_vectors


def _validate_lambdas(lambdas: Sequence[float], n_experts: int) -> list[float]:
    if n_experts <= 0:
        raise ValueError("Task Arithmetic requires at least one expert")
    if not lambdas:
        return [1.0 / n_experts] * n_experts
    if len(lambdas) != n_experts:
        raise ValueError(f"lambdas length {len(lambdas)} != n_experts {n_experts}")
    values = [float(v) for v in lambdas]
    if not all(torch.isfinite(torch.tensor(v)).item() for v in values):
        raise ValueError("lambdas must be finite")
    return values


def merge_task_arithmetic(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    device: str = "cpu",
) -> MergeResult:
    """Execute θ* = θ₀ + α Σᵢ λᵢ Δᵢ.

    The input modules are treated as immutable. The returned model is a deep
    copy of the base model and is the only object whose parameters/device are
    changed by this function.
    """
    if config.method != MergeMethod.TASK_ARITHMETIC:
        raise ValueError(
            f"merge_task_arithmetic called with method={config.method}, "
            "expected task_arithmetic"
        )

    n_experts = len(experts)
    validate_parameter_names(experts, base_model)
    lambdas = _validate_lambdas(config.lambdas, n_experts)
    scale = float(config.task_scale)
    if not torch.isfinite(torch.tensor(scale)).item():
        raise ValueError("task_scale must be finite")

    base_params_cpu, task_vectors = _cpu_task_vectors(base_model, experts)
    merged = copy.deepcopy(base_model).cpu()

    with torch.no_grad():
        for name, param in merged.named_parameters():
            base_param = base_params_cpu[name]
            merged_delta = torch.zeros_like(base_param)
            for i, task_vector in enumerate(task_vectors):
                merged_delta.add_(task_vector[name], alpha=lambdas[i])
            param.copy_(base_param + scale * merged_delta)

    merged.to(device)

    trace = OperatorTrace(
        method="task_arithmetic",
        operators=["TASK_ARITHMETIC"],
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
        method="task_arithmetic",
    )


def merge_frozen(
    base_model: nn.Module,
    config: MergeConfig,
    device: str = "cpu",
) -> MergeResult:
    """Return an unchanged copy of the base model."""
    merged = copy.deepcopy(base_model).to(device)
    trace = OperatorTrace(
        method="frozen",
        operators=["FROZEN"],
        config_hash=config.config_hash(),
    )
    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method="frozen",
    )


# Backward-compatible alpha-only helper. The production optimizer is
# daph_exfusion.merge.task_search.search_ta1, which jointly searches λ and α.
DEFAULT_SCALE_GRID: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25)


def search_task_arithmetic_scale(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    evaluator,
    scale_grid: Optional[Sequence[float]] = None,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Search α with uniform λ. Retained only as the TA-0 compatibility helper."""
    grid = tuple(scale_grid) if scale_grid is not None else DEFAULT_SCALE_GRID
    if not grid:
        raise ValueError("scale_grid must not be empty")

    best_scale = float(grid[0])
    best_score = float("inf")
    for scale in grid:
        config = MergeConfig(method=MergeMethod.TASK_ARITHMETIC, task_scale=float(scale))
        result = merge_task_arithmetic(base_model, experts, config, device=device)
        score = float(evaluator(result.merged_model))
        if score < best_score:
            best_score = score
            best_scale = float(scale)
    return best_scale, best_score
