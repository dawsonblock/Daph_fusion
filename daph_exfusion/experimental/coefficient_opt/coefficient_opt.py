"""Dense Coefficient Optimization module for ExFusion v3.

Optimizes expert coefficients λ on the REAL merged model forward pass,
not an approximation.

Definition:
    θ(λ) = θ₀ + Σᵢ λᵢ Δᵢ

where Δᵢ = θᵢ - θ₀ are task vectors and λ are learned per-group
coefficients.

Objective:
    L_specialist(λ) = (1/D) Σ_d L_d(θ(λ))

evaluated on the actual merged model via ``torch.func.functional_call``,
NOT approximated as h_merged = h₀ + Σᵢ λᵢ(hᵢ - h₀).

Granularity:
    GLOBAL  — one λ per expert                        (N variables)
    FAMILY  — one λ per expert per architecture family (N×F variables)
    LAYER   — one λ per expert per layer               (N×L variables)

Parameterization:
    SOFTMAX       — λ_i = exp(a_i) / Σ exp(a_j)  (convex, per group)
    SIGMOID       — λ_i = σ(a_i)                 (independent, allows Σ>1)
    SIGNED        — λ_i = tanh(a_i)              (allows extrapolation)
    UNCONSTRAINED — λ_i = a_i                    (no constraints)
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call

from daph_exfusion.merge.types import (
    CoefficientGranularity,
    CoefficientParameterization,
    MergeConfig,
    MergeMethod,
    MergeResult,
    OperatorTrace,
    classify_parameter_family,
    count_layers,
    extract_task_vectors,
    get_layer_index,
    validate_parameter_names,
)


# =============================================================================
# Coarse family groups for FAMILY granularity
# =============================================================================


_COARSE_FAMILIES: Tuple[str, ...] = ("attention", "ffn", "ssm", "other")
_COARSE_FAMILY_INDEX: Dict[str, int] = {f: i for i, f in enumerate(_COARSE_FAMILIES)}


def _to_coarse_family(detailed_family: str) -> str:
    """Map a detailed ``classify_parameter_family`` result to a coarse group."""
    if "attention" in detailed_family:
        return "attention"
    if "ffn" in detailed_family:
        return "ffn"
    if "ssm" in detailed_family:
        return "ssm"
    return "other"


# =============================================================================
# Parameter-to-group mapping
# =============================================================================


def _build_param_to_group(
    param_names: Sequence[str],
    granularity: CoefficientGranularity,
    num_layers: int,
) -> Dict[str, int]:
    """Map each parameter name to a group index based on granularity.

    GLOBAL:  all parameters share group 0.
    FAMILY:  group = coarse family index ∈ {attention, ffn, ssm, other}.
    LAYER:   group = layer_idx + 1 (group 0 reserved for non-layer params
             such as embeddings, lm_head, normalization).
    """
    if granularity == CoefficientGranularity.GLOBAL:
        return {name: 0 for name in param_names}

    if granularity == CoefficientGranularity.FAMILY:
        groups: Dict[str, int] = {}
        for name in param_names:
            layer_idx = get_layer_index(name)
            detailed = classify_parameter_family(name, layer_idx, num_layers)
            coarse = _to_coarse_family(detailed)
            groups[name] = _COARSE_FAMILY_INDEX[coarse]
        return groups

    groups = {}
    for name in param_names:
        layer_idx = get_layer_index(name)
        if layer_idx < 0:
            groups[name] = 0
        else:
            groups[name] = layer_idx + 1
    return groups


def _count_groups(
    granularity: CoefficientGranularity,
    num_layers: int,
    param_to_group: Dict[str, int],
) -> int:
    """Return the number of coefficient groups for the given granularity."""
    if granularity == CoefficientGranularity.GLOBAL:
        return 1
    if granularity == CoefficientGranularity.FAMILY:
        return len(_COARSE_FAMILIES)
    return max(param_to_group.values()) + 1


# =============================================================================
# Coefficient parameterization
# =============================================================================


def _parameterize_coeffs(
    raw: Tensor,
    parameterization: CoefficientParameterization,
) -> Tensor:
    """Convert raw coefficients ``a`` to λ via the parameterization.

    Args:
        raw: shape ``(n_experts, n_groups)``.
        parameterization: how to map raw values to λ.

    Returns:
        Tensor of shape ``(n_experts, n_groups)``.

    SOFTMAX:       λ_i = exp(a_i) / Σ_j exp(a_j)  — convex, per group
    SIGMOID:       λ_i = σ(a_i)                   — independent
    SIGNED:        λ_i = tanh(a_i)                — allows extrapolation
    UNCONSTRAINED: λ_i = a_i                      — no constraints
    """
    if parameterization == CoefficientParameterization.SOFTMAX:
        return torch.softmax(raw, dim=0)
    if parameterization == CoefficientParameterization.SIGMOID:
        return torch.sigmoid(raw)
    if parameterization == CoefficientParameterization.SIGNED:
        return torch.tanh(raw)
    return raw


def _init_raw_value(
    parameterization: CoefficientParameterization,
    n_experts: int,
) -> float:
    """Initial raw value so that λ ≈ 1/N at the start of optimization."""
    if parameterization == CoefficientParameterization.SOFTMAX:
        return 0.0
    if parameterization == CoefficientParameterization.SIGMOID:
        if n_experts <= 1:
            return 4.0
        return -math.log(n_experts - 1)
    if parameterization == CoefficientParameterization.SIGNED:
        if n_experts <= 1:
            return 4.0
        return math.atanh(1.0 / n_experts)
    return 1.0 / n_experts


# =============================================================================
# Merged parameter construction (differentiable)
# =============================================================================


def _build_merged_params(
    base_params: Dict[str, Tensor],
    task_vectors: List[Dict[str, Tensor]],
    lambdas: Tensor,
    param_to_group: Dict[str, int],
    scale: float,
) -> Dict[str, Tensor]:
    """Build θ(λ) = θ₀ + scale · Σᵢ λᵢ Δᵢ for all parameters.

    Args:
        base_params: ``{name: θ₀_name}`` (detached, FP32, on device).
        task_vectors: list of ``{name: Δᵢ_name}`` per expert.
        lambdas: shape ``(n_experts, n_groups)`` — differentiable w.r.t. raw coeffs.
        param_to_group: mapping from parameter name to group index.
        scale: task scale α.

    Returns:
        ``{name: θ(λ)_name}`` — differentiable tensors connected to ``lambdas``.
    """
    n_experts = len(task_vectors)
    merged: Dict[str, Tensor] = {}
    for name, base_p in base_params.items():
        g = param_to_group[name]
        delta = torch.zeros_like(base_p)
        for i in range(n_experts):
            if name in task_vectors[i]:
                delta = delta + lambdas[i, g] * task_vectors[i][name]
        merged[name] = base_p + scale * delta
    return merged


# =============================================================================
# Functional forward pass
# =============================================================================


def _functional_forward(
    model: nn.Module,
    params: Dict[str, Tensor],
    batch: Any,
    device: Union[str, torch.device],
) -> Any:
    """Run a functional forward pass via ``torch.func.functional_call``.

    This evaluates the model with ``params`` substituted for its named
    parameters, without copying or mutating the module.
    """
    if isinstance(batch, dict):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
    else:
        input_ids = batch.to(device)
        attention_mask = None
        labels = None

    kwargs: Dict[str, Any] = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(device)
    if labels is not None:
        kwargs["labels"] = labels.to(device)

    return functional_call(model, params, args=(input_ids,), kwargs=kwargs)


def _compute_loss(
    output: Any,
    batch: Any,
    evaluator: Optional[Callable[[Any, Any], Tensor]],
    device: Union[str, torch.device],
) -> Tensor:
    """Compute a scalar loss from the model output.

    If ``evaluator`` is provided, it is called as ``evaluator(output, batch)``
    and must return a scalar ``Tensor``.

    Otherwise, the loss is taken from ``output.loss`` if available, or
    computed as shifted cross-entropy on ``output.logits``.
    """
    if evaluator is not None:
        return evaluator(output, batch)

    if hasattr(output, "loss") and output.loss is not None:
        return output.loss

    logits = output.logits if hasattr(output, "logits") else output
    if isinstance(batch, dict):
        labels = batch["labels"].to(device)
    else:
        labels = batch.to(device)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


# =============================================================================
# Main merge entry point
# =============================================================================


def merge_coefficient_opt(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    calibration_data: Any,
    evaluator: Optional[Callable[[Any, Any], Tensor]] = None,
    device: str = "cpu",
) -> MergeResult:
    """Execute Dense Coefficient Optimization merge.

    θ(λ) = θ₀ + Σᵢ λᵢ Δᵢ

    Optimizes coefficients λ on the **real** merged model forward pass using
    ``torch.func.functional_call``, without copying the model each iteration.

    Objective:
        L_specialist(λ) = (1/D) Σ_d L_d(θ(λ))

    The loss L_d(θ(λ)) is computed on the actual merged model — NOT
    approximated as h_merged = h₀ + Σᵢ λᵢ(hᵢ - h₀).

    Args:
        base_model: Base model θ₀.
        experts: List of specialist models θᵢ.
        config: Merge configuration (method must be ``coefficient_opt``).
            Relevant fields: ``coefficient_granularity``,
            ``coefficient_parameterization``, ``coefficient_lr``,
            ``coefficient_steps``, ``task_scale``, ``lambdas``, ``seed``.
        calibration_data: Iterable of batches for the forward pass.
            Each batch is a dict with ``input_ids`` and optionally
            ``attention_mask`` / ``labels``, or a raw ``Tensor`` of input ids.
        evaluator: Optional callable ``(output, batch) -> scalar loss Tensor``.
            If ``None``, uses the model's built-in loss or manual cross-entropy.
        device: Device for computation.

    Returns:
        ``MergeResult`` with the merged model and operator trace.

    Raises:
        ValueError: if ``config.method`` is not ``coefficient_opt``.
    """
    if config.method != MergeMethod.COEFFICIENT_OPT:
        raise ValueError(
            f"merge_coefficient_opt called with method={config.method}, "
            f"expected coefficient_opt"
        )

    n_experts = len(experts)
    if n_experts == 0:
        raise ValueError("merge_coefficient_opt requires at least one expert")

    validate_parameter_names(experts, base_model)
    torch.manual_seed(config.seed)

    scale = config.task_scale
    granularity = config.coefficient_granularity
    parameterization = config.coefficient_parameterization

    # Use a deepcopy on the device for functional_call forward passes so the
    # original base_model is never mutated back and forth (which causes device
    # mismatches on GPU). base_cpu stays on CPU for task vector extraction.
    working_model = copy.deepcopy(base_model).to(device)
    working_model.eval()

    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    task_vectors_cpu = extract_task_vectors(experts, base_cpu)

    base_params: Dict[str, Tensor] = {
        name: p.detach().float().to(device)
        for name, p in working_model.named_parameters()
    }
    task_vectors: List[Dict[str, Tensor]] = [
        {name: tv.to(device) for name, tv in tv_dict.items()}
        for tv_dict in task_vectors_cpu
    ]

    num_layers = count_layers(working_model)
    param_names = list(base_params.keys())
    param_to_group = _build_param_to_group(param_names, granularity, num_layers)
    n_groups = _count_groups(granularity, num_layers, param_to_group)

    init_val = _init_raw_value(parameterization, n_experts)
    raw_coeffs = nn.Parameter(
        torch.full(
            (n_experts, n_groups),
            init_val,
            device=device,
            dtype=torch.float32,
        )
    )

    optimizer = torch.optim.Adam([raw_coeffs], lr=config.coefficient_lr)

    if not isinstance(calibration_data, (list, tuple)):
        calibration_data = list(calibration_data)

    for _step in range(config.coefficient_steps):
        optimizer.zero_grad()

        lambdas = _parameterize_coeffs(raw_coeffs, parameterization)
        merged_params = _build_merged_params(
            base_params, task_vectors, lambdas, param_to_group, scale
        )

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        n_batches = 0
        for batch in calibration_data:
            output = _functional_forward(working_model, merged_params, batch, device)
            loss = _compute_loss(output, batch, evaluator, device)
            total_loss = total_loss + loss
            n_batches += 1

        if n_batches == 0:
            raise ValueError("calibration_data yielded no batches")

        total_loss = total_loss / n_batches
        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        lambdas_final = _parameterize_coeffs(raw_coeffs, parameterization).detach()
        lambdas_cpu = lambdas_final.cpu()

    merged = copy.deepcopy(base_cpu)
    merged_params_dict = dict(merged.named_parameters())

    with torch.no_grad():
        for name, param in merged_params_dict.items():
            g = param_to_group[name]
            delta = torch.zeros_like(param.detach().float())
            for i in range(n_experts):
                if name in task_vectors_cpu[i]:
                    delta = delta + lambdas_cpu[i, g] * task_vectors_cpu[i][name]
            param.copy_(param.detach().float() + scale * delta)

    merged.to(device)

    if granularity == CoefficientGranularity.GLOBAL:
        trace_lambdas = lambdas_cpu[:, 0].tolist()
    else:
        trace_lambdas = lambdas_cpu.flatten().tolist()

    trace = OperatorTrace(
        method="coefficient_opt",
        operators=["COEFFICIENT_OPT"],
        task_scale=scale,
        lambdas=trace_lambdas,
        coefficient_granularity=granularity.value,
        coefficient_parameterization=parameterization.value,
        config_hash=config.config_hash(),
    )

    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method="coefficient_opt",
    )
