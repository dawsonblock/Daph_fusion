"""RegMean++ merge module — propagation-aware recalibration.

RegMean assumes the merged layer receives the same input distribution as the
corresponding expert. This assumption breaks after earlier layers are merged:
the output of merged layer L becomes the input to layer L+1, so layer L+1's
input distribution diverges from what each expert produced.

RegMean++ addresses this by recalibrating deeper layers using activations
collected from the partially merged model:

    merge layer 0 → run calibration → collect layer 1 merged-input
    activations → merge layer 1 → run calibration → collect layer 2
    activations → ...

For the first layer, per-expert activation covariances C_i are used (same as
standard RegMean). For layer L+1 onward, the covariance C_merged is collected
from the partially merged model's forward pass and used for all experts:

    W*_{L+1} = (Σ_i λ_i W_i C_merged)(Σ_i λ_i C_merged + ρI)^{-1}

Non-RegMean-eligible parameters (norms, biases, embeddings, SSM recurrence)
fall back to Task Arithmetic: θ* = θ_0 + α Σ_i λ_i Δ_i.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from daph_exfusion.merge.regmean import regmean_solve, is_regmean_eligible


_KFAC_EXCLUDE_PATTERNS = (
    "norm", "ln", "rms", "layernorm", "bias",
    "embed", "wte", "position", "pos_embed",
    "lm_head", "output",
    "a_log", "dt_proj",
    ".d",
)


def _get_layer_index(param_name: str) -> int:
    """Extract the layer index from a parameter name.

    Handles both 'model.layers.N.*' and 'h.N.*' naming conventions.
    Returns -1 if no layer index is found.
    """
    parts = param_name.split(".")
    for i, p in enumerate(parts):
        if p in ("layers", "h") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def _count_layers(model: nn.Module) -> int:
    """Count the number of transformer/mamba layers in a model."""
    max_layer = -1
    for name in dict(model.named_parameters()).keys():
        idx = _get_layer_index(name)
        if idx > max_layer:
            max_layer = idx
    return max_layer + 1 if max_layer >= 0 else 0


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


def _collect_activation_covariances(
    model: nn.Module,
    calibration_data: Any,
    forward_fn: Optional[Callable],
    target_param_names: List[str],
    device: str,
    max_batches: int = 10,
) -> Dict[str, Tensor]:
    """Run a forward pass on calibration data and collect input activation
    covariances E[aa^T] for the specified parameters.

    Registers forward hooks on the linear modules corresponding to
    target_param_names, runs the model on calibration data, and computes
    the covariance of the input activations for each module.

    Args:
        model: The (partially) merged model to run forward on.
        calibration_data: Iterable of batches. Each batch is a dict with
            'input_ids', 'attention_mask' or a raw tensor.
        forward_fn: Callable(model, input_ids, attention_mask) -> output.
            If None, calls model(input_ids, attention_mask=...).
        target_param_names: Parameter names to collect covariances for.
        device: Device for the forward pass.
        max_batches: Maximum number of batches to process.

    Returns:
        Dict mapping param_name -> covariance tensor [in_dim, in_dim].
    """
    named_modules = dict(model.named_modules())
    param_to_module: Dict[str, nn.Module] = {}
    for param_name in target_param_names:
        module_path = param_name.rsplit(".", 1)[0]
        module = named_modules.get(module_path)
        if module is not None:
            param_to_module[param_name] = module

    if not param_to_module:
        return {}

    collected: Dict[str, List[Tensor]] = {name: [] for name in param_to_module}

    def _make_hook(pname: str):
        def _hook(module: nn.Module, inp: Tuple, out: Any) -> None:
            x = inp[0]
            if not isinstance(x, Tensor):
                return
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            collected[pname].append(x.detach().float().cpu())
        return _hook

    hooks = []
    for pname, module in param_to_module.items():
        h = module.register_forward_hook(_make_hook(pname))
        hooks.append(h)

    model.to(device)
    model.eval()
    batch_count = 0
    try:
        with torch.no_grad():
            for batch in calibration_data:
                if batch_count >= max_batches:
                    break
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    attention_mask = batch.get("attention_mask")
                else:
                    input_ids = batch
                    attention_mask = None

                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                if forward_fn is not None:
                    forward_fn(model, input_ids, attention_mask)
                else:
                    model(input_ids, attention_mask=attention_mask)

                batch_count += 1
    finally:
        for h in hooks:
            h.remove()
        model.cpu()

    covariances: Dict[str, Tensor] = {}
    for pname, acts in collected.items():
        if not acts:
            continue
        all_acts = torch.cat(acts, dim=0)
        n = all_acts.shape[0]
        cov = (all_acts.t() @ all_acts) / max(n, 1)
        covariances[pname] = cov

    return covariances


def merge_regmean_pp(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    activation_bank: Dict[str, Dict[str, Tensor]],
    calibration_data: Any,
    forward_fn: Optional[Callable],
    device: str = "cpu",
) -> MergeResult:
    """RegMean++ merge with propagation-aware recalibration.

    Processes layers sequentially from first to last. After merging each
    layer, runs a forward pass on calibration data to collect updated
    activations for the next layer's RegMean solve. This recalibrates
    deeper layers using activations from the partially merged model.

    For layer 0, per-expert activation covariances C_i from activation_bank
    are used (standard RegMean). For layer L+1 onward, the covariance
    C_merged is collected from the partially merged model and used for all
    experts:

        W*_{L+1} = (Σ_i λ_i W_i C_merged)(Σ_i λ_i C_merged + ρI)^{-1}

    Non-RegMean-eligible parameters fall back to Task Arithmetic.

    Args:
        base_model: Base model θ_0.
        experts: Specialist models.
        config: Merge configuration (method must be regmean_pp).
        activation_bank: Dict mapping expert_name -> {param_name: covariance}.
        calibration_data: Iterable of batches for recalibration forward passes.
        forward_fn: Callable(model, input_ids, attention_mask) for forward pass.
        device: Device for computation.

    Returns:
        MergeResult with operator trace.
    """
    if config.method != MergeMethod.REGMEAN_PP:
        raise ValueError(
            f"merge_regmean_pp called with method={config.method}, "
            f"expected regmean_pp"
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

    num_layers = _count_layers(base_cpu)
    layer_params: Dict[int, List[str]] = {i: [] for i in range(num_layers)}
    non_layer_params: List[str] = []

    for name in merged_params:
        idx = _get_layer_index(name)
        if idx >= 0:
            layer_params[idx].append(name)
        else:
            non_layer_params.append(name)

    regmean_count = 0
    ta_count = 0
    recalibration_count = 0

    with torch.no_grad():
        for name in non_layer_params:
            deltas = []
            for i, tv in enumerate(task_vectors):
                if name in tv:
                    deltas.append(tv[name] * lambdas[i])
            if not deltas:
                continue
            base_param = base_params[name].detach().float()
            merged_delta = sum(deltas)
            merged_params[name].copy_(base_param + merged_delta * scale)
            ta_count += 1

        current_cov_bank: Dict[str, Dict[str, Tensor]] = {
            ename: dict(activation_bank.get(ename, {})) for ename in expert_names
        }

        for layer_idx in range(num_layers):
            params_in_layer = layer_params[layer_idx]
            regmean_params_merged: List[str] = []

            for name in params_in_layer:
                base_param = base_params[name].detach().float()
                use_regmean = is_regmean_eligible(name, base_param)

                if use_regmean:
                    cov_available = all(
                        ename in current_cov_bank and name in current_cov_bank[ename]
                        for ename in expert_names
                    )
                    if cov_available:
                        expert_weights = [
                            (base_param + tv[name]) for tv in task_vectors
                        ]
                        covariances = [
                            current_cov_bank[ename][name].float()
                            for ename in expert_names
                        ]
                        merged_weight = regmean_solve(
                            expert_weights,
                            covariances,
                            ridge=ridge,
                            mode=config.regmean_mode,
                            block_size=config.regmean_block_size,
                            low_rank=config.regmean_low_rank,
                        )
                        merged_delta = (merged_weight - base_param) * scale
                        merged_params[name].copy_(base_param + merged_delta)
                        regmean_count += 1
                        regmean_params_merged.append(name)
                    else:
                        deltas = [
                            tv[name] * lambdas[i]
                            for i, tv in enumerate(task_vectors) if name in tv
                        ]
                        if not deltas:
                            continue
                        merged_delta = sum(deltas)
                        merged_params[name].copy_(base_param + merged_delta * scale)
                        ta_count += 1
                else:
                    deltas = [
                        tv[name] * lambdas[i]
                        for i, tv in enumerate(task_vectors) if name in tv
                    ]
                    if not deltas:
                        continue
                    merged_delta = sum(deltas)
                    merged_params[name].copy_(base_param + merged_delta * scale)
                    ta_count += 1

            if layer_idx < num_layers - 1 and calibration_data is not None:
                next_regmean_params: List[str] = []
                for name in layer_params[layer_idx + 1]:
                    base_param = base_params[name].detach().float()
                    if is_regmean_eligible(name, base_param):
                        next_regmean_params.append(name)

                if next_regmean_params and regmean_params_merged:
                    updated_covs = _collect_activation_covariances(
                        merged,
                        calibration_data,
                        forward_fn,
                        next_regmean_params,
                        device,
                    )

                    if updated_covs:
                        for pname, cov in updated_covs.items():
                            for ename in expert_names:
                                current_cov_bank[ename][pname] = cov
                        recalibration_count += 1

    merged.to(device)

    trace = OperatorTrace(
        method="regmean_pp",
        operators=["REGMEAN_PP", "PROPAGATION_RECALIBRATION"],
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
        method="regmean_pp",
    )
