"""Canonical ExFusion v3 merge dispatcher.

Production path:
    TA-0/TA-1  task_arithmetic
    TA-2       fisher_dense (implemented as Fisher-weighted Task Arithmetic)
    TA-3       family-weighted Task Arithmetic via task_search

Legacy sparse methods remain controlled baselines. Research methods are lazily
loaded from ``daph_exfusion.experimental`` and are not part of the production
selection path.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence

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
from daph_exfusion.merge.task_arithmetic import merge_frozen, merge_task_arithmetic
from daph_exfusion.merge.fisher_task_arithmetic import merge_fisher_task_arithmetic
from daph_exfusion.merge.fisher_dense import merge_fisher_base_anchored


def merge_experts(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    curvature_bank: Optional[Dict[str, Dict[str, Tensor]]] = None,
    base_fisher: Optional[Dict[str, Tensor]] = None,
    activation_bank: Optional[Dict[str, Dict[str, Tensor]]] = None,
    kfac_bank: Optional[Dict[str, Dict[str, Any]]] = None,
    calibration_data: Optional[Any] = None,
    evaluator: Optional[Callable] = None,
    forward_fn: Optional[Callable] = None,
    device: str = "cpu",
) -> MergeResult:
    """Single canonical entry point for all v3 merge operations."""
    method = config.method

    # Production dense methods.
    if method == MergeMethod.TASK_ARITHMETIC:
        return merge_task_arithmetic(base_model, experts, config, device=device)
    if method == MergeMethod.FROZEN:
        return merge_frozen(base_model, config, device=device)
    if method == MergeMethod.FISHER_DENSE:
        if curvature_bank is None:
            raise ValueError("fisher_dense requires curvature_bank")
        return merge_fisher_task_arithmetic(
            base_model, experts, config, curvature_bank, device=device
        )
    if method == MergeMethod.FISHER_BASE_ANCHORED:
        if curvature_bank is None:
            raise ValueError("fisher_base_anchored requires curvature_bank")
        if base_fisher is None:
            raise ValueError("fisher_base_anchored requires base_fisher")
        # The base-anchored implementation is retained as an explicit alternate
        # baseline. Clone inputs to prevent its legacy CPU staging from changing
        # caller module placement.
        return merge_fisher_base_anchored(
            copy.deepcopy(base_model),
            [copy.deepcopy(e) for e in experts],
            config,
            curvature_bank,
            base_fisher,
            device=device,
        )

    # Controlled legacy baselines.
    if method in (
        MergeMethod.DARE,
        MergeMethod.TIES_MAGNITUDE,
        MergeMethod.TIES_MAJORITY,
        MergeMethod.DARE_TIES,
    ):
        return _merge_legacy_baseline(base_model, experts, config, device)

    # Experimental research methods. Imported lazily so production TA users do
    # not pay their dependency or import cost.
    if method == MergeMethod.REGMEAN:
        from daph_exfusion.experimental.regmean.regmean import merge_regmean
        if activation_bank is None:
            raise ValueError("regmean requires activation_bank")
        return merge_regmean(base_model, experts, config, activation_bank, device=device)

    if method == MergeMethod.REGMEAN_PP:
        from daph_exfusion.experimental.regmean.regmean_pp import merge_regmean_pp
        if activation_bank is None:
            raise ValueError("regmean_pp requires activation_bank")
        if calibration_data is None:
            raise ValueError("regmean_pp requires calibration_data")
        return merge_regmean_pp(
            base_model,
            experts,
            config,
            activation_bank,
            calibration_data,
            forward_fn=forward_fn,
            device=device,
        )

    if method == MergeMethod.COEFFICIENT_OPT:
        from daph_exfusion.experimental.coefficient_opt.coefficient_opt import merge_coefficient_opt
        if calibration_data is None:
            raise ValueError("coefficient_opt requires calibration_data")
        return merge_coefficient_opt(
            base_model,
            experts,
            config,
            calibration_data,
            evaluator,
            device=device,
        )

    if method == MergeMethod.TRUST_REGION:
        from daph_exfusion.experimental.trust_region.trust_region import merge_trust_region
        if base_fisher is None:
            raise ValueError("trust_region requires base_fisher")
        if calibration_data is None:
            raise ValueError("trust_region requires calibration_data")
        if evaluator is None:
            raise ValueError("trust_region requires evaluator")
        return merge_trust_region(
            base_model,
            experts,
            config,
            base_fisher,
            calibration_data,
            evaluator,
            device=device,
        )

    if method == MergeMethod.KFAC_BARYCENTER:
        from daph_exfusion.experimental.kfac.kfac_merge import merge_kfac
        if kfac_bank is None:
            raise ValueError("kfac_barycenter requires kfac_bank")
        return merge_kfac(base_model, experts, config, kfac_bank, device=device)

    if method == MergeMethod.AGX:
        from daph_exfusion.experimental.agx.agx import merge_agx
        return merge_agx(
            base_model,
            experts,
            config,
            curvature_bank=curvature_bank,
            base_fisher=base_fisher,
            activation_bank=activation_bank,
            calibration_data=calibration_data,
            evaluator=evaluator,
            device=device,
        )

    if method == MergeMethod.EMR:
        raise NotImplementedError("EMR not implemented in v3 pipeline")
    if method == MergeMethod.MODEL_STOCK:
        raise NotImplementedError("Model Stock not implemented in v3 pipeline")
    if method == MergeMethod.SLERP:
        raise NotImplementedError("SLERP not implemented in v3 pipeline")

    raise ValueError(f"Unknown merge method: {method!r}")


def _cpu_task_vectors_no_mutation(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
) -> tuple[Dict[str, Tensor], list[Dict[str, Tensor]]]:
    base_params = {
        name: p.detach().float().cpu()
        for name, p in base_model.named_parameters()
    }
    task_vectors: list[Dict[str, Tensor]] = []
    for expert in experts:
        e_params = dict(expert.named_parameters())
        task_vectors.append({
            name: e_params[name].detach().float().cpu() - base_p
            for name, base_p in base_params.items()
        })
    return base_params, task_vectors


def _merge_legacy_baseline(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    device: str,
) -> MergeResult:
    """Run legacy DARE/TIES baselines without mutating caller modules."""
    from daph_exfusion.baselines import op_dare, op_dare_ties, op_ties

    validate_parameter_names(experts, base_model)
    n_experts = len(experts)
    if n_experts == 0:
        raise ValueError("legacy baseline requires at least one expert")
    lambdas = list(config.lambdas) if config.lambdas else [1.0 / n_experts] * n_experts
    if len(lambdas) != n_experts:
        raise ValueError(f"lambdas length {len(lambdas)} != n_experts {n_experts}")

    scale = float(config.task_scale)
    method = config.method
    base_params, task_vectors = _cpu_task_vectors_no_mutation(base_model, experts)
    merged = copy.deepcopy(base_model).cpu()

    legacy = config.legacy_sparse
    dare_drop = float(legacy.get("dare_drop_rate", 0.2))
    ties_trim = float(legacy.get("ties_trim_fraction", 0.2))
    configured_sign_mode = str(legacy.get("ties_sign_mode", "magnitude"))
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    if method == MergeMethod.DARE:
        operators = ["DARE"]
        dare_used, ties_used = True, False
    elif method == MergeMethod.TIES_MAGNITUDE:
        operators = ["TIES_MAGNITUDE"]
        dare_used, ties_used = False, True
    elif method == MergeMethod.TIES_MAJORITY:
        operators = ["TIES_MAJORITY"]
        dare_used, ties_used = False, True
    else:
        operators = ["DARE", f"TIES_{configured_sign_mode.upper()}"]
        dare_used, ties_used = True, True

    with torch.no_grad():
        for name, out_param in merged.named_parameters():
            deltas = [tv[name] * lambdas[i] for i, tv in enumerate(task_vectors)]
            if method == MergeMethod.DARE:
                merged_delta = sum(
                    op_dare(d, drop_probability=dare_drop, generator=generator)
                    for d in deltas
                )
            elif method == MergeMethod.TIES_MAGNITUDE:
                merged_delta = op_ties(deltas, trim_fraction=ties_trim, sign_mode="magnitude")
            elif method == MergeMethod.TIES_MAJORITY:
                merged_delta = op_ties(deltas, trim_fraction=ties_trim, sign_mode="majority")
            else:
                merged_delta = op_dare_ties(
                    deltas,
                    drop_probability=dare_drop,
                    trim_fraction=ties_trim,
                    sign_mode=configured_sign_mode,
                    generator=generator,
                )
            out_param.copy_(base_params[name] + scale * merged_delta)

    merged.to(device)
    trace = OperatorTrace(
        method=method.value,
        operators=operators,
        dare_used=dare_used,
        ties_used=ties_used,
        task_scale=scale,
        lambdas=lambdas,
        config_hash=config.config_hash(),
    )
    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method=method.value,
    )
