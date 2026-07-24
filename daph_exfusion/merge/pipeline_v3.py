"""Canonical merge pipeline v3 — single entry point for all dense merge methods.

Every merge — CLI, tests, AGX, experiments, notebooks — goes through
``merge_experts``. This guarantees that the algorithm name in the config
matches the mathematical operations actually performed, and produces a
``MergeResult`` with an ``OperatorTrace`` for provenance.

Supported canonical dense methods (mainline):
    task_arithmetic         — θ* = θ₀ + α Σᵢ λᵢ Δᵢ
    fisher_dense            — θ*_k = Σᵢ λᵢ F_{i,k}^γ θ_{i,k} / (Σᵢ λᵢ F_{i,k}^γ + ε)
    fisher_base_anchored    — includes base Fisher F₀ with weight λ₀
    regmean                 — W* = (Σᵢ WᵢCᵢ)(ΣᵢCᵢ + ρI)⁻¹
    regmean_pp              — RegMean with propagation-aware recalibration
    coefficient_opt         — θ(λ) = θ₀ + Σᵢ λᵢ Δᵢ, optimize λ on real forward
    trust_region            — coefficient_opt with ½ Δᵀ F₀ Δ ≤ ε constraint
    kfac_barycenter         — K-FAC structured merge (G⊗A weighted)
    agx                     — architecture-aware selection of dense merge geometry

Legacy benchmark methods (controlled baselines):
    dare, ties_magnitude, ties_majority, dare_ties, emr, model_stock, slerp
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from daph_exfusion.merge.types import (
    ExpertSpec,
    MergeConfig,
    MergeMethod,
    MergeResult,
    OperatorTrace,
    extract_task_vectors,
    validate_parameter_names,
)
from daph_exfusion.merge.task_arithmetic import merge_task_arithmetic, merge_frozen


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
    """Merge experts into a single model using the canonical v3 pipeline.

    This is the single entry point for all merging. The method is selected
    by config.method, and the appropriate dense merge algorithm is dispatched.

    Args:
        base_model: The base model (θ₀).
        experts: List of specialist models.
        config: Merge configuration (method, hyperparameters).
        curvature_bank: Dict mapping expert_name → {param_name: Fisher diagonal}.
                        Required for fisher_dense, fisher_base_anchored.
        base_fisher: Dict mapping param_name → base model Fisher diagonal.
                     Required for fisher_base_anchored, trust_region.
        activation_bank: Dict mapping expert_name → {param_name: covariance}.
                         Required for regmean, regmean_pp.
        kfac_bank: Dict mapping expert_name → {param_name → (A_factor, G_factor)}.
                   Required for kfac_barycenter.
        calibration_data: Calibration data for coefficient_opt, trust_region, regmean_pp.
        evaluator: Callable(merged_model) → float for coefficient_opt, trust_region.
        forward_fn: Custom forward function for regmean_pp.
        device: Device for computation.

    Returns:
        MergeResult with the merged model and operator trace.
    """
    method = config.method

    # Dense mainline methods
    if method == MergeMethod.TASK_ARITHMETIC:
        return merge_task_arithmetic(base_model, experts, config, device=device)

    elif method == MergeMethod.FROZEN:
        return merge_frozen(base_model, config, device=device)

    elif method == MergeMethod.FISHER_DENSE:
        if curvature_bank is None:
            raise ValueError(
                "fisher_dense requires curvature_bank "
                "(empirical Fisher diagonals)."
            )
        from daph_exfusion.merge.fisher_dense import merge_fisher_dense
        return merge_fisher_dense(base_model, experts, config, curvature_bank, device=device)

    elif method == MergeMethod.FISHER_BASE_ANCHORED:
        if curvature_bank is None:
            raise ValueError(
                "fisher_base_anchored requires curvature_bank."
            )
        if base_fisher is None:
            raise ValueError(
                "fisher_base_anchored requires base_fisher."
            )
        from daph_exfusion.merge.fisher_dense import merge_fisher_base_anchored
        return merge_fisher_base_anchored(
            base_model, experts, config, curvature_bank, base_fisher, device=device
        )

    elif method == MergeMethod.REGMEAN:
        if activation_bank is None:
            raise ValueError(
                "regmean requires activation_bank (activation covariance)."
            )
        from daph_exfusion.merge.regmean import merge_regmean
        return merge_regmean(base_model, experts, config, activation_bank, device=device)

    elif method == MergeMethod.REGMEAN_PP:
        if activation_bank is None:
            raise ValueError("regmean_pp requires activation_bank.")
        if calibration_data is None:
            raise ValueError("regmean_pp requires calibration_data.")
        from daph_exfusion.merge.regmean_pp import merge_regmean_pp
        return merge_regmean_pp(
            base_model, experts, config, activation_bank,
            calibration_data, forward_fn=forward_fn, device=device,
        )

    elif method == MergeMethod.COEFFICIENT_OPT:
        if calibration_data is None:
            raise ValueError("coefficient_opt requires calibration_data.")
        if evaluator is None:
            raise ValueError("coefficient_opt requires evaluator.")
        from daph_exfusion.merge.coefficient_opt import merge_coefficient_opt
        return merge_coefficient_opt(
            base_model, experts, config, calibration_data, evaluator, device=device,
        )

    elif method == MergeMethod.TRUST_REGION:
        if base_fisher is None:
            raise ValueError("trust_region requires base_fisher.")
        if calibration_data is None:
            raise ValueError("trust_region requires calibration_data.")
        if evaluator is None:
            raise ValueError("trust_region requires evaluator.")
        from daph_exfusion.merge.trust_region import merge_trust_region
        return merge_trust_region(
            base_model, experts, config, base_fisher,
            calibration_data, evaluator, device=device,
        )

    elif method == MergeMethod.KFAC_BARYCENTER:
        if kfac_bank is None:
            raise ValueError("kfac_barycenter requires kfac_bank.")
        from daph_exfusion.merge.kfac_merge import merge_kfac
        return merge_kfac(base_model, experts, config, kfac_bank, device=device)

    elif method == MergeMethod.AGX:
        from daph_exfusion.search.agx import merge_agx
        return merge_agx(
            base_model, experts, config,
            curvature_bank=curvature_bank,
            base_fisher=base_fisher,
            activation_bank=activation_bank,
            calibration_data=calibration_data,
            evaluator=evaluator,
            device=device,
        )

    # Legacy benchmark methods
    elif method in (MergeMethod.DARE, MergeMethod.TIES_MAGNITUDE,
                    MergeMethod.TIES_MAJORITY, MergeMethod.DARE_TIES):
        return _merge_legacy_sparse(base_model, experts, config, curvature_bank, device)

    elif method == MergeMethod.EMR:
        raise NotImplementedError("EMR not yet implemented in v3 pipeline")

    elif method == MergeMethod.MODEL_STOCK:
        raise NotImplementedError("Model Stock not yet implemented in v3 pipeline")

    elif method == MergeMethod.SLERP:
        raise NotImplementedError("SLERP not yet implemented in v3 pipeline")

    else:
        raise ValueError(f"Unknown merge method: '{method}'")


def _merge_legacy_sparse(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    curvature_bank: Optional[Dict[str, Dict[str, Tensor]]],
    device: str,
) -> MergeResult:
    """Dispatch to legacy sparse merge methods (DARE, TIES, DARE-TIES).

    These are controlled baselines, not mainline. They use the legacy
    operators from daph_exfusion.merge.legacy.
    """
    from daph_exfusion.merge.legacy import op_dare, op_ties, op_dare_ties

    n_experts = len(experts)
    validate_parameter_names(experts, base_model)

    if config.lambdas:
        lambdas = list(config.lambdas)
    else:
        lambdas = [1.0 / n_experts] * n_experts

    scale = config.task_scale
    method = config.method

    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    task_vectors = extract_task_vectors(experts, base_cpu)

    merged = copy.deepcopy(base_cpu)
    merged_params = dict(merged.named_parameters())

    legacy_config = config.legacy_sparse
    dare_drop = legacy_config.get("dare_drop_rate", 0.2)
    ties_trim = legacy_config.get("ties_trim_fraction", 0.2)
    sign_mode = legacy_config.get("ties_sign_mode", "magnitude")
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    operators: List[str] = []
    dare_used = False
    ties_used = False

    with torch.no_grad():
        for name, param in merged_params.items():
            deltas = []
            for i, tv in enumerate(task_vectors):
                if name in tv:
                    deltas.append(tv[name] * lambdas[i])

            if not deltas:
                continue

            if method == MergeMethod.DARE:
                operators = ["DARE"]
                dare_used = True
                dare_deltas = [op_dare(d, drop_probability=dare_drop, generator=generator) for d in deltas]
                merged_delta = sum(dare_deltas)

            elif method in (MergeMethod.TIES_MAGNITUDE, MergeMethod.TIES_MAJORITY):
                operators = [f"TIES_{sign_mode.upper()}"]
                ties_used = True
                sm = "magnitude" if method == MergeMethod.TIES_MAGNITUDE else "majority"
                merged_delta = op_ties(deltas, trim_fraction=ties_trim, sign_mode=sm)

            elif method == MergeMethod.DARE_TIES:
                operators = ["DARE", f"TIES_{sign_mode.upper()}"]
                dare_used = True
                ties_used = True
                merged_delta = op_dare_ties(
                    deltas, drop_probability=dare_drop,
                    trim_fraction=ties_trim, sign_mode=sign_mode,
                    generator=generator,
                )
            else:
                raise ValueError(f"Unknown legacy method: {method}")

            param.copy_(param.detach().float() + merged_delta * scale)

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
