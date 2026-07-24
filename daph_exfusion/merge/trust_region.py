"""Trust-Region Constrained Merge module (ExFusion v3).

Implements trust-region constrained coefficient optimization:

    min_λ  L_specialist(θ(λ))
    subject to: L_general(θ(λ)) - L_general(θ₀) ≤ ε

Using a curvature proxy for the general-model degradation:

    ΔL₀ ≈ ½ Δᵀ F₀ Δ

So the constraint becomes:

    ½ Δ(λ)ᵀ F₀ Δ(λ) ≤ ε

For task vectors Δᵢ, the Fisher interaction matrix is:

    G_ij = Δᵢᵀ F₀ Δⱼ

Then:

    Δ(λ)ᵀ F₀ Δ(λ) = λᵀ G λ

This collapses a billions-dimensional curvature constraint into an N×N
matrix (for 3 experts, G ∈ R^{3×3}), making the trust-region projection
a trivial quadratic program solved in closed form.

The merged model is:

    θ(λ) = θ₀ + α Σᵢ λᵢ Δᵢ

Coefficients λ are optimized via projected gradient descent with an
increasing penalty on constraint violation, followed by a final radial
projection onto the ellipsoidal trust region {λ : ½ λᵀ G λ ≤ ε}.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from daph_exfusion.merge.types import (
    CoefficientParameterization,
    MergeConfig,
    MergeMethod,
    MergeResult,
    OperatorTrace,
    extract_task_vectors,
    validate_parameter_names,
)


# =============================================================================
# Fisher interaction matrix
# =============================================================================


def compute_fisher_interaction_matrix(
    task_vectors: List[Dict[str, Tensor]],
    base_fisher: Dict[str, Tensor],
) -> Tensor:
    """Compute the Fisher interaction matrix G.

    G_ij = Δᵢᵀ F₀ Δⱼ = Σ_k Σ_elements (Δᵢ_k ⊙ F₀_k ⊙ Δⱼ_k)

    where the sum ranges over all parameter names k and all elements
    within each parameter tensor. F₀ is the diagonal Fisher of the
    base model, so the quadratic form reduces to an elementwise product
    followed by a sum — no matrix inversion is required.

    For N experts, G ∈ R^{N×N} is symmetric positive-semidefinite.

    Args:
        task_vectors: List of N task-vector dicts, each mapping
            param_name -> Δᵢ tensor.
        base_fisher: Dict mapping param_name -> F₀ diagonal tensor.

    Returns:
        G: Tensor of shape (N, N) where G[i, j] = Δᵢᵀ F₀ Δⱼ.
    """
    n = len(task_vectors)
    G = torch.zeros(n, n, dtype=torch.float32)

    param_names = set()
    for tv in task_vectors:
        param_names.update(tv.keys())

    for name in param_names:
        fisher_diag = base_fisher.get(name)
        if fisher_diag is None:
            continue
        fisher_diag = fisher_diag.float()
        for i in range(n):
            if name not in task_vectors[i]:
                continue
            di = task_vectors[i][name].float()
            fi_di = fisher_diag * di
            for j in range(i, n):
                if name not in task_vectors[j]:
                    continue
                dj = task_vectors[j][name].float()
                contrib = (fi_di * dj).sum().item()
                G[i, j] += contrib
                if i != j:
                    G[j, i] += contrib

    return G


def compute_curvature_cosine(
    task_vectors: List[Dict[str, Tensor]],
    base_fisher: Dict[str, Tensor],
) -> Tensor:
    """Compute the curvature-normalized Fisher cosine matrix C^F.

    C^F_ij = Δᵢᵀ F₀ Δⱼ / (||Δᵢ||_F₀ · ||Δⱼ||_F₀)

    where the Fisher-weighted norm is:

        ||Δᵢ||_F₀ = sqrt(Δᵢᵀ F₀ Δᵢ) = sqrt(G_ii)

    This is the cosine similarity in the Fisher-induced metric. Values
    near +1 indicate that two task vectors point in the same curvature
    direction (synergistic), values near -1 indicate conflict, and
    values near 0 indicate orthogonality.

    The diagonal is exactly 1.0 by construction.

    Args:
        task_vectors: List of N task-vector dicts.
        base_fisher: Dict mapping param_name -> F₀ diagonal tensor.

    Returns:
        C^F: Tensor of shape (N, N) with values in [-1, 1].
    """
    G = compute_fisher_interaction_matrix(task_vectors, base_fisher)
    diag = torch.diagonal(G).clamp(min=0.0)
    norms = torch.sqrt(diag)
    norms = norms.clamp(min=1e-12)
    outer = norms.unsqueeze(0) * norms.unsqueeze(1)
    C = G / outer
    C = C.clamp(min=-1.0, max=1.0)
    return C


# =============================================================================
# Coefficient parameterization
# =============================================================================


def _parameterize(
    raw: Tensor,
    mode: CoefficientParameterization,
) -> Tensor:
    """Map raw unconstrained parameters to merge coefficients λ.

    SOFTMAX:        λ_i = exp(a_i) / Σ exp(a_j)  — convex (Σλ = 1)
    SIGMOID:        λ_i = σ(a_i)                 — independent, allows Σ > 1
    SIGNED:         λ_i = tanh(a_i)              — allows extrapolation
    UNCONSTRAINED:  λ_i = a_i                    — no constraints

    Args:
        raw: Tensor of shape (N,) — unconstrained parameters a.
        mode: Parameterization mode.

    Returns:
        λ: Tensor of shape (N,) — merge coefficients.
    """
    if mode == CoefficientParameterization.SOFTMAX:
        return F.softmax(raw, dim=-1)
    elif mode == CoefficientParameterization.SIGMOID:
        return torch.sigmoid(raw)
    elif mode == CoefficientParameterization.SIGNED:
        return torch.tanh(raw)
    elif mode == CoefficientParameterization.UNCONSTRAINED:
        return raw
    else:
        raise ValueError(f"Unknown coefficient parameterization: {mode}")


def _init_raw_params(
    n_experts: int,
    initial_lambdas: List[float],
    mode: CoefficientParameterization,
    device: Any,
) -> Tensor:
    """Initialize raw parameters to produce the desired starting λ.

    Args:
        n_experts: Number of experts.
        initial_lambdas: Desired initial coefficient values.
        mode: Parameterization mode.
        device: Device for the tensor.

    Returns:
        raw: Tensor of shape (N,) with requires_grad=True.
    """
    lambdas_t = torch.tensor(initial_lambdas, dtype=torch.float32, device=device)

    if mode == CoefficientParameterization.SOFTMAX:
        raw = torch.log(lambdas_t.clamp(min=1e-8) + 1e-8)
    elif mode == CoefficientParameterization.SIGMOID:
        clamped = lambdas_t.clamp(min=1e-6, max=1.0 - 1e-6)
        raw = torch.log(clamped / (1.0 - clamped))
    elif mode == CoefficientParameterization.SIGNED:
        clamped = lambdas_t.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        raw = 0.5 * torch.log((1.0 + clamped) / (1.0 - clamped))
    else:
        raw = lambdas_t.clone()

    raw = raw.detach().requires_grad_(True)
    return raw


# =============================================================================
# Trust-region projection
# =============================================================================


def _trust_region_violation(
    lambdas: Tensor,
    G: Tensor,
    budget: float,
) -> float:
    """Compute the trust-region constraint violation.

    Constraint: ½ λᵀ G λ ≤ ε

    Returns max(0, ½ λᵀ G λ - ε), which is 0 when feasible.

    Args:
        lambdas: Coefficient vector λ of shape (N,).
        G: Fisher interaction matrix of shape (N, N).
        budget: Trust-region budget ε.

    Returns:
        Violation amount (0.0 if feasible).
    """
    quad = 0.5 * (lambdas @ G @ lambdas).item()
    return max(0.0, quad - budget)


def _project_to_trust_region(
    lambdas: Tensor,
    G: Tensor,
    budget: float,
) -> Tensor:
    """Radially project λ onto the ellipsoidal trust region.

    If ½ λᵀ G λ > ε, scale λ by:

        s = sqrt(ε / (½ λᵀ G λ))

    so that the projected λ lies exactly on the constraint boundary.
    This is the exact projection for a spherical constraint and a
    tight heuristic for the general ellipsoidal case.

    Args:
        lambdas: Coefficient vector λ of shape (N,).
        G: Fisher interaction matrix of shape (N, N).
        budget: Trust-region budget ε.

    Returns:
        Projected λ (unchanged if already feasible).
    """
    quad = 0.5 * (lambdas @ G @ lambdas).item()
    if quad <= budget or quad <= 0.0:
        return lambdas
    scale = math.sqrt(budget / quad)
    return lambdas * scale


# =============================================================================
# Specialist loss via functional_call
# =============================================================================


def _build_merged_params(
    base_params: Dict[str, Tensor],
    task_vectors: List[Dict[str, Tensor]],
    lambdas: Tensor,
    scale: float,
) -> Dict[str, Tensor]:
    """Build merged parameter dict as a differentiable function of λ.

    θ(λ)_k = θ₀_k + α Σᵢ λᵢ Δᵢ_k

    Args:
        base_params: Dict of base model parameters (detached, FP32).
        task_vectors: List of N task-vector dicts.
        lambdas: Coefficient vector λ of shape (N,).
        scale: Global scale α.

    Returns:
        Dict mapping param_name -> merged parameter tensor.
    """
    merged: Dict[str, Tensor] = {}
    for name, base_p in base_params.items():
        delta = torch.zeros_like(base_p)
        for i, tv in enumerate(task_vectors):
            if name in tv:
                delta = delta + lambdas[i] * tv[name]
        merged[name] = base_p + scale * delta
    return merged


def _compute_specialist_loss(
    model: nn.Module,
    params: Dict[str, Tensor],
    calibration_data: Any,
    device: Any,
    max_batches: int = 4,
) -> Tensor:
    """Compute the average specialist loss via functional_call.

    Uses torch.func.functional_call so that the loss is differentiable
    with respect to the entries of ``params``, which are themselves
    differentiable functions of λ.

    Args:
        model: Model module (provides the forward graph).
        params: Parameter dict to substitute during the forward pass.
        calibration_data: Iterable of batches. Each batch is a dict
            with 'input_ids', 'attention_mask', and 'labels'.
        device: Device for computation.
        max_batches: Maximum number of batches to evaluate.

    Returns:
        Scalar loss tensor (differentiable w.r.t. params).
    """
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    count = 0

    for batch in calibration_data:
        if count >= max_batches:
            break

        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch.get("labels", input_ids.clone()).to(device)
        else:
            input_ids = batch.to(device)
            attention_mask = None
            labels = input_ids.clone()

        kwargs: Dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        try:
            out = torch.func.functional_call(
                model, params, (input_ids,), kwargs=kwargs,
            )
        except TypeError:
            out = torch.func.functional_call(
                model, params, (input_ids,), kwargs,
            )

        if hasattr(out, "loss") and out.loss is not None:
            loss = out.loss
        else:
            logits = out.logits if hasattr(out, "logits") else out
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        total_loss = total_loss + loss
        count += 1

    if count == 0:
        return total_loss

    return total_loss / count


# =============================================================================
# Trust-region merge entry point
# =============================================================================


def merge_trust_region(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    base_fisher: Dict[str, Tensor],
    calibration_data: Any,
    evaluator: Any,
    device: Any = "cpu",
) -> MergeResult:
    """Execute Trust-Region Constrained merge.

    Solves:

        min_λ  L_specialist(θ(λ))
        s.t.   ½ λᵀ G λ ≤ ε

    where θ(λ) = θ₀ + α Σᵢ λᵢ Δᵢ and G is the N×N Fisher interaction
    matrix with G_ij = Δᵢᵀ F₀ Δⱼ.

    Optimization uses projected gradient descent with an increasing
    quadratic penalty on constraint violation, followed by a final
    radial projection onto the trust region.

    Args:
        base_model: Base model θ₀.
        experts: List of N specialist models.
        config: Merge configuration (method must be trust_region).
        base_fisher: Dict mapping param_name -> base model Fisher
            diagonal F₀ (computed via exact per-sample empirical Fisher).
        calibration_data: Iterable of batches for specialist loss
            computation. Each batch is a dict with 'input_ids',
            'attention_mask', and 'labels'.
        evaluator: Callable(merged_model) -> float for final evaluation
            (lower is better, e.g. NLL).
        device: Device for computation.

    Returns:
        MergeResult with the merged model and operator trace.
    """
    if config.method != MergeMethod.TRUST_REGION:
        raise ValueError(
            f"merge_trust_region called with method={config.method}, "
            f"expected trust_region"
        )

    n_experts = len(experts)
    validate_parameter_names(experts, base_model)

    if config.lambdas:
        initial_lambdas = list(config.lambdas)
        if len(initial_lambdas) != n_experts:
            raise ValueError(
                f"lambdas length {len(initial_lambdas)} != n_experts {n_experts}"
            )
    else:
        initial_lambdas = [1.0 / n_experts] * n_experts

    scale = config.task_scale
    budget = config.trust_region_budget
    lr = config.coefficient_lr
    n_steps = config.coefficient_steps
    param_mode = config.coefficient_parameterization

    torch.manual_seed(config.seed)

    base_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    task_vectors = extract_task_vectors(experts, base_cpu)
    base_params = {name: p.detach().float() for name, p in base_cpu.named_parameters()}

    G = compute_fisher_interaction_matrix(task_vectors, base_fisher)
    G = G.to(device)

    raw = _init_raw_params(n_experts, initial_lambdas, param_mode, device)
    optimizer = torch.optim.Adam([raw], lr=lr)

    penalty = 1.0
    penalty_growth = 1.5
    max_penalty = 1e6

    model_on_device = copy.deepcopy(base_cpu).to(device)
    model_on_device.eval()

    for step in range(n_steps):
        optimizer.zero_grad()

        lambdas = _parameterize(raw, param_mode)
        lambdas = lambdas.to(device)

        merged_params = _build_merged_params(
            base_params, task_vectors, lambdas, scale,
        )
        merged_params = {k: v.to(device) for k, v in merged_params.items()}

        loss = _compute_specialist_loss(
            model_on_device, merged_params, calibration_data, device,
        )

        quad = 0.5 * (lambdas @ G @ lambdas)
        violation = F.relu(quad - budget)
        aug_loss = loss + penalty * violation

        aug_loss.backward()
        optimizer.step()

        penalty = min(penalty * penalty_growth, max_penalty)

    with torch.no_grad():
        final_lambdas = _parameterize(raw, param_mode).detach().cpu()
        final_lambdas = _project_to_trust_region(final_lambdas, G.cpu(), budget)
        final_lambdas_list = final_lambdas.tolist()

    merged = copy.deepcopy(base_cpu)
    merged_params_dict = dict(merged.named_parameters())

    with torch.no_grad():
        for name, param in merged_params_dict.items():
            delta = torch.zeros_like(param.detach().float())
            for i, tv in enumerate(task_vectors):
                if name in tv:
                    delta = delta + final_lambdas[i] * tv[name]
            param.copy_(param.detach().float() + delta * scale)

    merged.to(device)

    trace = OperatorTrace(
        method="trust_region",
        operators=["TRUST_REGION", "FISHER_INTERACTION_MATRIX"],
        trust_region_enforced=True,
        fisher_used=True,
        fisher_estimator="exact_per_sample",
        task_scale=scale,
        fisher_gamma=config.fisher_gamma,
        lambdas=final_lambdas_list,
        config_hash=config.config_hash(),
    )

    return MergeResult(
        merged_model=merged,
        trace=trace,
        config=config,
        method="trust_region",
    )
