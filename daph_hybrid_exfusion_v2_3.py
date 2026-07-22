#!/usr/bin/env python3
"""
DAPH ExFusion Hybrid v2.3

Hybrid Mamba/Transformer decoder layer with difficulty-aware macro-routing and
architecture-aware DARE -> TIES -> Fisher model merging.

Status: corrected integration draft. The built-in deterministic CPU self-tests
exercise the repaired attention, cache, routing, and merge semantics. This is
still not production-certified.

Key corrections in v2.3:
  - Causal decoder attention with separate key-padding and causal attention masks.
  - History padding is applied through key_padding_mask, not attn_mask.
  - Stateful Mamba output and next recurrent state come from one shared call.
  - Cached Mamba state now affects routed streaming output.
  - routing_mode implements hard, soft, and nucleus/top-p routing.
  - Batch hard routing uses full-batch masked Mamba recurrence to preserve state.
  - TIES sign-agreement merge normalizes by agreeing expert weight, not count.
  - Fisher denominator excludes DARE-dropped elements via explicit keep masks.
  - SSM Fisher boost changes per-parameter TIES/Fisher blend and no longer cancels.
  - Mamba and Transformer expert families are merged independently.
  - Merge targets are expected to be copies of their corresponding base modules.
  - SSM allowlist/blocklist policy propagates through DARE, TIES, and Fisher.
  - DARE accepts an optional torch.Generator for deterministic merges.
  - Empirical Fisher accepts structured inputs and an explicit loss_fn.
  - Recurrent selective-scan state arithmetic is retained in FP32.

v2.3 unification (ports from the v2.1 line, on the v1.9.1 base):
  - mask_convention config: "hf" (default; 1/True = valid) or "pytorch"
    (bool True = blocked), converted at the attention boundary.
  - merge_mode policy: "full" (DARE->TIES->Fisher) or "weighted_average"
    (plain weighted average of task vectors, no trim/elect/boost).
  - Canonical Switch load-balancing loss: f_i is the hard argmax fraction
    (detached) times soft mean probability P_i (was N * sum(mean_p^2)).
  - Sparse gather/scatter dispatch for pointwise paths (Trans-ExFusion,
    Cheap) in token-level hard routing, preserving per-token difficulty.

v2.3 hardening (tenth external review):
  - Attention-sink preservation: attn_sink_tokens config anchors the first N
    tokens in the sliding-window cache trim (StreamingLLM, arXiv:2309.17453).
  - Opt-in bypass decay (ssm_bypass_decay, default 0.0): gamma-decays the
    SSM state on bypassed steps instead of exact preservation. Default 0.0
    keeps the bit-exact h_t = h_{t-1} guarantee.
  - Micro-batched empirical Fisher (fisher_micro_batch, default 1). NOTE:
    micro-batching squares the MEAN gradient of each micro-batch, an
    approximation of E[g^2] (bounded cancellation); default 1 stays rigorous.
  - Mask normalization consolidated at the forward entry point (single
    conversion; downstream consumers reuse the standardized valid mask).
  - Pluggable scan dispatch interface (dispatch_selective_scan): extension
    point for native kernels (e.g. mamba_ssm/Triton) with graceful fallback
    to the compiled/eager FP32 scan.

License: MIT
"""

from __future__ import annotations

import copy
import math
import os
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 1. CORE POLICY & HELPERS
# =============================================================================

DEFAULT_MAMBA_POLICIES: Dict[str, Any] = {
    "ssm_groups": ["A_log", "D", "dt_proj"],
    "proj_groups": [
        "in_proj", "conv1d", "out_proj", "q_proj", "k_proj",
        "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj",
        "fc1", "fc2",
    ],
    "ssm_drop_reduction": 0.5,
    "ssm_soft_merge": True,
    "ssm_fisher_boost": 2.5,
    "fisher_power": 1.0,
    "fisher_floor": 1e-8,
    "dare_base_p": 0.25,
    "ties_trim_ratio": 0.2,
    "sign_mode": "majority",
    "ties_fisher_blend": 0.5,
    "merge_mode": "full",   # "full" | "weighted_average"
}


StructuredBatch = Union[Tensor, Sequence[Any], Mapping[str, Any]]
LossFn = Callable[[Any, Any], Tensor]
ForwardFn = Callable[[nn.Module, Any], Any]


def _validate_probability(name: str, value: float, *, inclusive_one: bool = True) -> None:
    upper_ok = value <= 1.0 if inclusive_one else value < 1.0
    if not math.isfinite(value) or value < 0.0 or not upper_ok:
        bound = "[0, 1]" if inclusive_one else "[0, 1)"
        raise ValueError(f"{name} must be finite and in {bound}; got {value}")


def _name_tokens(name: str) -> List[str]:
    return [t for t in name.lower().replace(".", "_").split("_") if t]


def is_ssm_core_param(
    param_name: str,
    policies: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return whether a parameter should receive SSM-core protection.

    Allowlist semantics are additive: an allowlist match marks the parameter as
    SSM-core. A blocklist match always excludes it.
    """
    policies = policies or {}
    name_lower = param_name.lower()

    block = policies.get("ssm_core_blocklist")
    if block is not None and any(str(token).lower() in name_lower for token in block):
        return False

    allow = policies.get("ssm_core_allowlist")
    if allow is not None and any(str(token).lower() in name_lower for token in allow):
        return True

    configured = [str(token).lower() for token in policies.get("ssm_groups", [])]
    if any(token in name_lower for token in configured):
        return True

    tokens = _name_tokens(param_name)
    joined = "_".join(tokens)
    if "a_log" in joined or "dt_proj" in joined:
        return True
    return bool(tokens) and tokens[-1] == "d"


def is_projection_param(
    param_name: str,
    policies: Optional[Mapping[str, Any]] = None,
) -> bool:
    policies = policies or {}
    proj_keywords = [
        str(token).lower()
        for token in policies.get("proj_groups", DEFAULT_MAMBA_POLICIES["proj_groups"])
    ]
    name_lower = param_name.lower()
    return any(keyword in name_lower for keyword in proj_keywords)


def compute_difficulty_metrics(logits: Tensor) -> Dict[str, Tensor]:
    if logits.shape[-1] < 1:
        raise ValueError("logits must have a non-empty final dimension")
    probs = F.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1)
    max_prob = probs.max(dim=-1).values
    variance = torch.var(probs, dim=-1, unbiased=False)
    norm_denom = max(math.log(max(2, logits.shape[-1])), math.log(2.0))
    return {
        "entropy": entropy,
        "max_prob": max_prob,
        "variance": variance,
        "difficulty_score": (entropy / norm_denom) * (1.0 - max_prob),
    }


def compute_difficulty_from_hidden_states(hidden_states: Tensor) -> Dict[str, Tensor]:
    """Derive token-local, batch-independent fallback difficulty metrics.

    The previous batch/sequence similarity fallback changed when batch members or
    chunk boundaries changed. That made routing non-deterministic between full
    sequence and streaming execution. Standardizing each token independently and
    interpreting its hidden channels as a feature distribution preserves the
    fallback's entropy signal without cross-sample or future-token dependence.
    """
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(1)
    if hidden_states.dim() != 3:
        raise ValueError(
            f"hidden_states must have shape (B, L, H) or (B, H); got {tuple(hidden_states.shape)}"
        )
    hidden = hidden_states.float()
    centered = hidden - hidden.mean(dim=-1, keepdim=True)
    scaled = centered / hidden.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return compute_difficulty_metrics(scaled)


def _align_difficulty(ds: Tensor, batch_size: int, seq_len: int) -> Tensor:
    ds = ds.float()
    if ds.dim() == 0:
        return ds.view(1, 1, 1).expand(batch_size, seq_len, 1)
    if ds.dim() == 1:
        if ds.shape[0] == seq_len and batch_size == 1:
            return ds.view(1, seq_len, 1)
        if ds.shape[0] == batch_size:
            return ds.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
        return ds.mean().view(1, 1, 1).expand(batch_size, seq_len, 1)
    if ds.dim() == 2 and ds.shape == (batch_size, seq_len):
        return ds.unsqueeze(-1)
    if ds.dim() == 3 and ds.shape[:2] == (batch_size, seq_len):
        return ds.mean(dim=-1, keepdim=True)
    return ds.mean().view(1, 1, 1).expand(batch_size, seq_len, 1)


def _gather_difficulty(
    difficulty_metrics: Optional[Dict[str, Tensor]],
    b_idx: Tensor,
    t_idx: Tensor,
    batch_size: int,
    seq_len: int,
) -> Dict[str, Tensor]:
    """Gather per-token difficulty values for sparse pointwise dispatch.

    Returns tensors shaped [N, 1] so downstream _align_difficulty keeps each
    selected token's OWN value instead of collapsing to a global mean.
    """
    if difficulty_metrics is None:
        return {}
    gathered: Dict[str, Tensor] = {}
    for key, value in difficulty_metrics.items():
        if not isinstance(value, Tensor):
            gathered[key] = value
        elif value.dim() == 2 and value.shape == (batch_size, seq_len):
            gathered[key] = value[b_idx, t_idx].unsqueeze(1)
        elif value.dim() == 1 and value.shape[0] == batch_size:
            gathered[key] = value[b_idx].unsqueeze(1)
        elif value.dim() == 1 and value.shape[0] == seq_len and batch_size == 1:
            gathered[key] = value[t_idx].unsqueeze(1)
        else:
            gathered[key] = value
    return gathered


def _normalize_expert_weights(
    memory_bank_weights: Tensor,
    num_experts: int,
    difficulty_importance: Optional[Tensor] = None,
) -> Tensor:
    raw = memory_bank_weights.float().reshape(-1)
    if raw.numel() != num_experts:
        raise ValueError(
            f"memory_bank_weights length {raw.numel()} != expert count {num_experts}"
        )
    if not torch.isfinite(raw).all():
        raise ValueError("memory_bank_weights must be finite")

    weights = F.softmax(raw, dim=0)
    if difficulty_importance is not None:
        difficulty = difficulty_importance.float().reshape(-1)
        if difficulty.numel() != num_experts:
            raise ValueError(
                f"difficulty_importance length {difficulty.numel()} != expert count {num_experts}"
            )
        if not torch.isfinite(difficulty).all() or (difficulty < 0).any():
            raise ValueError("difficulty_importance must be finite and non-negative")
        weights = weights * difficulty
        total = weights.sum()
        if total <= 0:
            raise ValueError("difficulty_importance zeroed every expert weight")
        weights = weights / total
    return weights


def _validate_homogeneous_task_vectors(task_vectors: List[Dict[str, Tensor]]) -> List[str]:
    if not task_vectors:
        raise ValueError("No task vectors provided")
    reference_keys = list(task_vectors[0].keys())
    if not reference_keys:
        raise ValueError("Task vectors are empty; expert/base parameter names do not match")
    reference_set = set(reference_keys)
    for index, task_vector in enumerate(task_vectors[1:], start=1):
        if set(task_vector) != reference_set:
            missing = sorted(reference_set - set(task_vector))
            extra = sorted(set(task_vector) - reference_set)
            raise ValueError(
                f"Expert {index} task-vector keys differ from expert 0; "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
        for name in reference_keys:
            if task_vector[name].shape != task_vectors[0][name].shape:
                raise ValueError(
                    f"Task-vector shape mismatch for '{name}': "
                    f"{tuple(task_vector[name].shape)} != {tuple(task_vectors[0][name].shape)}"
                )
    return reference_keys


# =============================================================================
# 2. MERGE STAGES
# =============================================================================


def extract_task_vectors(
    experts: Sequence[nn.Module],
    base: nn.Module,
) -> List[Dict[str, Tensor]]:
    if not experts:
        raise ValueError("At least one expert is required")
    base_sd = {name: value.detach().clone() for name, value in base.state_dict().items()}
    task_vectors: List[Dict[str, Tensor]] = []

    for expert_index, expert in enumerate(experts):
        expert_sd = expert.state_dict()
        task_vector: Dict[str, Tensor] = {}
        for name, base_value in base_sd.items():
            expert_value = expert_sd.get(name)
            if expert_value is None:
                continue
            if expert_value.shape != base_value.shape:
                continue
            if expert_value.is_floating_point() and base_value.is_floating_point():
                task_vector[name] = (expert_value.detach() - base_value).clone()
        if not task_vector:
            raise ValueError(
                f"Expert {expert_index} has no compatible floating-point parameters with its base module"
            )
        task_vectors.append(task_vector)

    _validate_homogeneous_task_vectors(task_vectors)
    return task_vectors


def apply_dare_preprocessing(
    task_vectors: List[Dict[str, Tensor]],
    difficulty_importance: Optional[Tensor] = None,
    dare_base_p: float = 0.25,
    ssm_drop_reduction: float = 0.5,
    policies: Optional[Mapping[str, Any]] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[Dict[str, Tensor]], List[Dict[str, Tensor]]]:
    _validate_probability("dare_base_p", dare_base_p, inclusive_one=False)
    _validate_probability("ssm_drop_reduction", ssm_drop_reduction)
    num_experts = len(task_vectors)
    _validate_homogeneous_task_vectors(task_vectors)

    if difficulty_importance is not None:
        difficulty = difficulty_importance.float().reshape(-1)
        if difficulty.numel() != num_experts:
            raise ValueError(
                f"difficulty_importance length {difficulty.numel()} != expert count {num_experts}"
            )
        if not torch.isfinite(difficulty).all() or (difficulty < 0).any():
            raise ValueError("difficulty_importance must be finite and non-negative")
    else:
        difficulty = None

    processed: List[Dict[str, Tensor]] = []
    keep_masks: List[Dict[str, Tensor]] = []

    for expert_index, task_vector in enumerate(task_vectors):
        p_effective = dare_base_p
        if difficulty is not None:
            # Bounded modulation: higher importance receives less dropping.
            importance = float(difficulty[expert_index].clamp(0.0, 1.0).item())
            p_effective = dare_base_p * (1.0 - 0.3 * importance)

        dropped: Dict[str, Tensor] = {}
        masks: Dict[str, Tensor] = {}
        for name, delta in task_vector.items():
            drop_rate = p_effective
            if is_ssm_core_param(name, policies):
                drop_rate *= ssm_drop_reduction
            drop_rate = min(0.95, max(0.0, float(drop_rate)))

            random_values = torch.rand(
                delta.shape,
                device=delta.device,
                dtype=torch.float32,
                generator=generator,
            )
            keep = random_values >= drop_rate
            scale = 1.0 / max(1.0 - drop_rate, 1e-8)
            dropped[name] = delta * keep.to(delta.dtype) * scale
            masks[name] = keep

        processed.append(dropped)
        keep_masks.append(masks)

    return processed, keep_masks


def difficulty_weighted_ties_merge(
    task_vectors: List[Dict[str, Tensor]],
    memory_bank_weights: Tensor,
    difficulty_importance: Optional[Tensor] = None,
    trim_ratio: float = 0.2,
    ssm_soft_merge: bool = True,
    sign_mode: str = "majority",
    policies: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Tensor]:
    _validate_probability("trim_ratio", trim_ratio, inclusive_one=False)
    if sign_mode != "majority":
        raise ValueError(f"Unsupported sign_mode '{sign_mode}'; only 'majority' is implemented")

    names = _validate_homogeneous_task_vectors(task_vectors)
    num_experts = len(task_vectors)
    weights = _normalize_expert_weights(
        memory_bank_weights,
        num_experts,
        difficulty_importance,
    )

    merged: Dict[str, Tensor] = {}
    for name in names:
        deltas = torch.stack([task_vector[name] for task_vector in task_vectors], dim=0)
        flat = deltas.flatten(1)
        num_elements = flat.shape[1]
        keep_count = max(1, math.ceil((1.0 - trim_ratio) * num_elements))
        kth_index = num_elements - keep_count + 1
        thresholds = flat.abs().kthvalue(kth_index, dim=1).values
        keep = flat.abs() >= thresholds.unsqueeze(1)
        trimmed = (flat * keep.to(flat.dtype)).view_as(deltas)

        if is_ssm_core_param(name, policies) and ssm_soft_merge:
            merged[name] = torch.sum(
                weights.view(num_experts, *([1] * (trimmed.dim() - 1))) * trimmed,
                dim=0,
            )
            continue

        # Pure sign-majority voting (TIES v2): decouple the consensus from
        # raw delta magnitudes so large outliers can't swamp the election.
        # (sign(delta) * delta.abs() == delta, which collapses the vote
        # into a weighted sum of deltas — the magnitude-weighted bug.)
        vote = torch.sum(
            weights.view(num_experts, *([1] * (trimmed.dim() - 1))) * torch.sign(trimmed),
            dim=0,
        )
        elected_sign = torch.sign(vote)
        agrees = (torch.sign(trimmed) == elected_sign.unsqueeze(0)) & (trimmed != 0)

        weighted_numerator = torch.zeros_like(trimmed[0])
        weighted_denominator = torch.zeros_like(trimmed[0])
        for expert_index in range(num_experts):
            agree_i = agrees[expert_index].to(trimmed.dtype)
            weighted_numerator += weights[expert_index] * trimmed[expert_index] * agree_i
            weighted_denominator += weights[expert_index] * agree_i

        merged[name] = torch.where(
            weighted_denominator > 0,
            weighted_numerator / weighted_denominator.clamp_min(1e-12),
            torch.zeros_like(weighted_numerator),
        )

    return merged


def _move_batch_to_device(batch: Any, device: Union[str, torch.device]) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: _move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]
    return batch


def _batch_size(batch: Any) -> int:
    if isinstance(batch, Tensor):
        return int(batch.shape[0])
    if isinstance(batch, Mapping):
        for value in batch.values():
            try:
                return _batch_size(value)
            except (TypeError, ValueError):
                continue
    if isinstance(batch, (tuple, list)):
        for value in batch:
            try:
                return _batch_size(value)
            except (TypeError, ValueError):
                continue
    raise ValueError("Unable to infer calibration batch size")


def _slice_batch(batch: Any, index: int) -> Any:
    if isinstance(batch, Tensor):
        return batch[index:index + 1]
    if isinstance(batch, Mapping):
        return {key: _slice_batch(value, index) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_slice_batch(value, index) for value in batch)
    if isinstance(batch, list):
        return [_slice_batch(value, index) for value in batch]
    return batch


def _default_forward(model: nn.Module, sample: Any) -> Any:
    if isinstance(sample, Mapping):
        return model(**sample)
    if isinstance(sample, tuple):
        return model(*sample)
    if isinstance(sample, list):
        return model(*sample)
    return model(sample)


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("Unable to extract a Tensor from model output")


def _default_fisher_loss(output: Any, _: Any) -> Tensor:
    logits = _extract_logits(output)
    if logits.numel() == 0:
        raise ValueError("Model output is empty")
    # Diagnostic fallback only. Real integrations should provide a task-specific loss_fn.
    if logits.dim() >= 2 and logits.shape[-1] >= 2:
        flat_logits = logits.float().reshape(-1, logits.shape[-1])
        with torch.no_grad():
            pseudo_targets = flat_logits.detach().argmax(dim=-1)
        return F.cross_entropy(flat_logits, pseudo_targets)
    return logits.float().square().mean()


def build_empirical_fisher_diagonals(
    model: nn.Module,
    calibration_batch: StructuredBatch,
    forward_fn: Optional[ForwardFn] = None,
    loss_fn: Optional[LossFn] = None,
    device: Union[str, torch.device] = "cpu",
    micro_batch_size: int = 1,
) -> Dict[str, Tensor]:
    """Empirical diagonal Fisher via per-sample gradient accumulation.

    micro_batch_size=1 (default) is rigorous: E[g^2] with no cancellation.
    micro_batch_size>1 processes small batches for speed but squares the MEAN
    gradient of each micro-batch - an approximation with cancellation bounded
    by the micro-batch size (it is NOT exact per-sample E[g^2]).
    """
    if _batch_size(calibration_batch) <= 0:
        raise ValueError("calibration_batch must contain at least one sample")
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be >= 1")

    original_training = model.training
    first_parameter = next(model.parameters(), None)
    original_device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    fisher = {
        name: torch.zeros_like(parameter, device=device, dtype=torch.float32)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    batch = _move_batch_to_device(calibration_batch, device)
    batch_size = _batch_size(batch)
    forward = forward_fn or _default_forward
    loss_builder = loss_fn or _default_fisher_loss

    try:
        model.to(device)
        model.eval()
        if micro_batch_size == 1:
            for sample_index in range(batch_size):
                model.zero_grad(set_to_none=True)
                sample = _slice_batch(batch, sample_index)
                output = forward(model, sample)
                loss = loss_builder(output, sample)
                if loss.dim() != 0:
                    loss = loss.mean()
                loss.backward()

                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        fisher[name] += parameter.grad.detach().float().square() / batch_size
        else:
            # Approximate path: mean gradient per micro-batch, squared.
            for start in range(0, batch_size, micro_batch_size):
                model.zero_grad(set_to_none=True)
                indices = list(range(start, min(start + micro_batch_size, batch_size)))
                samples = [_slice_batch(batch, i) for i in indices]
                if isinstance(batch, Tensor):
                    sub_batch = torch.cat(samples, dim=0)
                elif isinstance(batch, Mapping):
                    sub_batch = {k: torch.cat([s[k] for s in samples], dim=0)
                                 for k in batch.keys()}
                else:
                    sub_batch = samples
                output = forward(model, sub_batch)
                loss = loss_builder(output, sub_batch)
                if loss.dim() != 0:
                    loss = loss.mean()
                loss.backward()
                weight = len(indices) / batch_size
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        fisher[name] += parameter.grad.detach().float().square() * weight
    finally:
        model.zero_grad(set_to_none=True)
        model.to(original_device)
        model.train(original_training)

    return fisher


def build_fisher_diagonals_from_tracker(
    model: nn.Module,
    tracker: Any,
) -> Dict[str, Tensor]:
    if tracker is None:
        raise ValueError("KFAC tracker is None; use build_empirical_fisher_diagonals")
    fisher: Dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        diagonal = tracker.get_diagonal(name)
        if diagonal is None:
            raise ValueError(f"Tracker returned no diagonal for '{name}'")
        if diagonal.shape != parameter.shape:
            raise ValueError(
                f"Tracker diagonal shape for '{name}' is {tuple(diagonal.shape)}, "
                f"expected {tuple(parameter.shape)}"
            )
        fisher[name] = diagonal
    return fisher


def difficulty_weighted_fisher_merge(
    task_vectors: List[Dict[str, Tensor]],
    fisher_diagonals: List[Dict[str, Tensor]],
    memory_bank_weights: Tensor,
    difficulty_importance: Optional[Tensor] = None,
    dare_keep_masks: Optional[List[Dict[str, Tensor]]] = None,
    fisher_power: float = 1.0,
    fisher_floor: float = 1e-8,
) -> Dict[str, Tensor]:
    if not math.isfinite(fisher_power) or fisher_power <= 0:
        raise ValueError(f"fisher_power must be finite and > 0; got {fisher_power}")
    if not math.isfinite(fisher_floor) or fisher_floor <= 0:
        raise ValueError(f"fisher_floor must be finite and > 0; got {fisher_floor}")

    names = _validate_homogeneous_task_vectors(task_vectors)
    num_experts = len(task_vectors)
    if len(fisher_diagonals) != num_experts:
        raise ValueError(
            f"fisher_diagonals count {len(fisher_diagonals)} != expert count {num_experts}"
        )
    if dare_keep_masks is not None and len(dare_keep_masks) != num_experts:
        raise ValueError(
            f"dare_keep_masks count {len(dare_keep_masks)} != expert count {num_experts}"
        )

    weights = _normalize_expert_weights(
        memory_bank_weights,
        num_experts,
        difficulty_importance,
    )

    merged: Dict[str, Tensor] = {}
    for name in names:
        numerator = torch.zeros_like(task_vectors[0][name], dtype=torch.float32)
        denominator = torch.zeros_like(task_vectors[0][name], dtype=torch.float32)

        for expert_index, task_vector in enumerate(task_vectors):
            delta = task_vector[name].float()
            fisher = fisher_diagonals[expert_index].get(name)
            if fisher is None:
                fisher = torch.full_like(delta, fisher_floor)
            else:
                if fisher.shape != delta.shape:
                    raise ValueError(
                        f"Fisher shape mismatch for expert {expert_index}, '{name}': "
                        f"{tuple(fisher.shape)} != {tuple(delta.shape)}"
                    )
                fisher = fisher.to(delta.device, dtype=torch.float32)

            importance = fisher.clamp_min(fisher_floor).pow(fisher_power)
            if dare_keep_masks is None:
                active = delta.ne(0)
            else:
                active = dare_keep_masks[expert_index][name].to(delta.device)

            weighted_importance = weights[expert_index].to(delta.device) * importance
            numerator += weighted_importance * delta
            denominator += weighted_importance * active.to(weighted_importance.dtype)

        value = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(fisher_floor),
            torch.zeros_like(numerator),
        )
        merged[name] = value.to(task_vectors[0][name].dtype)

    return merged


def blend_ties_and_fisher(
    ties_delta: Dict[str, Tensor],
    fisher_delta: Dict[str, Tensor],
    ties_fisher_blend: float,
    ssm_fisher_boost: float,
    policies: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Tensor]:
    _validate_probability("ties_fisher_blend", ties_fisher_blend)
    if not math.isfinite(ssm_fisher_boost) or ssm_fisher_boost <= 0:
        raise ValueError("ssm_fisher_boost must be finite and > 0")

    if set(ties_delta) != set(fisher_delta):
        raise ValueError("TIES and Fisher deltas must have identical parameter keys")

    merged: Dict[str, Tensor] = {}
    base_fisher_fraction = 1.0 - ties_fisher_blend
    for name, ties_value in ties_delta.items():
        fisher_fraction = base_fisher_fraction
        if is_ssm_core_param(name, policies):
            odds = fisher_fraction / max(1.0 - fisher_fraction, 1e-12)
            boosted_odds = odds * ssm_fisher_boost
            fisher_fraction = boosted_odds / (1.0 + boosted_odds)
        ties_fraction = 1.0 - fisher_fraction
        merged[name] = ties_fraction * ties_value + fisher_fraction * fisher_delta[name]
    return merged


# =============================================================================
# 3. SELECTIVE SSM
# =============================================================================


def _selective_scan_impl(
    xin: Tensor,
    b_matrix: Tensor,
    c_matrix: Tensor,
    dt: Tensor,
    a_matrix: Tensor,
    d_skip: Tensor,
    h_init: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Minimal selective scan using FP32 recurrent state arithmetic."""
    input_dtype = xin.dtype
    xin_f = xin.float()
    b_f = b_matrix.float()
    c_f = c_matrix.float()
    dt_f = dt.float()
    a_f = a_matrix.float()
    d_f = d_skip.float()
    state = h_init.float()

    outputs: List[Tensor] = []
    for token_index in range(xin.shape[1]):
        dt_t = dt_f[:, token_index].unsqueeze(-1)
        transition = torch.exp(dt_t * a_f.unsqueeze(0))
        input_term = (
            dt_t
            * b_f[:, token_index].unsqueeze(1)
            * xin_f[:, token_index].unsqueeze(-1)
        )
        state = transition * state + input_term
        projected = (
            state * c_f[:, token_index].unsqueeze(1)
        ).sum(dim=-1)
        outputs.append(projected)

    y = torch.stack(outputs, dim=1)
    y = y + d_f * xin_f
    return y.to(input_dtype), state


_SCAN_BACKENDS: Dict[str, Callable[..., Tuple[Tensor, Tensor]]] = {}
_COMPILED_FALLBACK: Optional[Callable[..., Tuple[Tensor, Tensor]]] = None


def register_scan_backend(
    name: str,
    fn: Callable[..., Tuple[Tensor, Tensor]],
) -> None:
    """Register an optimized selective-scan backend (e.g. a mamba_ssm/Triton
    adapter). Select it at runtime with DAPH_SCAN_BACKEND=<name>. The
    callable must match _selective_scan_impl's signature."""
    _SCAN_BACKENDS[name] = fn


def dispatch_selective_scan(
    xin, b_matrix, c_matrix, dt, a_matrix, d_skip, h_init,
) -> Tuple[Tensor, Tensor]:
    """Route the scan to a registered backend (DAPH_SCAN_BACKEND) or fall
    back to the compiled/eager FP32 reference implementation."""
    global _COMPILED_FALLBACK
    backend = os.environ.get("DAPH_SCAN_BACKEND")
    if backend:
        if backend not in _SCAN_BACKENDS:
            raise ValueError(
                f"DAPH_SCAN_BACKEND='{backend}' not registered; "
                f"available: {sorted(_SCAN_BACKENDS)}")
        return _SCAN_BACKENDS[backend](
            xin, b_matrix, c_matrix, dt, a_matrix, d_skip, h_init)
    if _COMPILED_FALLBACK is None:
        _COMPILED_FALLBACK = _maybe_compiled_scan()
    return _COMPILED_FALLBACK(
        xin, b_matrix, c_matrix, dt, a_matrix, d_skip, h_init)


def _maybe_compiled_scan() -> Callable[..., Tuple[Tensor, Tensor]]:
    if os.environ.get("DAPH_USE_COMPILE", "0") != "1":
        return _selective_scan_impl
    try:
        return torch.compile(_selective_scan_impl, dynamic=True, fullgraph=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(f"torch.compile unavailable ({exc}); using eager scan")
        return _selective_scan_impl


class SelectiveSSM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or state_size <= 0:
            raise ValueError("hidden_size and state_size must be positive")
        if not (0 < dt_min <= dt_max):
            raise ValueError("Require 0 < dt_min <= dt_max")

        self.hidden_size = hidden_size
        self.state_size = state_size
        self.in_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.x_proj = nn.Linear(hidden_size, state_size * 2 + hidden_size, bias=False)
        self.dt_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_size + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(hidden_size, -1)
            .clone()
        )
        self.D = nn.Parameter(torch.ones(hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dt_min = dt_min
        self.dt_max = dt_max
        # Scan operator: pluggable dispatch (DAPH_SCAN_BACKEND registry)
        # with compiled/eager FP32 fallback. Kept as an attribute so tests
        # and integrations can wrap/override it.
        self._scan = dispatch_selective_scan

    def _validate_state(self, state: Tensor, batch_size: int) -> None:
        expected = (batch_size, self.hidden_size, self.state_size)
        if tuple(state.shape) != expected:
            raise ValueError(
                f"Invalid SSM state shape {tuple(state.shape)}; expected {expected}"
            )

    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bypass_decay: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, L, H); got {tuple(x.shape)}")
        batch_size, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Input hidden size {hidden_size} != configured {self.hidden_size}"
            )

        xz = self.in_proj(x)
        xin, gate = xz.chunk(2, dim=-1)
        bcdt = self.x_proj(xin)
        b_matrix, c_matrix, dt_raw = torch.split(
            bcdt,
            [self.state_size, self.state_size, self.hidden_size],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(xin) + dt_raw).clamp(self.dt_min, self.dt_max)

        if mask is not None:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"SSM mask shape {tuple(mask.shape)}; expected {(batch_size, seq_len)}"
                )
            mask_f = mask.unsqueeze(-1).to(dt.dtype)
            if bypass_decay > 0.0:
                # Opt-in: gamma-decay the state on bypassed steps (models
                # temporal discharge over omitted steps) instead of exact
                # preservation. bypass_decay=0.0 keeps h_t = h_{t-1} exactly.
                dt = dt * mask_f + bypass_decay * (1.0 - mask_f)
            else:
                dt = dt * mask_f

        a_matrix = -torch.exp(self.A_log.float())
        if state is None:
            h_init = torch.zeros(
                batch_size,
                self.hidden_size,
                self.state_size,
                dtype=torch.float32,
                device=x.device,
            )
        else:
            self._validate_state(state, batch_size)
            h_init = state.to(device=x.device, dtype=torch.float32)

        y, new_state = self._scan(
            xin,
            b_matrix,
            c_matrix,
            dt,
            a_matrix,
            self.D,
            h_init,
        )
        output = self.out_proj(y * F.silu(gate))
        return output, new_state


# =============================================================================
# 4. EXFUSION PATHS
# =============================================================================


class MemoryBankExFusionFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.memory_bank = nn.Parameter(
            torch.randn(num_experts, hidden_size) / math.sqrt(hidden_size)
        )
        self.router = nn.Linear(hidden_size, num_experts)
        self.difficulty_bias = nn.Parameter(torch.randn(num_experts) * 0.02)
        self.expert_ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.GELU(),
                    nn.Linear(intermediate_size, hidden_size),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        difficulty_metrics: Optional[Dict[str, Tensor]] = None,
        **_: Any,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        router_logits = self.router(hidden_states)
        memory_similarity = torch.einsum(
            "blh,eh->ble",
            hidden_states,
            self.memory_bank,
        )
        router_logits = router_logits + 0.1 * memory_similarity
        if difficulty_metrics is not None:
            difficulty = difficulty_metrics.get("difficulty_score")
            if difficulty is not None:
                router_logits = router_logits + (
                    _align_difficulty(difficulty, batch_size, seq_len)
                    * self.difficulty_bias
                )

        routing_weights = F.softmax(router_logits, dim=-1)
        output = torch.zeros_like(hidden_states)
        for expert_index, expert in enumerate(self.expert_ffn):
            output += (
                routing_weights[..., expert_index:expert_index + 1]
                * expert(hidden_states)
            )
        return self.layer_norm(output + hidden_states)


class MemoryBankExFusionMamba(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        num_experts: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_experts = num_experts
        self.memory_bank = nn.Parameter(
            torch.randn(num_experts, hidden_size) / math.sqrt(hidden_size)
        )
        self.router = nn.Linear(hidden_size, num_experts)
        self.difficulty_bias = nn.Parameter(torch.randn(num_experts) * 0.02)
        self.expert_mamba = nn.ModuleList(
            [SelectiveSSM(hidden_size, state_size) for _ in range(num_experts)]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # opt-in bypass decay (0.0 = exact state preservation)
        self.bypass_decay: float = 0.0

    def _validate_state_list(
        self,
        state: Optional[List[Optional[Tensor]]],
    ) -> None:
        if state is not None and len(state) != self.num_experts:
            raise ValueError(
                f"Mamba state list length {len(state)} != expert count {self.num_experts}"
            )

    def forward(
        self,
        hidden_states: Tensor,
        difficulty_metrics: Optional[Dict[str, Tensor]] = None,
        state: Optional[List[Optional[Tensor]]] = None,
        return_state: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        self._validate_state_list(state)
        batch_size, seq_len, _ = hidden_states.shape
        router_logits = self.router(hidden_states)
        memory_similarity = torch.einsum(
            "blh,eh->ble",
            hidden_states,
            self.memory_bank,
        )
        router_logits = router_logits + 0.1 * memory_similarity
        if difficulty_metrics is not None:
            difficulty = difficulty_metrics.get("difficulty_score")
            if difficulty is not None:
                router_logits = router_logits + (
                    _align_difficulty(difficulty, batch_size, seq_len)
                    * self.difficulty_bias
                )

        routing_weights = F.softmax(router_logits, dim=-1)
        output = torch.zeros_like(hidden_states)
        next_states: List[Tensor] = []
        for expert_index, expert in enumerate(self.expert_mamba):
            state_in = None if state is None else state[expert_index]
            expert_output, state_out = expert(
                hidden_states, state_in, mask,
                bypass_decay=self.bypass_decay,
            )
            next_states.append(state_out)
            output += (
                routing_weights[..., expert_index:expert_index + 1]
                * self.dropout(expert_output)
            )

        output = self.layer_norm(output + hidden_states)
        return output, next_states if return_state else None


class CheapPath(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: Tensor, **_: Any) -> Tensor:
        return self.norm(self.linear(hidden_states) + hidden_states)


# =============================================================================
# 5. ROUTER
# =============================================================================


class PredictiveDifficultyMacroRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_paths: int = 4,
        hidden_router: int = 64,
        granularity: str = "batch",
    ) -> None:
        super().__init__()
        if granularity not in {"batch", "token"}:
            raise ValueError("granularity must be 'batch' or 'token'")
        if num_paths <= 0:
            raise ValueError("num_paths must be positive")
        self.num_paths = num_paths
        self.granularity = granularity
        self.input_norm = nn.LayerNorm(hidden_size)
        self.router_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_router),
            nn.GELU(),
            nn.Linear(hidden_router, hidden_router),
            nn.GELU(),
        )
        self.difficulty_proj = nn.Linear(4, hidden_router)
        self.final_router = nn.Linear(hidden_router, num_paths)

    def forward(
        self,
        hidden_states: Tensor,
        difficulty_metrics: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        normalized = self.input_norm(hidden_states)
        features = normalized.mean(dim=1) if self.granularity == "batch" else normalized
        router_features = self.router_net(features)

        if difficulty_metrics is not None:
            batch_size, seq_len = hidden_states.shape[:2]
            components: List[Tensor] = []
            for key in ("entropy", "max_prob", "variance", "difficulty_score"):
                value = difficulty_metrics.get(key)
                if value is None:
                    shape = (batch_size,) if self.granularity == "batch" else (batch_size, seq_len)
                    value = torch.zeros(shape, device=hidden_states.device)
                value = value.float().to(hidden_states.device)

                if self.granularity == "batch":
                    while value.dim() > 1:
                        value = value.mean(dim=-1)
                    if value.shape[0] != batch_size:
                        value = value.mean().reshape(1).expand(batch_size)
                else:
                    if value.dim() == 3:
                        value = value.mean(dim=-1)
                    if value.dim() == 1:
                        if value.shape[0] == seq_len and batch_size == 1:
                            value = value.view(1, seq_len)
                        elif value.shape[0] == batch_size:
                            value = value.unsqueeze(1).expand(batch_size, seq_len)
                        else:
                            value = value.mean().reshape(1, 1).expand(batch_size, seq_len)
                    elif value.shape != (batch_size, seq_len):
                        value = value.mean().reshape(1, 1).expand(batch_size, seq_len)
                components.append(value)

            difficulty_vector = torch.stack(components, dim=-1)
            router_features = router_features + self.difficulty_proj(difficulty_vector)

        return self.final_router(router_features)


def router_auxiliary_loss(path_probs: Tensor, num_paths: int) -> Tensor:
    """Canonical Switch-Transformer load balance + uncertainty penalty.

    f_i = fraction of tokens HARD assigned to path i (argmax, detached);
    P_i = mean soft router probability. (The v1.9.1 form collapsed to
    num_paths * sum(mean_p^2).)
    """
    flattened = path_probs.reshape(-1, path_probs.shape[-1])
    with torch.no_grad():
        hard_idx = flattened.argmax(dim=-1)
        f_i = torch.stack([(hard_idx == i).float().mean()
                           for i in range(num_paths)])
    P_i = flattened.mean(dim=0)
    load_balance = num_paths * (f_i * P_i).sum()
    entropy = -(
        flattened * torch.log(flattened.clamp_min(1e-10))
    ).sum(dim=-1).mean()
    # Minimizing this term penalizes both collapse and excessive uncertainty.
    return load_balance + 0.1 * entropy


def _top_p_probabilities(probabilities: Tensor, top_p: float) -> Tensor:
    _validate_probability("routing_top_p", top_p)
    if top_p <= 0:
        raise ValueError("routing_top_p must be > 0")

    sorted_probs, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Keep the smallest prefix whose cumulative mass reaches top_p. Subtracting
    # the current probability means the threshold-crossing entry is retained.
    keep_sorted = (cumulative - sorted_probs) < top_p
    keep_sorted[..., 0] = True

    filtered_sorted = sorted_probs * keep_sorted.to(sorted_probs.dtype)
    filtered = torch.zeros_like(probabilities).scatter(-1, sorted_indices, filtered_sorted)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)


# =============================================================================
# 6. FULL HYBRID DECODER LAYER
# =============================================================================


@dataclass
class DAPHConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    state_size: int = 16
    num_experts: int = 2
    dropout: float = 0.1
    num_paths: int = 4
    routing_granularity: str = "batch"
    routing_mode: str = "hard"  # hard | soft | top_p
    routing_top_p: float = 0.9
    attn_history_window: Optional[int] = None
    attn_window: Optional[int] = None
    # attention_mask convention: "hf" (1/True = valid token) or "pytorch"
    # (bool True = blocked); converted at the attention boundary
    mask_convention: str = "hf"
    # Number of initial "attention sink" tokens anchored in the cache when
    # the sliding window trims history (StreamingLLM). 0 disables.
    attn_sink_tokens: int = 4
    # Opt-in continuous decay of the SSM state on bypassed steps (0.0 =
    # exact preservation, the bit-exact guarantee from the drift fix).
    ssm_bypass_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if self.num_attention_heads <= 0 or self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size")
        if self.state_size <= 0 or self.num_experts <= 0:
            raise ValueError("state_size and num_experts must be positive")
        if self.num_paths not in {4, 5}:
            raise ValueError("This implementation currently requires num_paths in {4, 5}")
        if self.routing_granularity not in {"batch", "token"}:
            raise ValueError("routing_granularity must be 'batch' or 'token'")
        if self.routing_mode not in {"hard", "soft", "top_p"}:
            raise ValueError("routing_mode must be 'hard', 'soft', or 'top_p'")
        _validate_probability("dropout", self.dropout)
        _validate_probability("routing_top_p", self.routing_top_p)
        if self.routing_mode == "top_p" and self.routing_top_p <= 0:
            raise ValueError("routing_top_p must be > 0 for top_p routing")
        if self.mask_convention not in {"hf", "pytorch"}:
            raise ValueError("mask_convention must be 'hf' or 'pytorch'")
        if self.attn_sink_tokens < 0:
            raise ValueError("attn_sink_tokens must be >= 0")
        if not math.isfinite(self.ssm_bypass_decay) or self.ssm_bypass_decay < 0:
            raise ValueError("ssm_bypass_decay must be finite and >= 0")
        for name, value in (
            ("attn_history_window", self.attn_history_window),
            ("attn_window", self.attn_window),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")


class DAPHHybridDecoderLayer(nn.Module):
    ATTENTION_PATH = 0
    MAMBA_PATH = 1
    TRANSFORMER_PATH = 2
    CHEAP_PATH = 3

    def __init__(self, config: DAPHConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.attention = nn.MultiheadAttention(
            hidden_size,
            config.num_attention_heads,
            batch_first=True,
            dropout=config.dropout,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.mamba_exfusion = MemoryBankExFusionMamba(
            hidden_size,
            config.state_size,
            config.num_experts,
            config.dropout,
        )
        self.mamba_exfusion.bypass_decay = config.ssm_bypass_decay
        self.trans_exfusion = MemoryBankExFusionFFN(
            hidden_size,
            config.intermediate_size,
            config.num_experts,
            config.dropout,
        )
        self.cheap_path = CheapPath(hidden_size)
        self.macro_router = PredictiveDifficultyMacroRouter(
            hidden_size,
            config.num_paths,
            granularity=config.routing_granularity,
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    @property
    def attention_window(self) -> Optional[int]:
        return self.config.attn_history_window or self.config.attn_window

    @staticmethod
    def _causal_attention_mask(
        query_length: int,
        history_length: int,
        device: torch.device,
    ) -> Tensor:
        key_length = history_length + query_length
        query_positions = history_length + torch.arange(query_length, device=device)
        key_positions = torch.arange(key_length, device=device)
        return key_positions.unsqueeze(0) > query_positions.unsqueeze(1)

    def _current_valid_mask(
        self,
        attention_mask: Optional[Tensor],
        batch_size: int,
        current_length: int,
        device: torch.device,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones(
                batch_size,
                current_length,
                dtype=torch.bool,
                device=device,
            )
        if attention_mask.dim() != 2 or attention_mask.shape != (batch_size, current_length):
            raise ValueError(
                "attention_mask must have shape (B, current_length) with 1/True for valid "
                f"tokens; got {tuple(attention_mask.shape)}"
            )
        mask = attention_mask.to(device=device).bool()
        if self.config.mask_convention == "pytorch":
            mask = ~mask  # pytorch: True = blocked -> invert to valid mask
        return mask

    def _normalize_key_padding_mask(
        self,
        attention_mask: Optional[Tensor],
        attn_padding_state: Optional[Tensor],
        batch_size: int,
        current_length: int,
        history_length: int,
        device: torch.device,
        valid_current: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if valid_current is None:
            valid_current = self._current_valid_mask(
                attention_mask,
                batch_size,
                current_length,
                device,
            )
        current_padding = ~valid_current
        if history_length == 0:
            return current_padding if attention_mask is not None else None

        if attn_padding_state is None:
            history_padding = torch.zeros(
                batch_size,
                history_length,
                dtype=torch.bool,
                device=device,
            )
        else:
            if attn_padding_state.shape != (batch_size, history_length):
                raise ValueError(
                    f"attn_padding_state shape {tuple(attn_padding_state.shape)}; "
                    f"expected {(batch_size, history_length)}"
                )
            history_padding = attn_padding_state.to(device=device).bool()
        return torch.cat([history_padding, current_padding], dim=1)

    def _run_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        attn_state: Optional[Tensor],
        attn_padding_state: Optional[Tensor],
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, current_length, _ = hidden_states.shape
        history_length = 0 if attn_state is None else attn_state.shape[1]
        key_value = hidden_states if attn_state is None else torch.cat(
            [attn_state.to(hidden_states.device, hidden_states.dtype), hidden_states],
            dim=1,
        )
        causal_mask = self._causal_attention_mask(
            current_length,
            history_length,
            hidden_states.device,
        )
        key_padding_mask = self._normalize_key_padding_mask(
            attention_mask,
            attn_padding_state,
            batch_size,
            current_length,
            history_length,
            hidden_states.device,
            valid_current=valid_mask,
        )
        attn_output, _ = self.attention(
            query=hidden_states,
            key=key_value,
            value=key_value,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.attn_norm(attn_output + hidden_states)

    def _path_outputs(
        self,
        hidden_states: Tensor,
        difficulty_metrics: Dict[str, Tensor],
        attention_mask: Optional[Tensor],
        mamba_mask: Tensor,
        mamba_state: Optional[List[Optional[Tensor]]],
        attn_state: Optional[Tensor],
        attn_padding_state: Optional[Tensor],
        use_cache: bool,
        required_paths: Iterable[int],
        valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Dict[int, Tensor], Optional[List[Tensor]]]:
        required = set(required_paths)
        outputs: Dict[int, Tensor] = {}
        next_mamba_state: Optional[List[Tensor]] = None

        if self.ATTENTION_PATH in required:
            outputs[self.ATTENTION_PATH] = self._run_attention(
                hidden_states,
                attention_mask,
                attn_state,
                attn_padding_state,
                valid_mask=valid_mask,
            )

        if self.MAMBA_PATH in required or use_cache:
            mamba_output, next_mamba_state = self.mamba_exfusion(
                hidden_states,
                difficulty_metrics,
                state=mamba_state,
                return_state=use_cache,
                mask=mamba_mask,
            )
            if self.MAMBA_PATH in required:
                outputs[self.MAMBA_PATH] = mamba_output

        if self.TRANSFORMER_PATH in required:
            outputs[self.TRANSFORMER_PATH] = self.trans_exfusion(
                hidden_states,
                difficulty_metrics,
            )

        if self.CHEAP_PATH in required:
            outputs[self.CHEAP_PATH] = self.cheap_path(hidden_states)

        return outputs, next_mamba_state

    def _weighted_route(
        self,
        hidden_states: Tensor,
        route_weights: Tensor,
        path_outputs: Dict[int, Tensor],
    ) -> Tensor:
        stacked = torch.stack(
            [path_outputs[path] for path in range(self.config.num_paths)],
            dim=-2,
        )
        if route_weights.dim() == 2:
            route_weights = route_weights.unsqueeze(1).expand(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.config.num_paths,
            )
        return (
            stacked * route_weights.unsqueeze(-1)
        ).sum(dim=-2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        difficulty_metrics: Optional[Dict[str, Tensor]] = None,
        use_cache: bool = False,
        mamba_state: Optional[List[Optional[Tensor]]] = None,
        attn_state: Optional[Tensor] = None,
        attn_padding_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must have shape (B, L, H); got {tuple(hidden_states.shape)}"
            )
        batch_size, seq_len, _ = hidden_states.shape
        valid_token_mask = self._current_valid_mask(
            attention_mask,
            batch_size,
            seq_len,
            hidden_states.device,
        )
        if difficulty_metrics is None:
            difficulty_metrics = compute_difficulty_from_hidden_states(hidden_states)

        router_logits = self.macro_router(hidden_states, difficulty_metrics)
        path_probs = F.softmax(router_logits, dim=-1)
        meta: Dict[str, Any] = {
            "router_logits": router_logits,
            "path_probs": path_probs,
            "difficulty": difficulty_metrics,
            "router_aux_loss": router_auxiliary_loss(path_probs, self.config.num_paths),
        }

        routing_mode = self.config.routing_mode
        if self.training and routing_mode == "hard":
            # Differentiable training defaults to soft mixing even when hard inference is configured.
            effective_mode = "soft"
        else:
            effective_mode = routing_mode

        if effective_mode == "hard":
            selected = path_probs.argmax(dim=-1)
            if selected.dim() == 1:
                mamba_mask = (
                    selected.eq(self.MAMBA_PATH)
                    .unsqueeze(1)
                    .expand(batch_size, seq_len)
                    .to(hidden_states.dtype)
                    * valid_token_mask.to(hidden_states.dtype)
                )
                required_paths = selected.unique().tolist()
            else:
                mamba_mask = (
                    selected.eq(self.MAMBA_PATH).to(hidden_states.dtype)
                    * valid_token_mask.to(hidden_states.dtype)
                )
                required_paths = selected.unique().tolist()

            # Token-level hard routing: sequence-dependent paths (Attention,
            # Mamba) run on the full sequence; POINTWISE paths (Trans-ExFusion,
            # Cheap) use sparse gather/scatter over only the selected tokens,
            # preserving each token's own difficulty values.
            token_level = selected.dim() == 2
            if token_level:
                seq_paths = [p for p in required_paths
                             if p in (self.ATTENTION_PATH, self.MAMBA_PATH)]
                point_paths = [p for p in required_paths
                               if p in (self.TRANSFORMER_PATH, self.CHEAP_PATH)]
            else:
                seq_paths = required_paths
                point_paths = []

            path_outputs, next_mamba_state = self._path_outputs(
                hidden_states,
                difficulty_metrics,
                attention_mask,
                mamba_mask,
                mamba_state,
                attn_state,
                attn_padding_state,
                use_cache,
                seq_paths,
                valid_mask=valid_token_mask,
            )

            output = torch.zeros_like(hidden_states)
            if not token_level:
                for path in required_paths:
                    batch_selector = selected.eq(path)
                    output[batch_selector] = path_outputs[path][batch_selector]
            else:
                for path in seq_paths:
                    token_selector = selected.eq(path).unsqueeze(-1)
                    output = torch.where(
                        token_selector,
                        path_outputs[path],
                        output,
                    )
                for path in point_paths:
                    token_selector = selected.eq(path)          # [B, L]
                    if not token_selector.any():
                        continue
                    b_idx, t_idx = torch.where(token_selector)
                    sparse_tokens = hidden_states[b_idx, t_idx].unsqueeze(1)
                    sparse_dm = _gather_difficulty(
                        difficulty_metrics, b_idx, t_idx,
                        batch_size, seq_len)
                    if path == self.TRANSFORMER_PATH:
                        sparse_out = self.trans_exfusion(sparse_tokens, sparse_dm)
                    else:
                        sparse_out = self.cheap_path(sparse_tokens)
                    output[b_idx, t_idx] = sparse_out.squeeze(1)
            meta["selected_paths"] = selected

        else:
            route_weights = path_probs if effective_mode == "soft" else _top_p_probabilities(
                path_probs,
                self.config.routing_top_p,
            )
            mamba_mask = valid_token_mask.to(hidden_states.dtype)
            path_outputs, next_mamba_state = self._path_outputs(
                hidden_states,
                difficulty_metrics,
                attention_mask,
                mamba_mask,
                mamba_state,
                attn_state,
                attn_padding_state,
                use_cache,
                range(self.config.num_paths),
                valid_mask=valid_token_mask,
            )
            output = self._weighted_route(hidden_states, route_weights, path_outputs)
            meta["route_weights"] = route_weights

        output = self.final_norm(output)

        if use_cache:
            if next_mamba_state is None:
                raise RuntimeError("use_cache=True did not produce Mamba state")
            meta["mamba_state"] = next_mamba_state

            detached_current = hidden_states.detach()
            current_padding = ~valid_token_mask
            new_attn_state = detached_current if attn_state is None else torch.cat(
                [attn_state.detach(), detached_current],
                dim=1,
            )
            if attn_state is None:
                if attn_padding_state is not None:
                    raise ValueError("attn_padding_state requires attn_state")
                new_attn_padding_state = current_padding
            else:
                if attn_padding_state is None:
                    previous_padding = torch.zeros(
                        batch_size,
                        attn_state.shape[1],
                        dtype=torch.bool,
                        device=current_padding.device,
                    )
                else:
                    if attn_padding_state.shape != (batch_size, attn_state.shape[1]):
                        raise ValueError(
                            f"attn_padding_state shape {tuple(attn_padding_state.shape)}; "
                            f"expected {(batch_size, attn_state.shape[1])}"
                        )
                    previous_padding = attn_padding_state.detach().to(
                        current_padding.device
                    ).bool()
                new_attn_padding_state = torch.cat(
                    [previous_padding, current_padding],
                    dim=1,
                )
            window = self.attention_window
            sinks = self.config.attn_sink_tokens
            if window is not None and new_attn_state.shape[1] > window:
                if 0 < sinks < window and new_attn_state.shape[1] > sinks:
                    # Anchor the first `sinks` tokens (attention sinks,
                    # arXiv:2309.17453) and slide the remainder of the window.
                    new_attn_state = torch.cat(
                        [new_attn_state[:, :sinks],
                         new_attn_state[:, -(window - sinks):]], dim=1)
                    new_attn_padding_state = torch.cat(
                        [new_attn_padding_state[:, :sinks],
                         new_attn_padding_state[:, -(window - sinks):]], dim=1)
                else:
                    new_attn_state = new_attn_state[:, -window:, :]
                    new_attn_padding_state = new_attn_padding_state[:, -window:]
            meta["attn_state"] = new_attn_state
            meta["attn_padding_state"] = new_attn_padding_state

        return output, meta


# =============================================================================
# 7. DELTA APPLICATION AND ARCHITECTURE-AWARE MERGE ENTRY
# =============================================================================


def _resolve_suffix_matches(
    target_parameters: Dict[str, nn.Parameter],
    name: str,
    shape: torch.Size,
) -> List[str]:
    exact = [
        key for key, parameter in target_parameters.items()
        if key == name and parameter.shape == shape
    ]
    if exact:
        return exact

    dotted = [
        key for key, parameter in target_parameters.items()
        if key.endswith("." + name) and parameter.shape == shape
    ]
    if dotted:
        return dotted

    terminal_name = name.split(".")[-1]
    return [
        key for key, parameter in target_parameters.items()
        if key.split(".")[-1] == terminal_name and parameter.shape == shape
    ]


def _apply_delta_to_module(
    target: nn.Module,
    delta: Dict[str, Tensor],
    scale: float = 1.0,
    require_full_coverage: bool = True,
) -> Tuple[int, List[str], List[str]]:
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")

    target_parameters = dict(target.named_parameters())
    applied = 0
    ambiguous: List[str] = []
    unmatched: List[str] = []

    with torch.no_grad():
        for name, value in delta.items():
            matches = _resolve_suffix_matches(target_parameters, name, value.shape)
            if len(matches) == 1:
                parameter = target_parameters[matches[0]]
                parameter.add_(value.to(parameter.device, parameter.dtype) * scale)
                applied += 1
            elif len(matches) > 1:
                description = f"{name} -> {matches}"
                ambiguous.append(description)
            else:
                unmatched.append(name)

    if ambiguous:
        warnings.warn(f"Ambiguous delta applications: {ambiguous[:8]}")
    if unmatched:
        warnings.warn(f"Unmatched merge deltas: {unmatched[:8]}")
    if require_full_coverage and (ambiguous or unmatched):
        raise ValueError(
            f"Delta application incomplete: {len(ambiguous)} ambiguous, "
            f"{len(unmatched)} unmatched"
        )
    return applied, ambiguous, unmatched


def merge_expert_family(
    experts: Sequence[nn.Module],
    base_model: nn.Module,
    memory_bank_weights: Tensor,
    difficulty_importance: Optional[Tensor] = None,
    calibration_batch: Optional[StructuredBatch] = None,
    apply_to: Optional[nn.Module] = None,
    policies: Optional[Mapping[str, Any]] = None,
    kfac_tracker: Optional[Any] = None,
    forward_fn: Optional[ForwardFn] = None,
    loss_fn: Optional[LossFn] = None,
    scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Tensor]:
    if not experts:
        return {}
    policy = {**DEFAULT_MAMBA_POLICIES, **dict(policies or {})}
    weights = _normalize_expert_weights(
        memory_bank_weights,
        len(experts),
        difficulty_importance,
    )

    task_vectors = extract_task_vectors(experts, base_model)

    # merge_mode="weighted_average": plain weighted average of task vectors,
    # no DARE drop / TIES trim-elect / Fisher boost. Available because
    # ExFusion ablations found trimming matched-or-underperformed plain
    # averaging for co-trained experts.
    merge_mode = str(policy.get("merge_mode", "full"))
    if merge_mode == "weighted_average":
        names = _validate_homogeneous_task_vectors(task_vectors)
        merged = {
            name: sum(weights[i] * task_vectors[i][name]
                      for i in range(len(task_vectors)))
            for name in names
        }
        if apply_to is not None:
            _apply_delta_to_module(apply_to, merged, scale=scale,
                                   require_full_coverage=True)
        return merged
    if merge_mode != "full":
        raise ValueError(f"Unknown merge_mode '{merge_mode}' "
                         f"(expected 'full' or 'weighted_average')")
    task_vectors, keep_masks = apply_dare_preprocessing(
        task_vectors,
        difficulty_importance,
        dare_base_p=float(policy["dare_base_p"]),
        ssm_drop_reduction=float(policy["ssm_drop_reduction"]),
        policies=policy,
        generator=generator,
    )

    ties_delta = difficulty_weighted_ties_merge(
        task_vectors,
        memory_bank_weights,
        difficulty_importance,
        trim_ratio=float(policy["ties_trim_ratio"]),
        ssm_soft_merge=bool(policy["ssm_soft_merge"]),
        sign_mode=str(policy["sign_mode"]),
        policies=policy,
    )

    fisher_diagonals: List[Dict[str, Tensor]] = []
    if kfac_tracker is not None:
        fisher_diagonals = [
            build_fisher_diagonals_from_tracker(expert, kfac_tracker)
            for expert in experts
        ]
    elif calibration_batch is not None:
        fisher_diagonals = [
            build_empirical_fisher_diagonals(
                expert,
                calibration_batch,
                forward_fn=forward_fn,
                loss_fn=loss_fn,
                device=next(expert.parameters()).device,
            )
            for expert in experts
        ]
    else:
        fisher_floor = float(policy["fisher_floor"])
        fisher_diagonals = [
            {
                name: torch.full_like(value, fisher_floor)
                for name, value in task_vector.items()
            }
            for task_vector in task_vectors
        ]

    fisher_delta = difficulty_weighted_fisher_merge(
        task_vectors,
        fisher_diagonals,
        memory_bank_weights,
        difficulty_importance,
        dare_keep_masks=keep_masks,
        fisher_power=float(policy["fisher_power"]),
        fisher_floor=float(policy["fisher_floor"]),
    )

    merged = blend_ties_and_fisher(
        ties_delta,
        fisher_delta,
        ties_fisher_blend=float(policy["ties_fisher_blend"]),
        ssm_fisher_boost=float(policy["ssm_fisher_boost"]),
        policies=policy,
    )

    if apply_to is not None:
        _apply_delta_to_module(apply_to, merged, scale=scale, require_full_coverage=True)
    return merged


def merge_mamba_transformer_hybrid(
    mamba_experts: Sequence[nn.Module],
    transformer_experts: Sequence[nn.Module],
    base_model: Optional[nn.Module] = None,
    layer: Optional[DAPHHybridDecoderLayer] = None,
    memory_bank_weights: Optional[Tensor] = None,
    difficulty_importance: Optional[Tensor] = None,
    calibration_batch: Optional[StructuredBatch] = None,
    apply_to: Optional[nn.Module] = None,
    policies: Optional[Mapping[str, Any]] = None,
    kfac_tracker: Optional[Any] = None,
    scale: float = 1.0,
    *,
    mamba_base_model: Optional[nn.Module] = None,
    transformer_base_model: Optional[nn.Module] = None,
    mamba_memory_bank_weights: Optional[Tensor] = None,
    transformer_memory_bank_weights: Optional[Tensor] = None,
    mamba_difficulty_importance: Optional[Tensor] = None,
    transformer_difficulty_importance: Optional[Tensor] = None,
    mamba_apply_to: Optional[nn.Module] = None,
    transformer_apply_to: Optional[nn.Module] = None,
    forward_fn: Optional[ForwardFn] = None,
    loss_fn: Optional[LossFn] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Dict[str, Tensor]]:
    """Merge Mamba and Transformer expert families without namespace mixing.

    The original positional API remains usable for a *single* expert family:
    ``base_model``, ``memory_bank_weights``, ``difficulty_importance``, and
    ``apply_to`` are assigned to that family. When both families are present,
    callers must provide the family-specific keyword arguments.

    ``layer`` is retained for source compatibility but no longer triggers an
    implicit delta application. Automatic application to an independently
    initialized layer was mathematically invalid; pass an explicit target that
    is a clone of the corresponding base module.
    """
    del layer  # compatibility-only argument; intentionally not used for application

    has_mamba = bool(mamba_experts)
    has_transformer = bool(transformer_experts)
    if not has_mamba and not has_transformer:
        raise ValueError("At least one expert family must be provided")

    if has_mamba and has_transformer:
        shared_arguments_used = any(
            value is not None
            for value in (
                base_model,
                memory_bank_weights,
                difficulty_importance,
                apply_to,
            )
        )
        if shared_arguments_used:
            raise ValueError(
                "A heterogeneous Mamba+Transformer merge cannot use shared base, "
                "weight, difficulty, or target arguments. Supply the corresponding "
                "mamba_* and transformer_* keyword arguments."
            )
    elif has_mamba:
        if mamba_base_model is None:
            mamba_base_model = base_model
        if mamba_memory_bank_weights is None:
            mamba_memory_bank_weights = memory_bank_weights
        if mamba_difficulty_importance is None:
            mamba_difficulty_importance = difficulty_importance
        if mamba_apply_to is None:
            mamba_apply_to = apply_to
    else:
        if transformer_base_model is None:
            transformer_base_model = base_model
        if transformer_memory_bank_weights is None:
            transformer_memory_bank_weights = memory_bank_weights
        if transformer_difficulty_importance is None:
            transformer_difficulty_importance = difficulty_importance
        if transformer_apply_to is None:
            transformer_apply_to = apply_to

    result: Dict[str, Dict[str, Tensor]] = {"mamba": {}, "transformer": {}}

    if has_mamba:
        if mamba_base_model is None:
            raise ValueError("mamba_base_model is required when mamba_experts are provided")
        if mamba_memory_bank_weights is None:
            raise ValueError("mamba_memory_bank_weights is required")
        result["mamba"] = merge_expert_family(
            mamba_experts,
            mamba_base_model,
            mamba_memory_bank_weights,
            mamba_difficulty_importance,
            calibration_batch,
            apply_to=mamba_apply_to,
            policies=policies,
            kfac_tracker=kfac_tracker,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
            scale=scale,
            generator=generator,
        )

    if has_transformer:
        if transformer_base_model is None:
            raise ValueError(
                "transformer_base_model is required when transformer_experts are provided"
            )
        if transformer_memory_bank_weights is None:
            raise ValueError("transformer_memory_bank_weights is required")
        result["transformer"] = merge_expert_family(
            transformer_experts,
            transformer_base_model,
            transformer_memory_bank_weights,
            transformer_difficulty_importance,
            calibration_batch,
            apply_to=transformer_apply_to,
            policies=policies,
            kfac_tracker=kfac_tracker,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
            scale=scale,
            generator=generator,
        )

    return result


# =============================================================================
# 8. DETERMINISTIC SELF-TESTS
# =============================================================================


class _ForcedRouter(nn.Module):
    def __init__(self, num_paths: int, selected_path: int, granularity: str = "token") -> None:
        super().__init__()
        self.num_paths = num_paths
        self.selected_path = selected_path
        self.granularity = granularity

    def forward(self, hidden_states: Tensor, difficulty_metrics: Optional[Dict[str, Tensor]] = None) -> Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        shape = (batch_size, self.num_paths) if self.granularity == "batch" else (
            batch_size,
            seq_len,
            self.num_paths,
        )
        logits = torch.full(shape, -30.0, device=hidden_states.device)
        logits[..., self.selected_path] = 30.0
        return logits


def _make_cloned_experts(base: nn.Module, num_experts: int) -> List[nn.Module]:
    experts: List[nn.Module] = []
    for expert_index in range(num_experts):
        expert = copy.deepcopy(base)
        with torch.no_grad():
            for parameter_index, parameter in enumerate(expert.parameters()):
                if parameter.is_floating_point():
                    parameter.add_(
                        (expert_index + 1)
                        * (parameter_index + 1)
                        * 1e-4
                        * torch.ones_like(parameter)
                    )
        experts.append(expert)
    return experts


def run_self_test() -> None:
    torch.manual_seed(0)

    # 1. Causal attention with history and a valid-token mask.
    attention_config = DAPHConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        state_size=4,
        num_experts=2,
        dropout=0.0,
        routing_granularity="token",
        routing_mode="hard",
        attn_history_window=4,
    )
    attention_layer = DAPHHybridDecoderLayer(attention_config).eval()
    attention_layer.macro_router = _ForcedRouter(4, 0, "token")
    history = torch.randn(2, 2, 32)
    current = torch.randn(2, 2, 32)
    valid_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    attention_output, attention_meta = attention_layer(
        current,
        attention_mask=valid_mask,
        use_cache=True,
        attn_state=history,
    )
    assert attention_output.shape == current.shape
    assert attention_meta["attn_state"].shape == (2, 4, 32)
    assert attention_meta["attn_padding_state"].shape == (2, 4)
    assert attention_meta["attn_padding_state"][1, -1]

    # Future-token invariance proves full-sequence attention is causal.
    x1 = torch.randn(1, 3, 32)
    x2 = x1.clone()
    x2[:, 2] += 100.0
    y1, _ = attention_layer(x1)
    y2, _ = attention_layer(x2)
    assert torch.allclose(y1[:, :2], y2[:, :2], atol=1e-5, rtol=1e-5)

    # Forced attention full-sequence and token-streaming results must agree.
    stream_input = torch.randn(2, 5, 32)
    attention_full, _ = attention_layer(stream_input)
    stream_outputs: List[Tensor] = []
    stream_mamba_state = None
    stream_attn_state = None
    stream_padding_state = None
    for token_index in range(stream_input.shape[1]):
        token_output, token_meta = attention_layer(
            stream_input[:, token_index:token_index + 1],
            use_cache=True,
            mamba_state=stream_mamba_state,
            attn_state=stream_attn_state,
            attn_padding_state=stream_padding_state,
        )
        stream_outputs.append(token_output)
        stream_mamba_state = token_meta["mamba_state"]
        stream_attn_state = token_meta["attn_state"]
        stream_padding_state = token_meta["attn_padding_state"]
    attention_stream = torch.cat(stream_outputs, dim=1)
    assert torch.allclose(attention_full, attention_stream, atol=2e-5, rtol=2e-5)

    # 2. Mamba cache affects output and the path is called once per invocation.
    mamba_layer = DAPHHybridDecoderLayer(attention_config).eval()
    mamba_layer.macro_router = _ForcedRouter(4, 1, "token")
    call_count = 0
    original_forward = mamba_layer.mamba_exfusion.forward

    def counted_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original_forward(*args, **kwargs)

    mamba_layer.mamba_exfusion.forward = counted_forward  # type: ignore[method-assign]
    token = torch.randn(1, 1, 32)
    first_output, first_meta = mamba_layer(token, use_cache=True)
    assert call_count == 1
    second_with_state, second_meta = mamba_layer(
        token,
        use_cache=True,
        mamba_state=first_meta["mamba_state"],
        attn_state=first_meta["attn_state"],
        attn_padding_state=first_meta["attn_padding_state"],
    )
    assert call_count == 2
    second_without_state, _ = mamba_layer(token, use_cache=True)
    assert not torch.allclose(second_with_state, second_without_state)
    assert len(second_meta["mamba_state"]) == attention_config.num_experts

    mamba_sequence = torch.randn(2, 5, 32)
    mamba_full, _ = mamba_layer(mamba_sequence)
    mamba_stream_outputs: List[Tensor] = []
    stream_mamba_state = None
    stream_attn_state = None
    stream_padding_state = None
    for token_index in range(mamba_sequence.shape[1]):
        token_output, token_meta = mamba_layer(
            mamba_sequence[:, token_index:token_index + 1],
            use_cache=True,
            mamba_state=stream_mamba_state,
            attn_state=stream_attn_state,
            attn_padding_state=stream_padding_state,
        )
        mamba_stream_outputs.append(token_output)
        stream_mamba_state = token_meta["mamba_state"]
        stream_attn_state = token_meta["attn_state"]
        stream_padding_state = token_meta["attn_padding_state"]
    mamba_stream = torch.cat(mamba_stream_outputs, dim=1)
    assert torch.allclose(mamba_full, mamba_stream, atol=2e-5, rtol=2e-5)

    if stream_mamba_state is not None:
        before_padding = [state.clone() for state in stream_mamba_state]
        _, padded_meta = mamba_layer(
            torch.randn(2, 1, 32),
            attention_mask=torch.zeros(2, 1),
            use_cache=True,
            mamba_state=stream_mamba_state,
            attn_state=stream_attn_state,
            attn_padding_state=stream_padding_state,
        )
        for state_before, state_after in zip(before_padding, padded_meta["mamba_state"]):
            assert torch.equal(state_before, state_after)

    # 3. Masked SSM steps leave recurrent state unchanged.
    ssm = SelectiveSSM(8, 3).eval()
    initial_state = torch.randn(2, 8, 3)
    _, unchanged_state = ssm(
        torch.randn(2, 4, 8),
        state=initial_state,
        mask=torch.zeros(2, 4),
    )
    assert torch.equal(unchanged_state, initial_state.float())

    # 4. TIES weighted denominator.
    ties_vectors = [
        {"weight": torch.tensor([2.0])},
        {"weight": torch.tensor([4.0])},
    ]
    ties_result = difficulty_weighted_ties_merge(
        ties_vectors,
        torch.tensor([0.0, 0.0]),
        trim_ratio=0.0,
        ssm_soft_merge=False,
    )
    assert torch.allclose(ties_result["weight"], torch.tensor([3.0]))

    # 5. Fisher denominator excludes DARE-dropped entries.
    fisher_vectors = [
        {"weight": torch.tensor([2.0])},
        {"weight": torch.tensor([0.0])},
    ]
    fisher_diagonals = [
        {"weight": torch.tensor([1.0])},
        {"weight": torch.tensor([100.0])},
    ]
    keep_masks = [
        {"weight": torch.tensor([True])},
        {"weight": torch.tensor([False])},
    ]
    fisher_result = difficulty_weighted_fisher_merge(
        fisher_vectors,
        fisher_diagonals,
        torch.tensor([0.0, 0.0]),
        dare_keep_masks=keep_masks,
    )
    assert torch.allclose(fisher_result["weight"], torch.tensor([2.0]))

    # 6. SSM Fisher boost materially changes the blend.
    ties = {"A_log": torch.tensor([0.0]), "linear.weight": torch.tensor([0.0])}
    fisher = {"A_log": torch.tensor([1.0]), "linear.weight": torch.tensor([1.0])}
    unboosted = blend_ties_and_fisher(ties, fisher, 0.5, 1.0)
    boosted = blend_ties_and_fisher(ties, fisher, 0.5, 4.0)
    assert boosted["A_log"].item() > unboosted["A_log"].item()
    assert torch.allclose(boosted["linear.weight"], unboosted["linear.weight"])

    # 7. Policy allowlist/blocklist propagation.
    policy = {
        "ssm_core_allowlist": ["special_param"],
        "ssm_core_blocklist": ["forbidden"],
    }
    assert is_ssm_core_param("module.special_param", policy)
    assert not is_ssm_core_param("module.forbidden.A_log", policy)

    # 8. Real task-vector merge applies to a base clone with full coverage.
    base = SelectiveSSM(8, 3)
    experts = _make_cloned_experts(base, 2)
    target = copy.deepcopy(base)
    generator = torch.Generator(device="cpu").manual_seed(123)
    merged = merge_expert_family(
        experts,
        base,
        memory_bank_weights=torch.tensor([0.0, 0.0]),
        difficulty_importance=torch.tensor([1.0, 1.0]),
        apply_to=target,
        policies={"dare_base_p": 0.0, "ties_trim_ratio": 0.0},
        generator=generator,
    )
    base_parameters = dict(base.named_parameters())
    target_parameters = dict(target.named_parameters())
    for name, delta in merged.items():
        assert torch.allclose(
            target_parameters[name],
            base_parameters[name] + delta.to(base_parameters[name].dtype),
            atol=1e-6,
            rtol=1e-6,
        )

    # 9. Top-p routing produces normalized sparse weights.
    top_p_config = copy.deepcopy(attention_config)
    top_p_config.routing_mode = "top_p"
    top_p_config.routing_top_p = 0.6
    top_p_layer = DAPHHybridDecoderLayer(top_p_config).eval()
    top_p_output, top_p_meta = top_p_layer(torch.randn(2, 3, 32))
    assert top_p_output.shape == (2, 3, 32)
    route_weights = top_p_meta["route_weights"]
    assert torch.allclose(route_weights.sum(dim=-1), torch.ones_like(route_weights[..., 0]))
    assert (route_weights == 0).any()

    # 10. Difficulty metrics remain finite for batch-size one and length one.
    difficulty = compute_difficulty_from_hidden_states(torch.randn(1, 1, 32))
    assert all(torch.isfinite(value).all() for value in difficulty.values())

    # 11. mask_convention="pytorch" inverts blocked masks to valid masks.
    cfg_pt = copy.deepcopy(attention_config)
    cfg_pt.mask_convention = "pytorch"
    layer_pt = DAPHHybridDecoderLayer(cfg_pt).eval()
    layer_pt.macro_router = _ForcedRouter(4, 0, "token")
    x_pt = torch.randn(2, 3, 32)
    none_blocked = torch.zeros(2, 3, dtype=torch.bool)  # pytorch: nothing blocked
    out_pt, _ = layer_pt(x_pt, attention_mask=none_blocked)
    out_none, _ = layer_pt(x_pt)                        # no mask == all valid
    assert torch.allclose(out_pt, out_none, atol=1e-5)
    mostly_blocked = torch.ones(2, 3, dtype=torch.bool)
    mostly_blocked[:, 0] = False
    out_blocked, _ = layer_pt(x_pt, attention_mask=mostly_blocked)
    assert not torch.allclose(out_blocked, out_none, atol=1e-5)
    print("11. mask_convention pytorch inversion (none-blocked == no mask) OK")

    # 12. Canonical Switch aux loss: hard f_i x soft P_i, gradient flows.
    probs = torch.tensor([[0.9, 0.05, 0.03, 0.02],
                          [0.8, 0.10, 0.05, 0.05]], requires_grad=True)
    lb = router_auxiliary_loss(probs, 4)
    lb.backward()
    assert probs.grad is not None and probs.grad.abs().sum() > 0
    confident_balanced = torch.tensor([
        [0.97, 0.01, 0.01, 0.01], [0.01, 0.97, 0.01, 0.01],
        [0.01, 0.01, 0.97, 0.01], [0.01, 0.01, 0.01, 0.97]])
    uncertain_uniform = torch.full((4, 4), 0.25)
    assert (router_auxiliary_loss(confident_balanced, 4)
            < router_auxiliary_loss(uncertain_uniform, 4))
    print("12. canonical Switch aux loss (grad + ordering) OK")

    # 13. merge_mode="weighted_average" applies and unknown modes raise.
    wa_target = copy.deepcopy(base)
    wa_merged = merge_expert_family(
        experts=experts,
        base_model=base,
        memory_bank_weights=torch.tensor([0.6, 0.4]),
        policies={"merge_mode": "weighted_average"},
        apply_to=wa_target,
    )
    assert wa_merged, "weighted_average returned empty delta"
    changed = any(
        not torch.allclose(p_t, p_b)
        for (n_t, p_t), (n_b, p_b) in zip(
            wa_target.named_parameters(), base.named_parameters())
    )
    assert changed, "weighted_average merge changed nothing"
    try:
        merge_expert_family(
            experts=experts, base_model=base,
            memory_bank_weights=torch.tensor([0.6, 0.4]),
            policies={"merge_mode": "bogus"})
        raise AssertionError("bogus merge_mode accepted")
    except ValueError:
        pass
    print("13. merge_mode weighted_average + validation OK")

    # 14. Sparse pointwise dispatch matches dense reference (token routing).
    class _AlternatingRouter(nn.Module):
        def forward(self, hidden_states, difficulty_metrics=None):
            B, L, _ = hidden_states.shape
            logits = torch.full((B, L, 4), -30.0)
            logits[:, 0::2, 2] = 30.0   # even tokens -> Trans-ExFusion
            logits[:, 1::2, 3] = 30.0   # odd tokens  -> Cheap
            return logits

    sparse_cfg = copy.deepcopy(attention_config)
    sparse_cfg.routing_granularity = "token"
    sparse_layer = DAPHHybridDecoderLayer(sparse_cfg).eval()
    sparse_layer.macro_router = _AlternatingRouter()
    x_sp = torch.randn(2, 5, 32)
    out_sparse, meta_sp = sparse_layer(x_sp)
    sel = meta_sp["selected_paths"]
    full_trans = sparse_layer.trans_exfusion(x_sp, meta_sp["difficulty"])
    full_cheap = sparse_layer.cheap_path(x_sp)
    ref = torch.where((sel == 2).unsqueeze(-1), full_trans, torch.zeros_like(x_sp))
    ref = torch.where((sel == 3).unsqueeze(-1), full_cheap, ref)
    assert torch.allclose(out_sparse, sparse_layer.final_norm(ref), atol=1e-5)
    assert (sel == 2).any() and (sel == 3).any()
    print("14. sparse pointwise dispatch == dense reference OK")

    # 15. Attention sinks anchored in sliding-window cache trim.
    sink_cfg = copy.deepcopy(attention_config)
    sink_cfg.attn_history_window = 4
    sink_cfg.attn_sink_tokens = 2
    sink_layer = DAPHHybridDecoderLayer(sink_cfg).eval()
    sink_layer.macro_router = _ForcedRouter(4, 0, "token")
    hist = None
    pad = None
    first_token = torch.randn(2, 1, 32)
    with torch.no_grad():
        _, m = sink_layer(first_token, use_cache=True)
        hist, pad = m["attn_state"], m["attn_padding_state"]
        for _ in range(7):
            _, m = sink_layer(torch.randn(2, 1, 32), use_cache=True,
                              attn_state=hist, attn_padding_state=pad)
            hist, pad = m["attn_state"], m["attn_padding_state"]
    assert hist.shape[1] == 4
    assert torch.allclose(hist[:, 0], first_token.squeeze(1)), \
        "attention sink token was evicted!"
    assert pad.shape[1] == 4
    print("15. attention-sink anchoring in window trim OK")

    # 16. Bypass decay: 0.0 exact, >0 discharges state norm.
    ssm_dec = SelectiveSSM(32, 8)
    x_dec = torch.randn(1, 6, 32)
    st0 = torch.randn(1, 32, 8)
    zero_mask = torch.zeros(1, 6)
    _, st_exact = ssm_dec(x_dec, state=st0, mask=zero_mask, bypass_decay=0.0)
    assert torch.allclose(st_exact, st0), "decay=0 must preserve state exactly"
    _, st_dec = ssm_dec(x_dec, state=st0, mask=zero_mask, bypass_decay=0.5)
    assert not torch.allclose(st_dec, st0), "decay=0.5 had no effect"
    print("16. bypass decay opt-in (exact at 0.0, active at 0.5) OK")

    # 17. Micro-batched Fisher: same keys, finite, rigorous default preserved.
    calib17 = torch.randn(6, 32)
    lin17 = nn.Linear(32, 4)
    f_exact = build_empirical_fisher_diagonals(lin17, calib17)
    f_micro = build_empirical_fisher_diagonals(lin17, calib17,
                                               micro_batch_size=3)
    assert set(f_exact) == set(f_micro)
    assert all(torch.isfinite(v).all() for v in f_micro.values())
    # rigorous path is exact per-sample E[g^2]: recompute manually
    manual = {n: torch.zeros_like(p) for n, p in lin17.named_parameters()}
    for i in range(6):
        lin17.zero_grad()
        out = lin17(calib17[i:i + 1])
        loss = F.cross_entropy(out, out.detach().argmax(-1))
        loss.backward()
        for n, p in lin17.named_parameters():
            manual[n] += p.grad.square() / 6
    for n in f_exact:
        assert torch.allclose(f_exact[n], manual[n], atol=1e-6)
    print("17. micro-batch Fisher (approx flagged, default rigorous) OK")

    # 18. Scan dispatch registry: custom backend invoked, bad name raises.
    calls18 = {"n": 0}
    def _spy_backend(*args):
        calls18["n"] += 1
        return _selective_scan_impl(*args)
    register_scan_backend("spy", _spy_backend)
    os.environ["DAPH_SCAN_BACKEND"] = "spy"
    try:
        ssm18 = SelectiveSSM(32, 8)
        ssm18(torch.randn(1, 4, 32))
        assert calls18["n"] == 1
        os.environ["DAPH_SCAN_BACKEND"] = "nonexistent"
        try:
            ssm18(torch.randn(1, 4, 32))
            raise AssertionError("unregistered backend accepted")
        except ValueError:
            pass
    finally:
        os.environ.pop("DAPH_SCAN_BACKEND", None)
    print("18. pluggable scan dispatch (registry + validation) OK")

    # 19. TIES v2 pure sign-majority: a heavy-magnitude outlier expert cannot
    # overturn a majority of weak-magnitude experts of the opposite sign.
    majority_vectors = [
        {"weight": torch.tensor([10.0])},   # heavy outlier, positive
        {"weight": torch.tensor([-1.0])},
        {"weight": torch.tensor([-1.0])},
    ]
    majority_result = difficulty_weighted_ties_merge(
        majority_vectors,
        torch.tensor([0.0, 0.0, 0.0]),
        trim_ratio=0.0,
        ssm_soft_merge=False,
    )
    # The magnitude-weighted election (the old bug) would elect + and yield
    # 10.0; pure sign-majority elects - and averages the two agreeing experts.
    assert torch.allclose(majority_result["weight"], torch.tensor([-1.0]))
    print("19. TIES v2 pure sign-majority election (outlier cannot dominate) OK")

    print("All DAPH ExFusion Hybrid v2.3 self-tests passed")


if __name__ == "__main__":
    run_self_test()
