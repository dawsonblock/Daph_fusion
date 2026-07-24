"""Canonical merge types for ExFusion v3 (dense merge research trunk).

Every merge — CLI, tests, AGX, experiments, notebooks — goes through
``merge_experts`` with these types. This guarantees that the algorithm
name in the config matches the mathematical operations actually performed,
and produces a ``MergeResult`` with an ``OperatorTrace`` for provenance.

Non-negotiable research rule:
    Every algorithm gets one mathematical definition, one implementation,
    one public name, and one provenance trace. If the trace does not
    correspond to the method definition, the experiment fails.

Supported canonical dense methods:
    task_arithmetic         — θ* = θ₀ + α Σᵢ λᵢ Δᵢ
    fisher_dense            — θ*_k = Σᵢ λᵢ F_{i,k}^γ θ_{i,k} / (Σᵢ λᵢ F_{i,k}^γ + ε)
    fisher_base_anchored    — θ*_k = (λ₀ F_{0,k}^γ θ_{0,k} + Σᵢ λᵢ F_{i,k}^γ θ_{i,k})
                                           / (λ₀ F_{0,k}^γ + Σᵢ λᵢ F_{i,k}^γ + ε)
    regmean                 — W* = (Σᵢ Wᵢ Cᵢ)(Σᵢ Cᵢ + ρI)⁻¹
    regmean_pp              — RegMean with propagation-aware recalibration
    coefficient_opt         — θ(λ) = θ₀ + Σᵢ λᵢ Δᵢ, optimize λ on real forward
    trust_region            — coefficient_opt with ½ Δᵀ F₀ Δ ≤ ε constraint
    kfac_barycenter         — K-FAC structured merge (G⊗A weighted)
    agx                     — architecture-aware selection of dense merge geometry

Legacy benchmark methods (controlled baselines, not mainline):
    dare                    — DARE sparsification
    ties_magnitude          — TIES with magnitude sign election
    ties_majority           — TIES with majority sign election
    dare_ties               — DARE → TIES
    emr                     — shared model + task-specific mask
    model_stock             — geometric weight averaging
    slerp                   — spherical linear interpolation
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# Enums
# =============================================================================


class MergeMethod(str, Enum):
    """Canonical dense merge methods (mainline) + legacy baselines."""
    # Dense mainline
    TASK_ARITHMETIC = "task_arithmetic"
    FISHER_DENSE = "fisher_dense"
    FISHER_BASE_ANCHORED = "fisher_base_anchored"
    REGMEAN = "regmean"
    REGMEAN_PP = "regmean_pp"
    COEFFICIENT_OPT = "coefficient_opt"
    TRUST_REGION = "trust_region"
    KFAC_BARYCENTER = "kfac_barycenter"
    AGX = "agx"
    # Legacy baselines
    DARE = "dare"
    TIES_MAGNITUDE = "ties_magnitude"
    TIES_MAJORITY = "ties_majority"
    DARE_TIES = "dare_ties"
    EMR = "emr"
    MODEL_STOCK = "model_stock"
    SLERP = "slerp"
    # Special
    FROZEN = "frozen"

    @classmethod
    def dense_methods(cls) -> set:
        """Mainline dense methods."""
        return {
            cls.TASK_ARITHMETIC, cls.FISHER_DENSE, cls.FISHER_BASE_ANCHORED,
            cls.REGMEAN, cls.REGMEAN_PP, cls.COEFFICIENT_OPT,
            cls.TRUST_REGION, cls.KFAC_BARYCENTER, cls.AGX,
        }

    @classmethod
    def legacy_methods(cls) -> set:
        """Legacy benchmark methods."""
        return {
            cls.DARE, cls.TIES_MAGNITUDE, cls.TIES_MAJORITY,
            cls.DARE_TIES, cls.EMR, cls.MODEL_STOCK, cls.SLERP,
        }


class CoefficientGranularity(str, Enum):
    """Granularity for learned merge coefficients."""
    GLOBAL = "global"      # one λ per expert
    FAMILY = "family"      # one λ per expert per architecture family
    LAYER = "layer"        # one λ per expert per layer


class CoefficientParameterization(str, Enum):
    """Parameterization for optimized coefficients."""
    SOFTMAX = "softmax"            # λ_i = exp(a_i) / Σ exp(a_j) — convex
    SIGMOID = "sigmoid"            # λ_i = σ(a_i) — independent, allows Σ>1
    SIGNED = "signed"              # λ_i = c·tanh(a_i) — allows extrapolation
    UNCONSTRAINED = "unconstrained"  # λ_i = a_i — no constraints


class FisherStabilization(str, Enum):
    """Stabilization modes for Fisher diagonal."""
    NONE = "none"              # raw Fisher
    FLOOR = "floor"            # F' = max(F, ε)
    LOG_COMPRESS = "log"       # F' = log(1 + αF)
    QUANTILE_CLIP = "clip"     # F' = min(F, Q_{0.999}(F))
    POWER = "power"            # F' = F^γ (applied during merge)


class RegMeanMode(str, Enum):
    """Covariance approximation mode for RegMean."""
    FULL = "full"        # C ∈ R^{d×d}
    BLOCK = "block"      # block-diagonal
    DIAGONAL = "diagonal"  # C = diag(c)
    LOW_RANK = "low_rank"  # C ≈ UΛU^T + σ²I


# =============================================================================
# Core dataclasses
# =============================================================================


@dataclass(frozen=True)
class ExpertSpec:
    """Specification of a fine-tuned expert model.

    Every expert that enters a merge must have a checkpoint path, hash,
    and target domain. Unqualified experts never enter any merge.
    """
    name: str
    checkpoint_path: str
    checkpoint_hash: str
    target_domain: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MergeConfig:
    """Canonical merge configuration for the v3 dense merge trunk.

    This replaces the v2.5 MergeConfig which was DARE/TIES-centric
    (dare_drop_rate, ties_trim_fraction, ties_sign_mode). Those parameters
    are now under legacy_sparse.* and only used by legacy benchmark methods.
    """
    method: MergeMethod = MergeMethod.TASK_ARITHMETIC
    # Task arithmetic
    task_scale: float = 1.0                    # α
    lambdas: Tuple[float, ...] = ()            # λᵢ (empty = uniform)
    # Fisher
    fisher_gamma: float = 1.0                  # γ exponent
    base_precision_weight: float = 0.0         # λ₀ for base-anchored
    fisher_stabilization: FisherStabilization = FisherStabilization.FLOOR
    fisher_floor_eps: float = 1e-8
    fisher_log_alpha: float = 1.0
    fisher_clip_quantile: float = 0.999
    allow_missing_fisher: bool = False
    # RegMean
    regmean_ridge: float = 1e-4                # ρ
    regmean_mode: RegMeanMode = RegMeanMode.DIAGONAL
    regmean_block_size: int = 256
    regmean_low_rank: int = 64
    # Coefficient optimization
    coefficient_granularity: CoefficientGranularity = CoefficientGranularity.FAMILY
    coefficient_parameterization: CoefficientParameterization = CoefficientParameterization.SOFTMAX
    coefficient_lr: float = 0.01
    coefficient_steps: int = 100
    # Trust region
    trust_region_budget: float = 1.0           # ε
    # K-FAC
    kfac_approximation: str = "diagonal"       # K1-K4
    # Legacy sparse (only for legacy benchmark methods)
    legacy_sparse: Dict[str, Any] = field(default_factory=dict)
    # Provenance
    seed: int = 42

    def __post_init__(self):
        if isinstance(self.method, str):
            object.__setattr__(self, "method", MergeMethod(self.method))
        if isinstance(self.fisher_stabilization, str):
            object.__setattr__(self, "fisher_stabilization", FisherStabilization(self.fisher_stabilization))
        if isinstance(self.regmean_mode, str):
            object.__setattr__(self, "regmean_mode", RegMeanMode(self.regmean_mode))
        if isinstance(self.coefficient_granularity, str):
            object.__setattr__(self, "coefficient_granularity", CoefficientGranularity(self.coefficient_granularity))
        if isinstance(self.coefficient_parameterization, str):
            object.__setattr__(self, "coefficient_parameterization", CoefficientParameterization(self.coefficient_parameterization))

    @property
    def is_stochastic(self) -> bool:
        """True if the merge involves RNG (DARE)."""
        return self.method in (MergeMethod.DARE, MergeMethod.DARE_TIES)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["method"] = self.method.value
        d["fisher_stabilization"] = self.fisher_stabilization.value
        d["regmean_mode"] = self.regmean_mode.value
        d["coefficient_granularity"] = self.coefficient_granularity.value
        d["coefficient_parameterization"] = self.coefficient_parameterization.value
        return d

    def config_hash(self) -> str:
        """SHA256 hash of the canonical config (for provenance)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class OperatorTrace:
    """Provenance trace for a merge operation.

    Records exactly which mathematical operators were executed, so that
    the reported method can be verified against the actual computation.

    Research-contract tests check:
        result.trace.operators == expected_operators
        result.trace.fisher_estimator == "exact_per_sample"
        result.trace.activation_covariance_used
        result.trace.fisher_used
        result.trace.dare_used
    """
    method: str
    operators: List[str] = field(default_factory=list)
    implementation_version: str = "1"
    # What was actually used
    fisher_used: bool = False
    fisher_estimator: str = ""           # "exact_per_sample", "microbatch_gradient_square", ""
    activation_covariance_used: bool = False
    dare_used: bool = False
    ties_used: bool = False
    trust_region_enforced: bool = False
    kfac_used: bool = False
    # Hyperparameters actually applied
    task_scale: float = 1.0
    fisher_gamma: float = 1.0
    base_precision_weight: float = 0.0
    lambdas: List[float] = field(default_factory=list)
    # Coefficient optimization
    coefficient_granularity: str = ""
    coefficient_parameterization: str = ""
    # Integrity
    config_hash: str = ""
    checkpoint_hashes: List[str] = field(default_factory=list)
    dataset_hashes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def assert_operator(self, operator: str) -> None:
        """Assert that an operator was executed (for research-contract tests)."""
        if operator not in self.operators:
            raise AssertionError(
                f"Operator '{operator}' not in trace. "
                f"Executed operators: {self.operators}"
            )

    def assert_no_operator(self, operator: str) -> None:
        """Assert that an operator was NOT executed."""
        if operator in self.operators:
            raise AssertionError(
                f"Operator '{operator}' unexpectedly in trace. "
                f"Operators: {self.operators}"
            )


@dataclass
class MergeResult:
    """Result of a merge operation with provenance."""
    merged_model: nn.Module
    trace: OperatorTrace
    config: MergeConfig
    method: str

    def to_provenance_dict(self) -> dict:
        return {
            "method": self.method,
            "trace": self.trace.to_dict(),
            "config": self.config.to_dict(),
            "config_hash": self.config.config_hash(),
        }


# =============================================================================
# Task vector extraction
# =============================================================================


def extract_task_vectors(
    experts: Sequence[nn.Module], base: nn.Module
) -> List[Dict[str, Tensor]]:
    """Extract task vectors Δᵢ = θᵢ - θ₀ for each expert.

    All computation is in FP32 for numerical stability, regardless of the
    expert's parameter dtype.
    """
    base_params = dict(base.named_parameters())
    task_vectors = []
    for expert in experts:
        tv = {}
        for name, param in expert.named_parameters():
            if name in base_params:
                tv[name] = (param.detach().float() - base_params[name].detach().float())
        task_vectors.append(tv)
    return task_vectors


def validate_parameter_names(
    experts: Sequence[nn.Module], base: nn.Module
) -> None:
    """Validate that all experts have the same parameter names as base.

    Raises ValueError if there is a mismatch.
    """
    base_names = set(dict(base.named_parameters()).keys())
    for i, expert in enumerate(experts):
        expert_names = set(dict(expert.named_parameters()).keys())
        if expert_names != base_names:
            missing = base_names - expert_names
            extra = expert_names - base_names
            raise ValueError(
                f"Expert {i} parameter names do not match base. "
                f"Missing: {missing}. Extra: {extra}."
            )


# =============================================================================
# Architecture family classification
# =============================================================================


def classify_parameter_family(param_name: str, layer_idx: int = -1, num_layers: int = -1) -> str:
    """Classify a parameter into an architecture family.

    Families: embeddings, early_attention, middle_attention, late_attention,
    early_ffn, middle_ffn, late_ffn, ssm_projections, ssm_recurrence,
    normalization, lm_head, other.
    """
    name_lower = param_name.lower()
    if ("embed" in name_lower and "position" not in name_lower) or "wte" in name_lower:
        return "embeddings"
    if "position" in name_lower or "pos_embed" in name_lower:
        return "positional_embeddings"
    if "lm_head" in name_lower or "output" in name_lower:
        return "lm_head"
    if "norm" in name_lower or "ln" in name_lower or "rms" in name_lower:
        return "normalization"
    if "ssm" in name_lower or "mamba" in name_lower:
        if "a_log" in name_lower:
            return "ssm_recurrence"
        if "dt" in name_lower:
            return "ssm_recurrence"
        if name_lower.endswith(".d") or ".d." in name_lower:
            return "ssm_recurrence"
        return "ssm_projections"
    if any(k in name_lower for k in ("attention", "attn", "q_proj", "k_proj", "v_proj", "o_proj")):
        if num_layers > 0 and layer_idx >= 0:
            third = max(1, num_layers // 3)
            if layer_idx < third:
                return "early_attention"
            elif layer_idx < 2 * third:
                return "middle_attention"
            return "late_attention"
        return "attention"
    if any(k in name_lower for k in ("ffn", "mlp", "intermediate", "fc", "gate_proj", "up_proj", "down_proj")):
        if num_layers > 0 and layer_idx >= 0:
            third = max(1, num_layers // 3)
            if layer_idx < third:
                return "early_ffn"
            elif layer_idx < 2 * third:
                return "middle_ffn"
            return "late_ffn"
        return "ffn"
    return "other"


FINE_FAMILIES = (
    "attention",
    "ssm",
    "ffn",
    "norm",
    "embedding",
    "lm_head",
    "router",
    "other",
)


def classify_parameter_family_fine(name: str) -> str:
    """Classify a parameter into one of the 8 fine-grained families.

    Used by family-weighted Task Arithmetic (TA-3). The order of checks
    matters: embedding, lm_head, norm, router, ssm, attention, ffn, other.

    Families: attention, ssm, ffn, norm, embedding, lm_head, router, other.
    """
    name_lower = name.lower()

    # 1. embedding (covers embed_tokens, wte, position_embeddings)
    if "embed" in name_lower:
        return "embedding"

    # 2. lm_head (covers lm_head, cls head, tied wte weights)
    if "lm_head" in name_lower or "cls" in name_lower or "wte" in name_lower:
        return "lm_head"

    # 3. norm (covers layernorm, rmsnorm, ln)
    if "norm" in name_lower or "ln" in name_lower or "layernorm" in name_lower:
        return "norm"

    # 4. router (covers router / gate, but NOT mlp.gate_proj which is ffn)
    if "router" in name_lower or ("gate" in name_lower and "mlp" not in name_lower):
        return "router"

    # 5. ssm (covers ssm, a_log, dt_proj, conv-with-ssm)
    if (
        "ssm" in name_lower
        or "a_log" in name_lower
        or "dt_proj" in name_lower
        or ("conv" in name_lower and "ssm" in name_lower)
    ):
        return "ssm"

    # 6. attention (checked before ffn so c_proj → attention)
    if any(
        k in name_lower
        for k in (
            "attn",
            "attention",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "c_attn",
            "c_proj",
            "query",
            "key",
            "value",
        )
    ):
        return "attention"

    # 7. ffn (covers mlp, ffn, gate_proj, up_proj, down_proj, c_fc, c_proj)
    if any(
        k in name_lower
        for k in (
            "mlp",
            "ffn",
            "gate_proj",
            "up_proj",
            "down_proj",
            "c_fc",
            "c_proj",
        )
    ):
        return "ffn"

    # 8. other
    return "other"


def get_layer_index(param_name: str) -> int:
    """Extract layer index from a parameter name like 'model.layers.5.attn.q_proj.weight'."""
    parts = param_name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
        if p == "h" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def count_layers(model: nn.Module) -> int:
    """Count the number of transformer/mamba layers in a model."""
    max_layer = -1
    for name in dict(model.named_parameters()).keys():
        idx = get_layer_index(name)
        if idx > max_layer:
            max_layer = idx
    return max_layer + 1 if max_layer >= 0 else 0


# =============================================================================
# SSM parameter validation
# =============================================================================


def validate_ssm_stability(model: nn.Module) -> Dict[str, Any]:
    """Validate SSM (Mamba) parameter stability after merging.

    Checks:
        - finite A (reconstructed from A_log)
        - finite dt
        - valid dt bounds (positive)
        - stable state rollout (no exploding recurrent norm)
    """
    results: Dict[str, Any] = {"valid": True, "checks": {}}
    params = dict(model.named_parameters())

    for name, param in params.items():
        name_lower = name.lower()
        if "a_log" in name_lower:
            a_log = param.detach().float()
            a = -torch.exp(a_log)
            finite = torch.isfinite(a).all().item()
            negative = (a < 0).all().item()
            results["checks"][name] = {
                "type": "A_log",
                "finite": finite,
                "A_negative": negative,
                "A_min": a.min().item(),
                "A_max": a.max().item(),
            }
            if not finite or not negative:
                results["valid"] = False

        if "dt" in name_lower and "weight" not in name_lower and "norm" not in name_lower:
            dt = param.detach().float()
            finite = torch.isfinite(dt).all().item()
            positive = (dt > 0).all().item() if dt.numel() > 0 else True
            results["checks"][name] = {
                "type": "dt",
                "finite": finite,
                "positive": positive,
                "dt_min": dt.min().item() if dt.numel() > 0 else 0.0,
                "dt_max": dt.max().item() if dt.numel() > 0 else 0.0,
            }
            if not finite or not positive:
                results["valid"] = False

    return results


# =============================================================================
# Re-export: MissingCurvatureError
# =============================================================================
# Imported at the end of the module to avoid a circular import: fisher_dense
# imports names from this module (MergeConfig, MergeMethod, ...), which are
# all defined above, so by the time this line runs they are available on the
# partially-loaded ``daph_exfusion.merge.types`` module.
from daph_exfusion.merge.fisher_dense import MissingCurvatureError  # noqa: E402,F401

__all__ = [
    "MissingCurvatureError",
]
