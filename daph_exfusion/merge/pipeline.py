"""Canonical merge pipeline (Phase 1 of the experimental-truth repair).

Every merge — experiment runner, AGX candidate, CLI, integration test — must
go through ``merge_experts``. This guarantees that the algorithm name in the
config matches the mathematical operations actually performed, and produces a
``MergeResult`` with an ``operator_trace`` for provenance.

Pipeline order (config.merge_order):

    task vectors → [DARE] → [TIES] → [Fisher] → apply to base

Supported canonical algorithms:
    task_arithmetic       — Σ λᵢ Δᵢ
    mean_merge            — (1/N) Σ λᵢ Δᵢ
    weighted_task_arithmetic — Σ λᵢ Δᵢ (λᵢ are per-expert weights)
    DARE                  — DARE(Δᵢ) then weighted sum
    TIES_MAGNITUDE        — TIES with magnitude sign election
    TIES_MAJORITY         — TIES with pure majority sign election
    DARE_TIES             — DARE → TIES
    FISHER                — Fisher-weighted merge (requires curvature_bank)
    TIES_FISHER           — TIES → Fisher-weighted disjoint merge
    DARE_TIES_FISHER      — DARE → TIES → Fisher-weighted (ExFusion-F)
    ExFusion              — alias for DARE_TIES_FISHER
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from daph_exfusion.geometry.operators import (
    op_dare,
    op_fisher_weighted,
    op_ties,
    op_ties_fisher,
)


@dataclass
class ExpertMergeState:
    """Per-expert state for merging."""
    name: str
    delta: Dict[str, Tensor]
    dare_mask: Optional[Dict[str, Tensor]] = None
    fisher: Optional[Dict[str, Tensor]] = None
    weight: float = 1.0


@dataclass
class MergeConfig:
    """Canonical merge configuration."""
    algorithm: str = "task_arithmetic"
    scale: float = 1.0
    dare_drop_rate: float = 0.0
    ties_trim_fraction: float = 0.2
    ties_sign_mode: str = "magnitude"  # "magnitude" or "majority"
    fisher_gamma: float = 0.5
    lambdas: Tuple[float, ...] = ()
    seed: int = 42

    def __post_init__(self):
        self.algorithm = self.algorithm.upper()

    @property
    def is_stochastic(self) -> bool:
        """True if the merge involves RNG (DARE)."""
        return "DARE" in self.algorithm and self.dare_drop_rate > 0


@dataclass
class MergeResult:
    """Result of a merge operation with provenance."""
    merged_model: nn.Module
    operator_trace: List[str]
    config: MergeConfig
    algorithm: str

    def to_provenance_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "operator_trace": self.operator_trace,
            "scale": self.config.scale,
            "dare_drop_rate": self.config.dare_drop_rate,
            "ties_trim_fraction": self.config.ties_trim_fraction,
            "ties_sign_mode": self.config.ties_sign_mode,
            "fisher_gamma": self.config.fisher_gamma,
            "lambdas": list(self.config.lambdas),
            "seed": self.config.seed,
            "is_stochastic": self.config.is_stochastic,
        }


def _extract_task_vectors(
    experts: Sequence[nn.Module], base: nn.Module
) -> List[Dict[str, Tensor]]:
    """Extract task vectors Δᵢ = θᵢ - θ₀ for each expert."""
    base_params = dict(base.named_parameters())
    task_vectors = []
    for expert in experts:
        tv = {}
        for name, param in expert.named_parameters():
            if name in base_params:
                tv[name] = (param.detach().float() - base_params[name].detach().float())
        task_vectors.append(tv)
    return task_vectors


def _make_persistent_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """Create a persistent generator that is consumed continuously (not reset)."""
    return torch.Generator(device=device).manual_seed(seed)


def merge_experts(
    base_model: nn.Module,
    experts: Sequence[nn.Module],
    config: MergeConfig,
    curvature_bank: Optional[Dict[str, Dict[str, Tensor]]] = None,
    device: str = "cpu",
) -> MergeResult:
    """Merge experts into a single model using the canonical pipeline.

    Args:
        base_model: The base model (θ₀).
        experts: List of specialist models.
        config: Merge configuration (algorithm, hyperparameters).
        curvature_bank: Optional dict mapping expert_name → {param_name: Fisher diagonal}.
                        Required for FISHER, TIES_FISHER, DARE_TIES_FISHER, ExFusion.
        device: Device for computation.

    Returns:
        MergeResult with the merged model and operator trace.
    """
    algo = config.algorithm
    n_experts = len(experts)
    lambdas = list(config.lambdas) if config.lambdas else [1.0] * n_experts
    scale = config.scale

    # Extract task vectors
    base_model_cpu = base_model.cpu()
    for e in experts:
        e.cpu()
    task_vectors = _extract_task_vectors(experts, base_model_cpu)

    # Determine operator trace
    trace: List[str] = []
    needs_fisher = algo in ("FISHER", "TIES_FISHER", "DARE_TIES_FISHER", "EXFUSION")
    if needs_fisher and curvature_bank is None:
        raise ValueError(
            f"Algorithm '{algo}' requires curvature_bank (empirical Fisher diagonals). "
            f"Pass curvature_bank={{expert_name: {{param_name: Fisher_tensor}}}}."
        )

    # Expert names for Fisher lookup
    expert_names = [f"expert_{i}" for i in range(n_experts)]

    # Create merged model as a copy of base
    merged = copy.deepcopy(base_model_cpu)
    merged_params = dict(merged.named_parameters())

    # Persistent generator for DARE (consumed continuously, not reset per-param)
    generator = _make_persistent_generator(config.seed, device="cpu")

    with torch.no_grad():
        for name, param in merged_params.items():
            deltas = []
            fishers = []
            for i, tv in enumerate(task_vectors):
                if name in tv:
                    deltas.append(tv[name] * lambdas[i])
                    if needs_fisher and curvature_bank is not None:
                        expert_key = expert_names[i]
                        if expert_key in curvature_bank and name in curvature_bank[expert_key]:
                            fishers.append(curvature_bank[expert_key][name].float())
                        else:
                            fishers.append(torch.ones_like(tv[name]))
                else:
                    deltas.append(None)

            valid_deltas = [d for d in deltas if d is not None]
            if not valid_deltas:
                continue

            # Apply the canonical algorithm
            if algo == "TASK_ARITHMETIC":
                trace = ["TASK_ARITHMETIC"]
                merged_delta = sum(valid_deltas)

            elif algo == "MEAN_MERGE":
                trace = ["MEAN_MERGE"]
                merged_delta = sum(valid_deltas) / len(valid_deltas)

            elif algo == "WEIGHTED_TASK_ARITHMETIC":
                trace = ["WEIGHTED_TASK_ARITHMETIC"]
                merged_delta = sum(valid_deltas)

            elif algo == "DARE":
                trace = ["DARE"]
                dare_deltas = [
                    op_dare(d, drop_probability=config.dare_drop_rate, generator=generator)
                    for d in valid_deltas
                ]
                merged_delta = sum(dare_deltas)

            elif algo in ("TIES", "TIES_MAGNITUDE"):
                trace = ["TIES_MAGNITUDE"]
                merged_delta = op_ties(
                    valid_deltas,
                    trim_fraction=config.ties_trim_fraction,
                    sign_mode="magnitude",
                )

            elif algo == "TIES_MAJORITY":
                trace = ["TIES_MAJORITY"]
                merged_delta = op_ties(
                    valid_deltas,
                    trim_fraction=config.ties_trim_fraction,
                    sign_mode="majority",
                )

            elif algo == "DARE_TIES":
                trace = ["DARE", "TIES_" + config.ties_sign_mode.upper()]
                dare_deltas = [
                    op_dare(d, drop_probability=config.dare_drop_rate, generator=generator)
                    for d in valid_deltas
                ]
                merged_delta = op_ties(
                    dare_deltas,
                    trim_fraction=config.ties_trim_fraction,
                    sign_mode=config.ties_sign_mode,
                )

            elif algo == "FISHER":
                trace = ["EMPIRICAL_FISHER"]
                merged_delta = op_fisher_weighted(
                    valid_deltas, fishers, gamma=config.fisher_gamma
                )

            elif algo == "TIES_FISHER":
                trace = ["TIES_" + config.ties_sign_mode.upper(), "EMPIRICAL_FISHER"]
                merged_delta = op_ties_fisher(
                    valid_deltas, fishers,
                    trim_fraction=config.ties_trim_fraction,
                    fisher_gamma=config.fisher_gamma,
                    sign_mode=config.ties_sign_mode,
                )

            elif algo in ("DARE_TIES_FISHER", "EXFUSION"):
                trace = ["DARE", "TIES_" + config.ties_sign_mode.upper(), "EMPIRICAL_FISHER"]
                dare_deltas = [
                    op_dare(d, drop_probability=config.dare_drop_rate, generator=generator)
                    for d in valid_deltas
                ]
                merged_delta = op_ties_fisher(
                    dare_deltas, fishers,
                    trim_fraction=config.ties_trim_fraction,
                    fisher_gamma=config.fisher_gamma,
                    sign_mode=config.ties_sign_mode,
                )

            else:
                raise ValueError(f"Unknown merge algorithm: '{algo}'")

            param.copy_(param.detach().float() + merged_delta * scale)

    merged.to(device)
    return MergeResult(
        merged_model=merged,
        operator_trace=trace,
        config=config,
        algorithm=algo,
    )
