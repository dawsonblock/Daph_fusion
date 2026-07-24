"""Multi-objective Pareto candidate scoring (Phase 13).

AGX should not optimize one scalar prematurely. Track at least:
  R_math, R_planning, R_coding, D_repr, L_base-regression, memory, latency

Hard constraints:
  qualification=pass, CKA drift <= threshold, no NaN, topology valid,
  base regression <= allowed budget

Then maintain a Pareto frontier. Only assign a scalar utility after
filtering feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CandidateObjectives:
    """Raw multi-objective evaluation for a candidate.

    All objectives are MINIMIZED except retention (maximized).
    For Pareto dominance, we convert maximization to minimization
    internally (1 - R for retention).
    """
    retention: Dict[str, float]  # domain -> R_d (maximize)
    repr_drift: float            # D_repr (minimize)
    base_regression: float       # L_base-regression (minimize)
    memory_mb: float = 0.0       # memory (minimize)
    latency_ms: float = 0.0      # latency (minimize)
    feasible: bool = True
    infeasibility_reason: Optional[str] = None

    @property
    def mean_retention(self) -> float:
        valid = [v for v in self.retention.values() if v is not None and np.isfinite(v)]
        return float(np.mean(valid)) if valid else float("-inf")

    @property
    def worst_retention(self) -> float:
        valid = [v for v in self.retention.values() if v is not None and np.isfinite(v)]
        return float(min(valid)) if valid else float("-inf")

    def to_minimization_vector(self) -> np.ndarray:
        """Convert to a vector where all objectives are minimized."""
        ret = [1.0 - v if v is not None and np.isfinite(v) else 1.0
               for v in self.retention.values()]
        return np.array(ret + [self.repr_drift, self.base_regression,
                               self.memory_mb, self.latency_ms])


@dataclass
class CandidateEvaluation:
    candidate_hash: str
    objectives: CandidateObjectives
    config: Optional[dict] = None


def pareto_dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a Pareto-dominates b (a is <= in all dims, < in at least one)."""
    return bool(np.all(a <= b) and np.any(a < b))


def compute_pareto_front(
    evaluations: List[CandidateEvaluation],
) -> List[CandidateEvaluation]:
    """Return the Pareto-optimal subset of evaluations.

    Only FEASIBLE candidates participate; infeasible ones are excluded.
    """
    feasible = [e for e in evaluations if e.objectives.feasible]
    if not feasible:
        return []

    vectors = np.array([e.objectives.to_minimization_vector() for e in feasible])
    n = len(feasible)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if pareto_dominates(vectors[j], vectors[i]):
                is_pareto[i] = False
                break

    return [feasible[i] for i in range(n) if is_pareto[i]]


def scalar_utility(
    obj: CandidateObjectives,
    alpha: float = 0.1,
    beta: float = 0.5,
    gamma: float = 0.01,
) -> float:
    """Scalar utility for ranking after feasibility filtering.

    J = mean_R - alpha * D_repr - beta * D_base - gamma * C_runtime

    Higher is better. Only meaningful for feasible candidates.
    """
    if not obj.feasible:
        return float("-inf")
    runtime_cost = obj.memory_mb + obj.latency_ms
    return (
        obj.mean_retention
        - alpha * obj.repr_drift
        - beta * obj.base_regression
        - gamma * runtime_cost
    )


def rank_candidates(
    evaluations: List[CandidateEvaluation],
    alpha: float = 0.1,
    beta: float = 0.5,
    gamma: float = 0.01,
) -> List[CandidateEvaluation]:
    """Rank candidates: Pareto front first (sorted by utility), then rest."""
    pareto = compute_pareto_front(evaluations)
    pareto_hashes = {e.candidate_hash for e in pareto}
    non_pareto = [e for e in evaluations if e.candidate_hash not in pareto_hashes]

    pareto_sorted = sorted(
        pareto,
        key=lambda e: scalar_utility(e.objectives, alpha, beta, gamma),
        reverse=True,
    )
    non_pareto_sorted = sorted(
        non_pareto,
        key=lambda e: scalar_utility(e.objectives, alpha, beta, gamma),
        reverse=True,
    )
    return pareto_sorted + non_pareto_sorted


class ParetoFrontier:
    """Backward-compatible Pareto frontier class (MAXIMIZATION semantics).

    Maintains a list of non-dominated candidates. When a new candidate
    is added, it removes any existing entries it dominates and is itself
    removed if any existing entry dominates it.

    NOTE: This class uses MAXIMIZATION semantics (higher objectives are
    better), matching the old test_roadmap_validity.py expectations.
    The new function-based API (compute_pareto_front) uses minimization
    semantics internally; this class negates objectives before comparing.
    """

    def __init__(self) -> None:
        self.entries: List[dict] = []

    def add_candidate(
        self,
        candidate_hash: str,
        config: dict,
        objectives: List[float],
    ) -> None:
        """Add a candidate to the frontier (maximization: higher is better)."""
        # Negate for minimization comparison
        new_vec = -np.array(objectives, dtype=float)

        # Remove entries dominated by the new candidate
        self.entries = [
            e for e in self.entries
            if not pareto_dominates(new_vec, -np.array(e["objectives"], dtype=float))
        ]

        # Check if any existing entry dominates the new candidate
        dominated = any(
            pareto_dominates(-np.array(e["objectives"], dtype=float), new_vec)
            for e in self.entries
        )

        if not dominated:
            self.entries.append({
                "candidate_hash": candidate_hash,
                "config": config,
                "objectives": list(objectives),
            })
