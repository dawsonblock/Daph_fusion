"""Successive halving candidate staging (Phase 14).

Before implementing fancy Bayesian search, make brute-force evaluation
cheaper by evaluating candidates in stages:

  Stage A: small validation subset, 1 seed, few batches -> retain top 25%
  Stage B: full validation, 1 seed -> retain top 25%
  Stage C: full validation, 3 seeds -> retain Pareto survivors
  Stage D: 5 seeds, test set -> final evaluation

This drastically reduces compute before any surrogate is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from daph_exfusion.search.pareto import (
    CandidateEvaluation,
    CandidateObjectives,
    rank_candidates,
    compute_pareto_front,
)


@dataclass
class HalvingStage:
    """Configuration for successive halving stages."""
    # Stage A: quick screen
    stage_a_subset_fraction: float = 0.1
    stage_a_seeds: int = 1
    stage_a_batches: int = 4
    stage_a_retain_fraction: float = 0.25

    # Stage B: full validation, single seed
    stage_b_seeds: int = 1
    stage_b_retain_fraction: float = 0.25

    # Stage C: multi-seed
    stage_c_seeds: int = 3
    stage_c_retain_pareto: bool = True

    # Stage D: final
    stage_d_seeds: int = 5


@dataclass
class HalvingResult:
    """Result of the successive halving process."""
    stage_a_survivors: List[CandidateEvaluation]
    stage_b_survivors: List[CandidateEvaluation]
    stage_c_survivors: List[CandidateEvaluation]
    final_evaluations: List[CandidateEvaluation]
    best: Optional[CandidateEvaluation]


def successive_halving(
    candidates: List[Tuple[str, dict]],
    evaluator: Callable[[dict, int, Optional[int], Optional[int]], CandidateObjectives],
    stages: Optional[HalvingStages] = None,
) -> HalvingResult:
    """Run successive halving over candidate configurations.

    Args:
        candidates: List of (candidate_hash, config) pairs.
        evaluator: Function(config, seeds, num_batches, subset_fraction)
                   -> CandidateObjectives. If num_batches/subset_fraction
                   is None, use the full validation set.
        stages: Halving stage configuration.

    Returns:
        HalvingResult with survivors from each stage.
    """
    if stages is None:
        stages = HalvingStages()

    # Stage A: quick screen
    stage_a_evals: List[CandidateEvaluation] = []
    for chash, config in candidates:
        obj = evaluator(
            config,
            stages.stage_a_seeds,
            stages.stage_a_batches,
            stages.stage_a_subset_fraction,
        )
        stage_a_evals.append(CandidateEvaluation(candidate_hash=chash, objectives=obj))

    ranked_a = rank_candidates(stage_a_evals)
    retain_a = max(1, int(len(ranked_a) * stages.stage_a_retain_fraction))
    stage_a_survivors = ranked_a[:retain_a]

    # Stage B: full validation, single seed
    stage_b_evals: List[CandidateEvaluation] = []
    for e in stage_a_survivors:
        # Re-evaluate with full validation (need the config)
        config = next(c[1] for c in candidates if c[0] == e.candidate_hash)
        obj = evaluator(config, stages.stage_b_seeds, None, None)
        stage_b_evals.append(CandidateEvaluation(candidate_hash=e.candidate_hash, objectives=obj))

    ranked_b = rank_candidates(stage_b_evals)
    retain_b = max(1, int(len(ranked_b) * stages.stage_b_retain_fraction))
    stage_b_survivors = ranked_b[:retain_b]

    # Stage C: multi-seed, retain Pareto front
    stage_c_evals: List[CandidateEvaluation] = []
    for e in stage_b_survivors:
        config = next(c[1] for c in candidates if c[0] == e.candidate_hash)
        obj = evaluator(config, stages.stage_c_seeds, None, None)
        stage_c_evals.append(CandidateEvaluation(candidate_hash=e.candidate_hash, objectives=obj))

    if stages.stage_c_retain_pareto:
        stage_c_survivors = compute_pareto_front(stage_c_evals)
    else:
        ranked_c = rank_candidates(stage_c_evals)
        stage_c_survivors = ranked_c[:max(1, len(ranked_c) // 2)]

    # Stage D: final evaluation with 5 seeds
    final_evals: List[CandidateEvaluation] = []
    for e in stage_c_survivors:
        config = next(c[1] for c in candidates if c[0] == e.candidate_hash)
        obj = evaluator(config, stages.stage_d_seeds, None, None)
        final_evals.append(CandidateEvaluation(candidate_hash=e.candidate_hash, objectives=obj))

    best = max(final_evals, key=lambda e: e.objectives.mean_retention) if final_evals else None

    return HalvingResult(
        stage_a_survivors=stage_a_survivors,
        stage_b_survivors=stage_b_survivors,
        stage_c_survivors=stage_c_survivors,
        final_evaluations=final_evals,
        best=best,
    )
