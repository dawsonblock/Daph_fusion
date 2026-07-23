"""
Pareto Frontier Tracking & Objective Multi-Selection (Phase 18).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def is_pareto_dominated(candidate_obj: List[float], other_obj: List[float]) -> bool:
    # Maximization assumed for all objectives in list
    better_or_equal = all(c <= o for c, o in zip(candidate_obj, other_obj))
    strictly_worse = any(c < o for c, o in zip(candidate_obj, other_obj))
    return better_or_equal and strictly_worse


class ParetoFrontier:

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def add_candidate(
        self, candidate_hash: str, config: Dict[str, Any], objectives: List[float]
    ) -> None:
        new_entry = {
            "candidate_hash": candidate_hash,
            "config": config,
            "objectives": objectives,
        }

        # Check if new entry is dominated by existing
        for existing in self.entries:
            if is_pareto_dominated(objectives, existing["objectives"]):
                return

        # Remove existing entries dominated by new
        self.entries = [
            e
            for e in self.entries
            if not is_pareto_dominated(e["objectives"], objectives)
        ]
        self.entries.append(new_entry)

    def select(self, mode: str = "balanced") -> Dict[str, Any]:
        if not self.entries:
            return {}

        if mode == "max_mean":
            return max(
                self.entries, key=lambda e: sum(e["objectives"]) / len(e["objectives"])
            )
        elif mode == "max_min":
            return max(self.entries, key=lambda e: min(e["objectives"]))
        else:  # balanced
            return max(
                self.entries, key=lambda e: sum(e["objectives"]) / len(e["objectives"])
            )
