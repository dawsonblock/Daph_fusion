"""Release gates (Phase 27).

Replace handwritten paper_ready. Generate automatically:

    paper_ready = all([
        full_tests_pass,
        checkpoints_verified,
        split_integrity_verified,
        expert_qualification_verified,
        algorithm_trace_verified,
        sample_statistics_verified,
        no_test_leakage,
        results_reproduced,
    ])

Method-specific:
    fisher_verified = (
        fisher_mode == "exact_per_sample"
        and calibration_hash_valid
        and not pseudo_labels
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from daph_exfusion.merge.types import OperatorTrace


@dataclass
class ReleaseGates:
    """Automatic release gate evaluation."""
    full_tests_pass: bool = False
    checkpoints_verified: bool = False
    split_integrity_verified: bool = False
    expert_qualification_verified: bool = False
    algorithm_trace_verified: bool = False
    sample_statistics_verified: bool = False
    no_test_leakage: bool = False
    results_reproduced: bool = False

    # Method-specific
    fisher_verified: bool = False
    agx_verified: bool = False

    @property
    def paper_ready(self) -> bool:
        """All gates must pass for paper_ready=True."""
        return all([
            self.full_tests_pass,
            self.checkpoints_verified,
            self.split_integrity_verified,
            self.expert_qualification_verified,
            self.algorithm_trace_verified,
            self.sample_statistics_verified,
            self.no_test_leakage,
            self.results_reproduced,
        ])

    @property
    def paper_ready_reason(self) -> str:
        """Human-readable reason if not paper_ready."""
        if self.paper_ready:
            return "All release gates passed."
        reasons = []
        if not self.full_tests_pass:
            reasons.append("full_tests_fail")
        if not self.checkpoints_verified:
            reasons.append("checkpoints_not_verified")
        if not self.split_integrity_verified:
            reasons.append("split_integrity_not_verified")
        if not self.expert_qualification_verified:
            reasons.append("expert_qualification_not_verified")
        if not self.algorithm_trace_verified:
            reasons.append("algorithm_trace_not_verified")
        if not self.sample_statistics_verified:
            reasons.append("sample_statistics_not_verified")
        if not self.no_test_leakage:
            reasons.append("test_leakage_detected")
        if not self.results_reproduced:
            reasons.append("results_not_reproduced")
        return "Gates failed: " + ", ".join(reasons)

    def to_dict(self) -> dict:
        return {
            "paper_ready": self.paper_ready,
            "paper_ready_reason": self.paper_ready_reason,
            "fisher_verified": self.fisher_verified,
            "agx_verified": self.agx_verified,
            "gates": {
                "full_tests_pass": self.full_tests_pass,
                "checkpoints_verified": self.checkpoints_verified,
                "split_integrity_verified": self.split_integrity_verified,
                "expert_qualification_verified": self.expert_qualification_verified,
                "algorithm_trace_verified": self.algorithm_trace_verified,
                "sample_statistics_verified": self.sample_statistics_verified,
                "no_test_leakage": self.no_test_leakage,
                "results_reproduced": self.results_reproduced,
            },
        }


def verify_fisher(trace: OperatorTrace, calibration_hash_valid: bool, pseudo_labels: bool) -> bool:
    """Verify that the Fisher computation is rigorous.

    fisher_verified = (
        fisher_mode == "exact_per_sample"
        and calibration_hash_valid
        and not pseudo_labels
    )
    """
    return (
        trace.fisher_used
        and trace.fisher_estimator == "exact_per_sample"
        and calibration_hash_valid
        and not pseudo_labels
    )


def verify_agx(trace: OperatorTrace, selected_matches_canonical: bool) -> bool:
    """Verify that AGX selected operator matches the canonical operator output."""
    return (
        "AGX" in trace.operators
        and selected_matches_canonical
    )


def verify_algorithm_trace(trace: OperatorTrace, expected_method: str) -> bool:
    """Verify that the operator trace corresponds to the reported method.

    Research-contract: if the trace does not correspond to the method
    definition, the experiment fails.
    """
    if trace.method != expected_method:
        return False

    # Check that the expected operators are present
    method_operator_map = {
        "task_arithmetic": ["TASK_ARITHMETIC"],
        "fisher_dense": ["EMPIRICAL_FISHER", "DENSE_PRECISION_MERGE"],
        "fisher_base_anchored": ["EMPIRICAL_FISHER", "BASE_ANCHOR", "DENSE_PRECISION_MERGE"],
        "regmean": ["REGMEAN"],
        "regmean_pp": ["REGMEAN_PP", "PROPAGATION_RECALIBRATION"],
        "coefficient_opt": ["COEFFICIENT_OPT"],
        "trust_region": ["TRUST_REGION", "FISHER_INTERACTION_MATRIX"],
        "kfac_barycenter": ["KFAC_MERGE"],
        "agx": ["AGX"],
        "dare": ["DARE"],
        "ties_magnitude": ["TIES_MAGNITUDE"],
        "ties_majority": ["TIES_MAJORITY"],
        "dare_ties": ["DARE", "TIES_MAGNITUDE"],
    }

    expected_ops = method_operator_map.get(expected_method, [])
    for op in expected_ops:
        if op not in trace.operators:
            return False

    return True
