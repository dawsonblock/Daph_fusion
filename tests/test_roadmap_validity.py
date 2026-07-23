"""
Comprehensive Test Suite for Roadmap Validity & AGX Components (Part XI).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from daph_exfusion.geometry.descriptors import (
    compute_cosine_similarity,
    compute_l2_norm,
    compute_sign_conflict_ratio,
)
from daph_exfusion.geometry.hierarchy import compute_global_geometry
from daph_exfusion.search.candidate import LayerMergeConfig, MergeCandidate
from daph_exfusion.search.pareto import ParetoFrontier
from experiments.qualification import ExpertQualificationPipeline, InvalidExperiment
from research_metrics import calculate_retention


def test_expert_qualification_pass() -> None:
    base = nn.Linear(4, 4)
    expert = nn.Linear(4, 4)
    pipe = ExpertQualificationPipeline(base, tokenizer=None)
    # Mocking NLL compute via monkeypatch or sub-calls if needed
    compat_arch, compat_state, reason = pipe.check_topology_compatibility(expert)
    assert compat_arch and compat_state
    assert reason is None


def test_expert_topology_mismatch_rejected() -> None:
    base = nn.Linear(4, 4)
    expert = nn.Linear(8, 8)
    pipe = ExpertQualificationPipeline(base, tokenizer=None)
    compat_arch, compat_state, reason = pipe.check_topology_compatibility(expert)
    assert not (compat_arch and compat_state)
    assert "Shape mismatch" in reason or "Key mismatch" in reason


def test_preflight_fails_closed_on_unqualified_expert() -> None:
    base = nn.Linear(4, 4)
    pipe = ExpertQualificationPipeline(base, tokenizer=None)
    from experiments.qualification import ExpertQualification

    q_failed = ExpertQualification(
        expert_name="bad_expert",
        expert_revision="v1",
        domain="email",
        base_nll=5.0,
        expert_nll=6.0,
        relative_improvement=-0.2,
        architecture_compatible=True,
        tokenizer_compatible=True,
        state_dict_compatible=True,
        passed=False,
        rejection_reason="Failed relative improvement threshold.",
    )
    with pytest.raises(InvalidExperiment):
        pipe.validate_preflight([q_failed])


def test_geometry_descriptor_norms_and_cosine() -> None:
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 2.0, 0.0])
    assert compute_l2_norm(v1) == 1.0
    assert compute_l2_norm(v2) == 2.0
    assert compute_cosine_similarity(v1, v2) == 0.0


def test_sign_conflict_ratio() -> None:
    v1 = torch.tensor([1.0, -1.0, 2.0])
    v2 = torch.tensor([-1.0, -1.0, -2.0])
    # nonzero at 0, 1, 2. signs: (+,-,+), (-,,-,-). conflicts at index 0 and 2 => 2/3
    assert abs(compute_sign_conflict_ratio(v1, v2) - (2.0 / 3.0)) < 1e-5


def test_pareto_dominance() -> None:
    pf = ParetoFrontier()
    pf.add_candidate("h1", {}, [0.8, 0.8])
    pf.add_candidate("h2", {}, [0.5, 0.5])  # Dominated
    assert len(pf.entries) == 1
    assert pf.entries[0]["candidate_hash"] == "h1"


def test_candidate_hash_stable() -> None:
    lcfg = LayerMergeConfig(operator="TIES", lambdas=(0.1, 0.2))
    cand1 = MergeCandidate(layer_configs={0: lcfg})
    cand2 = MergeCandidate(layer_configs={0: lcfg})
    assert cand1.compute_hash() == cand2.compute_hash()
