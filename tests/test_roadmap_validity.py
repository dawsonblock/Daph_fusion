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


# =============================================================================
# Phase 4: Disk-backed / offloaded Fisher accumulation
# =============================================================================


class _TinyLM(nn.Module):
    """Minimal causal-LM-like module: logits [B, L, V]."""

    def __init__(self, vocab: int = 11, hidden: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None):
        return self.head(self.embed(input_ids))


def test_offloaded_fisher_cpu_and_mmap_agree(tmp_path) -> None:
    from daph_exfusion.geometry.curvature import (
        build_empirical_fisher_diagonals_offloaded,
    )

    torch.manual_seed(0)
    model = _TinyLM()
    batch = {
        "input_ids": torch.randint(0, 11, (4, 6)),
        "attention_mask": torch.ones(4, 6, dtype=torch.long),
    }

    fisher_cpu = build_empirical_fisher_diagonals_offloaded(
        model, batch, micro_batch_size=2, use_mmap=False
    )
    fisher_mmap = build_empirical_fisher_diagonals_offloaded(
        model, batch, micro_batch_size=2, use_mmap=True, mmap_dir=str(tmp_path)
    )

    assert set(fisher_cpu) == set(fisher_mmap)
    for name in fisher_cpu:
        assert fisher_cpu[name].shape == fisher_mmap[name].shape
        assert torch.allclose(fisher_cpu[name], fisher_mmap[name], atol=1e-6)
        assert (fisher_cpu[name] >= 0).all()
    # mmap buffers must be materialized on disk
    assert any(f.suffix == ".bin" for f in tmp_path.iterdir())


# =============================================================================
# Phase 5: AGX layerwise merge operators & search engine
# =============================================================================


def test_apply_layer_merge_operator_raw() -> None:
    from daph_exfusion.search.optimization import apply_layer_merge_operator

    torch.manual_seed(1)
    base = nn.Linear(4, 4)
    expert_a = nn.Linear(4, 4)
    expert_b = nn.Linear(4, 4)
    target = nn.Linear(4, 4)

    apply_layer_merge_operator(
        target_layer=target,
        base_layer=base,
        expert_layers=[expert_a, expert_b],
        operator="RAW",
        lambdas=[0.5, 0.25],
    )

    expected_w = (
        base.weight
        + 0.5 * (expert_a.weight - base.weight)
        + 0.25 * (expert_b.weight - base.weight)
    )
    assert torch.allclose(target.weight, expected_w, atol=1e-6)


def test_layerwise_search_engine_rejects_high_drift() -> None:
    from daph_exfusion.search.candidate import LayerMergeConfig, MergeCandidate
    from daph_exfusion.search.optimization import LayerwiseGeometrySearchEngine

    torch.manual_seed(2)

    class _StackModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

        def forward(self, x):
            hidden_states = [x]
            for layer in self.layers:
                x = torch.tanh(layer(x))
                hidden_states.append(x)
            return hidden_states

    base = _StackModel()
    expert = _StackModel()
    with torch.no_grad():
        for p_b, p_e in zip(base.parameters(), expert.parameters()):
            p_e.copy_(p_b + 50.0)  # pathological expert with massive delta

    val_inputs = torch.randn(6, 4)

    engine = LayerwiseGeometrySearchEngine(
        base_model=base,
        experts=[expert],
        validation_batch={"input_ids": val_inputs},
        max_cka_drift=0.15,
        layer_module_fn=lambda m: m.layers,
        hidden_state_fn=lambda m, b: m(b["input_ids"]),
    )
    assert engine.num_layers == 2

    # Zero-lambda candidate: identical to base, zero drift, feasible.
    identity_candidate = MergeCandidate(
        layer_configs={
            0: LayerMergeConfig(operator="RAW", lambdas=(0.0,)),
            1: LayerMergeConfig(operator="RAW", lambdas=(0.0,)),
        }
    )
    merged = engine.build_candidate_model(identity_candidate)
    drifts = engine.measure_representation_drift(merged)
    assert max(drifts) <= 1e-5

    # Full-strength pathological candidate must violate the CKA safeguard.
    hostile_candidate = MergeCandidate(
        layer_configs={
            0: LayerMergeConfig(operator="RAW", lambdas=(1.0,)),
            1: LayerMergeConfig(operator="RAW", lambdas=(1.0,)),
        }
    )
    merged_hostile = engine.build_candidate_model(hostile_candidate)
    hostile_drifts = engine.measure_representation_drift(merged_hostile)
    assert max(hostile_drifts) > 0.15
