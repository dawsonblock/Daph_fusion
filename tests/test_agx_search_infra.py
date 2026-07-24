"""Phase 12-18: AGX search infrastructure tests."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from daph_exfusion.experimental.agx.groupwise import (
    GroupCandidate,
    GroupMergeConfig,
    generate_random_group_candidate,
    classify_layer_group,
    DEFAULT_LAYER_GROUPS,
)
from daph_exfusion.experimental.agx.pareto import (
    CandidateEvaluation,
    CandidateObjectives,
    compute_pareto_front,
    rank_candidates,
    scalar_utility,
)
from daph_exfusion.experimental.agx.halving import (
    HalvingStages,
    successive_halving,
)
from daph_exfusion.experimental.agx.surrogate import (
    TreeSurrogatePredictor,
    constrained_expected_improvement,
)
from daph_exfusion.policies.adaptive_policy import (
    VariableNGeometryPolicy,
    AdaptiveGeometryPolicy,
)
from daph_exfusion.experimental.agx.candidate_vocab import CandidateVocabularyRouter


# Phase 12: groupwise search


def test_group_candidate_expands_to_layers():
    groups = {"early": [0, 1], "late": [2, 3]}
    gc = GroupCandidate({
        "early": GroupMergeConfig(operator="DARE", lambdas=(0.1, 0.2)),
        "late": GroupMergeConfig(operator="TIES", lambdas=(0.3, 0.4)),
    })
    layer_candidate = gc.to_layer_candidate(groups)
    assert layer_candidate.layer_configs[0].operator == "DARE"
    assert layer_candidate.layer_configs[1].operator == "DARE"
    assert layer_candidate.layer_configs[2].operator == "TIES"
    assert layer_candidate.layer_configs[3].operator == "TIES"


def test_classify_layer_group():
    assert classify_layer_group("transformer.h.0.attn.q_proj", 0, 6) == "early_attention"
    assert classify_layer_group("transformer.h.5.attn.q_proj", 5, 6) == "late_attention"
    assert classify_layer_group("transformer.wte.weight", 0, 6) == "token_embeddings"
    assert classify_layer_group("lm_head.weight", 0, 6) == "lm_head"


# Phase 13: Pareto


def _make_obj(ret, drift=0.1, reg=0.0, feasible=True):
    return CandidateObjectives(
        retention={"math": ret}, repr_drift=drift, base_regression=reg,
        feasible=feasible,
    )


def test_pareto_front_isolates_non_dominated():
    evals = [
        CandidateEvaluation("a", _make_obj(0.8, drift=0.1)),
        CandidateEvaluation("b", _make_obj(0.9, drift=0.1)),  # dominates a
        CandidateEvaluation("c", _make_obj(0.9, drift=0.05)),  # dominates b
        CandidateEvaluation("d", _make_obj(0.5, drift=0.01)),  # not dominated (low drift)
    ]
    front = compute_pareto_front(evals)
    hashes = {e.candidate_hash for e in front}
    assert "c" in hashes  # dominates b and a
    assert "d" in hashes  # non-dominated (best drift)
    assert "a" not in hashes  # dominated by b and c
    assert "b" not in hashes  # dominated by c


def test_pareto_excludes_infeasible():
    evals = [
        CandidateEvaluation("a", _make_obj(0.9, feasible=True)),
        CandidateEvaluation("b", _make_obj(0.99, feasible=False)),
    ]
    front = compute_pareto_front(evals)
    assert len(front) == 1
    assert front[0].candidate_hash == "a"


def test_rank_candidates_pareto_first():
    evals = [
        CandidateEvaluation("a", _make_obj(0.8)),
        CandidateEvaluation("b", _make_obj(0.9)),
    ]
    ranked = rank_candidates(evals)
    assert ranked[0].candidate_hash == "b"  # higher retention


# Phase 14: successive halving


def test_successive_halving_retains_best():
    candidates = [(f"c{i}", {"id": i}) for i in range(20)]

    def evaluator(config, seeds, batches, subset):
        # Higher id -> better retention
        ret = config["id"] / 20.0
        return CandidateObjectives(
            retention={"math": ret}, repr_drift=0.1, base_regression=0.0,
        )

    result = successive_halving(candidates, evaluator)
    # Best should be the highest-id candidate
    assert result.best is not None
    assert result.best.objectives.mean_retention > 0.5


# Phase 15-16: surrogate


def test_surrogate_uses_y_in_fit():
    """The surrogate MUST use y in fit (unlike DummySurrogatePredictor)."""
    np.random.seed(0)
    X = np.random.randn(50, 4)
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5  # y depends on X
    surrogate = TreeSurrogatePredictor(min_samples_for_acquisition=10)
    diag = surrogate.fit(X, y)
    assert diag.n_samples == 50
    # Should have positive R² (the relationship is learnable)
    assert diag.cv_r2 > 0.3, f"Surrogate didn't learn: R²={diag.cv_r2}"


def test_surrogate_predicts_after_fit():
    np.random.seed(0)
    X = np.random.randn(30, 3)
    y = X.sum(axis=1)
    surrogate = TreeSurrogatePredictor(min_samples_for_acquisition=10)
    surrogate.fit(X, y)
    mean, std = surrogate.predict(X[:5])
    assert mean.shape == (5, 1)
    assert std.shape == (5, 1)
    assert np.all(std > 0)


def test_surrogate_unusable_with_few_samples():
    X = np.random.randn(3, 2)
    y = np.array([1.0, 2.0, 3.0])
    surrogate = TreeSurrogatePredictor(min_samples_for_acquisition=10)
    diag = surrogate.fit(X, y)
    assert not diag.usable
    assert "min" in (diag.reason or "")


def test_constrained_ei():
    np.random.seed(0)
    X = np.random.randn(50, 3)
    y = X[:, 0]
    surrogate = TreeSurrogatePredictor(min_samples_for_acquisition=10)
    surrogate.fit(X, y)
    X_new = np.random.randn(10, 3)
    ei = constrained_expected_improvement(X_new, surrogate, best_y=0.0)
    assert ei.shape == (10,)
    assert np.all(ei >= 0)


# Phase 17: variable-N geometry policy


def test_variable_n_policy_supports_different_n():
    policy = VariableNGeometryPolicy(expert_descriptor_dim=8, hidden_dim=32)
    # N=2
    x2 = torch.randn(2, 8)
    out2 = policy(x2)
    assert out2["lambdas"].shape == (2,)
    assert out2["operator_logits"].shape[0] == 7
    # N=5
    x5 = torch.randn(5, 8)
    out5 = policy(x5)
    assert out5["lambdas"].shape == (5,)


def test_variable_n_policy_is_permutation_invariant():
    """Reordering experts should produce the same aggregated embedding."""
    torch.manual_seed(0)
    policy = VariableNGeometryPolicy(expert_descriptor_dim=8, hidden_dim=32)
    policy.eval()
    x = torch.randn(4, 8)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        out_orig = policy(x)
        out_perm = policy(x[perm])
    # Aggregated embedding should be identical (permutation-invariant)
    assert torch.allclose(
        out_orig["aggregated_embedding"],
        out_perm["aggregated_embedding"],
        atol=1e-4,
    )
    # Operator logits should be identical
    assert torch.allclose(
        out_orig["operator_logits"],
        out_perm["operator_logits"],
        atol=1e-4,
    )


# Phase 18: candidate vocabulary routing


def test_candidate_vocab_reduces_complexity():
    """Projection should be against K candidates, not full vocab V."""
    V, H = 1000, 64
    embed = nn.Embedding(V, H)
    candidates = {"math": [10, 20, 30, 40, 50]}  # K=5
    router = CandidateVocabularyRouter(embed, candidates)
    B, L = 2, 8
    h = torch.randn(B, L, H)
    logits, cand_ids = router.project_against_candidates(h, "math")
    # Should be [B, L, K=5], not [B, L, V=1000]
    assert logits.shape == (B, L, 5)
    assert cand_ids.shape == (5,)


def test_candidate_vocab_solve_no_one_hot():
    V, H = 100, 32
    embed = nn.Embedding(V, H)
    candidates = {"math": [10, 20, 30]}
    router = CandidateVocabularyRouter(embed, candidates)
    h = torch.randn(2, 4, H)
    solved_embeds, solved_ids = router.solve_and_embed(h, "math")
    # Output should be [B, L, H], not [B, L, V]
    assert solved_embeds.shape == (2, 4, H)
    assert solved_ids.shape == (2, 4)
    # Solved IDs should be from the candidate set
    assert all(int(sid) in candidates["math"] for sid in solved_ids.flatten())


def test_subword_sequence_bridge_classification():
    from daph_nesy_v1_0 import SubwordSequenceBridge
    assert SubwordSequenceBridge.backend == "cpu_compatibility"
    assert SubwordSequenceBridge.compile_safe is False
    assert SubwordSequenceBridge.cuda_graph_safe is False
    assert SubwordSequenceBridge.production_default is False
