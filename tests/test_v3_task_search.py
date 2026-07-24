"""Tests for the Task Arithmetic grid search (task_search.py).

Verifies:
    - Simplex grid generation correctness
    - TA-0 (uniform) search
    - TA-1 (weighted) search with constrained objective
    - TA-2 (Fisher-weighted) search
    - TA-3 (family-weighted) search
    - Constraint enforcement (R_min ≥ τ, G_regression ≤ δ)
    - Refinement around best
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from daph_exfusion.merge.types import MergeConfig, MergeMethod
from daph_exfusion.merge.task_search import (
    generate_simplex_grid,
    generate_scale_grid,
    search_ta0,
    search_ta1,
    search_ta2,
    search_ta3,
    search_task_arithmetic,
    evaluate_config,
    check_constraints,
    EvaluationResult,
    SearchResult,
)


# =============================================================================
# Fixtures
# =============================================================================


class TinyLM(nn.Module):
    def __init__(self, vocab=20, hidden=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.linear = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.embed(input_ids)
        h = self.linear(h)
        logits = self.lm_head(h)
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            return type("Out", (), {"loss": loss, "logits": logits})()
        return type("Out", (), {"logits": logits})()


def make_experts(base, n=3, delta_scale=0.5):
    experts = []
    for i in range(n):
        e = TinyLM()
        with torch.no_grad():
            for name, p in e.named_parameters():
                bp = dict(base.named_parameters())[name]
                p.copy_(bp + delta_scale * (i + 1) * torch.randn_like(p))
        experts.append(e)
    return experts


def make_evaluator(base, domain_data):
    """Create an evaluator that computes retention metrics.

    Returns a callable(merged_model) -> dict with:
        mean_retention, min_retention, general_regression, per_domain_retention
    """
    base_nlls = {}
    expert_nlls = {}
    for domain, data in domain_data.items():
        base_loss = 0
        n = 0
        for batch in data:
            with torch.no_grad():
                out = base(batch["input_ids"], labels=batch["labels"])
                base_loss += out.loss.item()
                n += 1
        base_nlls[domain] = base_loss / max(n, 1)

    def evaluator(merged_model):
        merged_model.eval()
        per_domain = {}
        gen_reg = 0.0
        for domain, data in domain_data.items():
            merged_loss = 0
            n = 0
            for batch in data:
                with torch.no_grad():
                    out = merged_model(batch["input_ids"], labels=batch["labels"])
                    merged_loss += out.loss.item()
                    n += 1
            merged_nll = merged_loss / max(n, 1)
            base_nll = base_nlls[domain]
            # Retention = (base - merged) / (base - expert)
            # For testing, assume expert_nll ≈ base_nll * 0.2
            expert_nll = base_nll * 0.2
            if base_nll > expert_nll:
                retention = (base_nll - merged_nll) / (base_nll - expert_nll)
            else:
                retention = 0.0
            per_domain[domain] = retention
            gen_reg += max(0, merged_nll - base_nll) / base_nll

        mean_ret = sum(per_domain.values()) / len(per_domain)
        min_ret = min(per_domain.values())
        return {
            "mean_retention": mean_ret,
            "min_retention": min_ret,
            "general_regression": gen_reg / len(per_domain),
            "per_domain_retention": per_domain,
        }

    return evaluator


def make_domain_data(vocab=20, n_samples=4, seq_len=8):
    data = {}
    for domain in ["math", "planning", "coding"]:
        samples = []
        for _ in range(n_samples):
            ids = torch.randint(0, vocab, (1, seq_len))
            samples.append({"input_ids": ids, "labels": ids})
        data[domain] = samples
    return data


# =============================================================================
# Simplex grid tests
# =============================================================================


class TestSimplexGrid:
    def test_3_experts_0_1_resolution(self):
        grid = generate_simplex_grid(3, 0.1)
        assert len(grid) == 66  # C(12,2) = 66

    def test_3_experts_0_05_resolution(self):
        grid = generate_simplex_grid(3, 0.05)
        assert len(grid) == 231  # C(22,2) = 231

    def test_2_experts_0_1_resolution(self):
        grid = generate_simplex_grid(2, 0.1)
        assert len(grid) == 11

    def test_all_points_sum_to_one(self):
        grid = generate_simplex_grid(3, 0.1)
        for point in grid:
            assert abs(sum(point) - 1.0) < 1e-10

    def test_all_points_non_negative(self):
        grid = generate_simplex_grid(3, 0.1)
        for point in grid:
            for v in point:
                assert v >= -1e-10

    def test_uniform_point_present(self):
        grid = generate_simplex_grid(3, 0.1)
        # At 0.1 resolution, uniform = (0.3, 0.4, 0.3) or similar approximation
        # Check that a near-uniform point exists
        found = False
        for p in grid:
            if all(abs(v - 1.0/3) < 0.07 for v in p):
                found = True
                break
        assert found, "No near-uniform point in grid"

    def test_corner_points_present(self):
        grid = generate_simplex_grid(3, 0.1)
        corners = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        for corner in corners:
            assert corner in grid


# =============================================================================
# Constraint checking tests
# =============================================================================


class TestConstraints:
    def test_feasible(self):
        result = EvaluationResult(
            lambdas=(0.5, 0.5), scale=1.0,
            mean_retention=0.8, min_retention=0.75,
            general_regression=0.2,
        )
        assert check_constraints(result, tau=0.70, delta=0.25)

    def test_min_retention_violated(self):
        result = EvaluationResult(
            lambdas=(0.5, 0.5), scale=1.0,
            mean_retention=0.8, min_retention=0.65,
            general_regression=0.2,
        )
        assert not check_constraints(result, tau=0.70, delta=0.25)

    def test_general_regression_violated(self):
        result = EvaluationResult(
            lambdas=(0.5, 0.5), scale=1.0,
            mean_retention=0.8, min_retention=0.75,
            general_regression=0.30,
        )
        assert not check_constraints(result, tau=0.70, delta=0.25)


# =============================================================================
# TA-0 search tests
# =============================================================================


class TestTA0Search:
    def test_ta0_finds_best_scale(self):
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        result = search_ta0(base, experts, evaluator, scales=[0.5, 1.0, 1.5])
        assert result.mode == "TA-0"
        assert result.n_configurations == 3
        assert result.best is not None
        assert result.best.scale in [0.5, 1.0, 1.5]
        # Uniform lambdas
        assert abs(result.best.lambdas[0] - 0.5) < 1e-10

    def test_ta0_all_scales_evaluated(self):
        base = TinyLM()
        experts = make_experts(base, n=3)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        result = search_ta0(base, experts, evaluator, scales=[0.25, 0.5, 0.75])
        assert len(result.all_results) == 3


# =============================================================================
# TA-1 search tests
# =============================================================================


class TestTA1Search:
    def test_ta1_search_produces_result(self):
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        result = search_ta1(
            base, experts, evaluator,
            resolution=0.5,  # coarse for speed
            scales=[0.5, 1.0],
            refine_around_best=False,
        )
        assert result.mode == "TA-1"
        assert result.n_configurations > 0
        assert result.best is not None
        # Best should have valid lambdas summing to 1
        assert abs(sum(result.best.lambdas) - 1.0) < 0.01

    def test_ta1_more_configurations_than_ta0(self):
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        ta0 = search_ta0(base, experts, evaluator, scales=[0.5, 1.0])
        ta1 = search_ta1(
            base, experts, evaluator,
            resolution=0.5, scales=[0.5, 1.0],
            refine_around_best=False,
        )
        assert ta1.n_configurations > ta0.n_configurations

    def test_ta1_constraint_filtering(self):
        """With tight constraints, fewer configs should be feasible."""
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        # Very tight constraints
        result = search_ta1(
            base, experts, evaluator,
            resolution=0.5, scales=[0.5, 1.0],
            tau=0.99, delta=0.01,
            refine_around_best=False,
        )
        # With very tight constraints, likely no feasible solutions
        # But best should still be selected (from all candidates)
        assert result.best is not None

    def test_ta1_refinement(self):
        """Refinement should add more configurations."""
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        no_refine = search_ta1(
            base, experts, evaluator,
            resolution=0.5, scales=[0.5],
            refine_around_best=False,
        )
        with_refine = search_ta1(
            base, experts, evaluator,
            resolution=0.5, scales=[0.5],
            refine_around_best=True,
            refine_resolution=0.25,
            refine_radius=0.25,
        )
        assert with_refine.n_configurations > no_refine.n_configurations


# =============================================================================
# TA-2 search tests
# =============================================================================


class TestTA2Search:
    def test_ta2_search_with_fisher(self):
        from daph_exfusion.merge.fisher_dense import build_exact_fisher

        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()

        # Build Fisher
        cal_data = [{"input_ids": d["input_ids"], "attention_mask": torch.ones(1, 8), "labels": d["labels"]} for d in data["math"]]
        bank = {}
        for i, e in enumerate(experts):
            f, _ = build_exact_fisher(e, cal_data, max_samples=4)
            bank[f"expert_{i}"] = f

        evaluator = make_evaluator(base, data)

        result = search_ta2(
            base, experts, evaluator, bank,
            resolution=0.5, scales=[0.5, 1.0],
            gamma_grid=[0.0, 0.5],
        )
        assert result.mode == "TA-2"
        assert result.n_configurations > 0
        assert result.best is not None


# =============================================================================
# TA-3 search tests
# =============================================================================


class TestTA3Search:
    def test_ta3_family_weighted(self):
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        result = search_ta3(
            base, experts, evaluator,
            resolution=0.5, scales=[0.5, 1.0],
        )
        assert result.mode == "TA-3"
        assert result.n_configurations > 0
        assert result.best is not None

    def test_ta3_frozen_families(self):
        """Frozen families should not be searched."""
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        result = search_ta3(
            base, experts, evaluator,
            resolution=0.5, scales=[0.5],
            frozen_families=("norm", "embedding", "lm_head"),
        )
        # Should still produce a result
        assert result.best is not None


# =============================================================================
# Unified entry point tests
# =============================================================================


class TestUnifiedSearch:
    def test_search_ta0_via_unified(self):
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        result = search_task_arithmetic(
            base, experts, evaluator, mode="TA-0",
            scales=[0.5, 1.0],
        )
        assert result.mode == "TA-0"

    def test_search_ta1_via_unified(self):
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        result = search_task_arithmetic(
            base, experts, evaluator, mode="TA-1",
            resolution=0.5, scales=[0.5],
            refine_around_best=False,
        )
        assert result.mode == "TA-1"

    def test_search_unknown_mode_raises(self):
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        with pytest.raises(ValueError):
            search_task_arithmetic(
                base, experts, evaluator, mode="TA-99",
            )

    def test_search_ta2_without_bank_raises(self):
        base = TinyLM()
        experts = make_experts(base, n=2)
        data = make_domain_data()
        evaluator = make_evaluator(base, data)

        with pytest.raises(ValueError):
            search_task_arithmetic(
                base, experts, evaluator, mode="TA-2",
            )


# =============================================================================
# SearchResult serialization tests
# =============================================================================


class TestSearchResultSerialization:
    def test_to_dict(self):
        result = SearchResult(
            mode="TA-1",
            resolution=0.1,
            scales=[0.5, 1.0],
            n_configurations=10,
            n_feasible=5,
            tau=0.70,
            delta=0.25,
        )
        result.best = EvaluationResult(
            lambdas=(0.5, 0.5), scale=1.0,
            mean_retention=0.8, min_retention=0.75,
            general_regression=0.2,
            per_domain_retention={"math": 0.8, "planning": 0.75},
        )
        d = result.to_dict()
        assert d["mode"] == "TA-1"
        assert d["n_configurations"] == 10
        assert d["best"]["lambdas"] == [0.5, 0.5]
        assert d["best"]["scale"] == 1.0
