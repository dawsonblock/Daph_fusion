"""Tests for v3 Fisher merge: exact empirical Fisher, dense merge, base-anchored.

Fisher tests (required by plan):
    test_exact_fisher_matches_manual_per_sample_loop
    test_batch_gradient_square_does_not_claim_exact
    test_fisher_not_delta_squared
    test_fisher_fp32_accumulation
    test_gamma_zero_reduces_to_uniform_precision
    test_base_anchor_zero_reduces_to_expert_only
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from daph_exfusion.merge.types import (
    MergeConfig, MergeMethod, OperatorTrace,
)
from daph_exfusion.merge.fisher_dense import (
    build_exact_fisher,
    merge_fisher_dense,
    merge_fisher_base_anchored,
    stabilize_fisher,
    compute_fisher_stats,
    FisherStabilization,
)


# =============================================================================
# Test fixtures
# =============================================================================


class TinyLM(nn.Module):
    """Tiny language model for testing."""
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


def make_calibration_data(vocab=20, n_samples=4, seq_len=8):
    """Make simple calibration data."""
    data = []
    for i in range(n_samples):
        input_ids = torch.randint(0, vocab, (1, seq_len))
        attention_mask = torch.ones(1, seq_len)
        labels = input_ids.clone()
        data.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
    return data


# =============================================================================
# Exact empirical Fisher tests
# =============================================================================


class TestExactFisher:
    def test_exact_fisher_matches_manual_per_sample_loop(self):
        """Fisher from build_exact_fisher must match manual per-sample loop."""
        torch.manual_seed(42)
        model = TinyLM()
        data = make_calibration_data(n_samples=4)

        # Manual per-sample Fisher
        manual_fisher = {}
        for name, param in model.named_parameters():
            manual_fisher[name] = torch.zeros_like(param.detach().float())
        count = 0
        for batch in data:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=batch["attention_mask"], labels=labels)
            out.loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    manual_fisher[name] += param.grad.detach().float().square()
            count += 1
        for name in manual_fisher:
            manual_fisher[name] /= count

        # API Fisher
        api_fisher, stats = build_exact_fisher(model, data, max_samples=4)

        # Compare
        for name in manual_fisher:
            assert name in api_fisher, f"Missing {name} in API fisher"
            assert torch.allclose(manual_fisher[name], api_fisher[name], atol=1e-5), \
                f"Fisher mismatch for {name}"

    def test_fisher_fp32_accumulation(self):
        """Fisher must accumulate in FP32."""
        model = TinyLM()
        data = make_calibration_data(n_samples=2)
        fisher, stats = build_exact_fisher(model, data, max_samples=2)
        for name, f in fisher.items():
            assert f.dtype == torch.float32, f"{name} is {f.dtype}, expected float32"

    def test_fisher_not_delta_squared(self):
        """Fisher should NOT equal delta² (it's gradient-based, not parameter-difference-based)."""
        model = TinyLM()
        data = make_calibration_data(n_samples=4)
        fisher, stats = build_exact_fisher(model, data, max_samples=4)

        # Fisher should have non-zero values that are NOT simply param²
        for name, f in fisher.items():
            param = dict(model.named_parameters())[name]
            delta_sq = param.detach().float() ** 2
            # Fisher and delta² should be different (unless by coincidence)
            if f.sum() > 0:
                assert not torch.allclose(f, delta_sq, atol=1e-6), \
                    f"Fisher equals delta² for {name} — this is wrong"

    def test_fisher_stats_valid(self):
        """Fisher stats should be computed and valid."""
        model = TinyLM()
        data = make_calibration_data(n_samples=2)
        fisher, stats = build_exact_fisher(model, data, max_samples=2)
        for name, s in stats.items():
            assert isinstance(s.min_val, float)
            assert isinstance(s.max_val, float)
            assert isinstance(s.fraction_nonfinite, float)
            assert s.fraction_nonfinite < 0.5  # mostly finite

    def test_fisher_zero_for_no_gradient(self):
        """Parameters with no gradient should have zero Fisher."""
        model = TinyLM()
        # Freeze embed so it gets no gradient
        model.embed.weight.requires_grad = False
        data = make_calibration_data(n_samples=2)
        fisher, stats = build_exact_fisher(model, data, max_samples=2)
        # embed should not be in fisher (requires_grad=False)
        assert "embed.weight" not in fisher


# =============================================================================
# Dense Fisher merge tests
# =============================================================================


class TestFisherDenseMerge:
    def test_fisher_dense_produces_valid_model(self):
        base = TinyLM()
        expert1 = TinyLM()
        expert2 = TinyLM()
        # Make experts different
        with torch.no_grad():
            expert1.linear.weight.copy_(base.linear.weight + 0.5)
            expert2.linear.weight.copy_(base.linear.weight - 0.3)

        data = make_calibration_data(n_samples=2)
        fisher1, _ = build_exact_fisher(expert1, data, max_samples=2)
        fisher2, _ = build_exact_fisher(expert2, data, max_samples=2)
        curvature_bank = {"expert_0": fisher1, "expert_1": fisher2}

        config = MergeConfig(method=MergeMethod.FISHER_DENSE, fisher_gamma=0.5)
        result = merge_fisher_dense(base, [expert1, expert2], config, curvature_bank)

        assert result.method == "fisher_dense"
        assert "EMPIRICAL_FISHER" in result.trace.operators
        assert "DENSE_PRECISION_MERGE" in result.trace.operators
        assert result.trace.fisher_used
        assert result.trace.fisher_estimator == "exact_per_sample"

    def test_gamma_zero_reduces_to_uniform(self):
        """γ=0 should collapse to uniform expert weighting (F^0 = 1 for all)."""
        base = TinyLM()
        expert1 = TinyLM()
        expert2 = TinyLM()
        with torch.no_grad():
            expert1.linear.weight.copy_(base.linear.weight + 1.0)
            expert2.linear.weight.copy_(base.linear.weight + 3.0)

        # Fisher with all ones (γ=0 makes F^0=1)
        curvature_bank = {
            "expert_0": {n: torch.ones_like(p) for n, p in base.named_parameters()},
            "expert_1": {n: torch.ones_like(p) for n, p in base.named_parameters()},
        }

        config = MergeConfig(method=MergeMethod.FISHER_DENSE, fisher_gamma=0.0)
        result = merge_fisher_dense(base, [expert1, expert2], config, curvature_bank)

        # With γ=0 and uniform Fisher, the merge should be the average of deltas
        # Δ* = (1*Δ₁ + 1*Δ₂) / (1 + 1) = (Δ₁ + Δ₂) / 2
        merged_weight = dict(result.merged_model.named_parameters())["linear.weight"]
        base_weight = dict(base.named_parameters())["linear.weight"]
        expected_delta = (1.0 + 3.0) / 2  # average of deltas
        assert torch.allclose(merged_weight, base_weight + expected_delta, atol=1e-4)


# =============================================================================
# Base-anchored Fisher merge tests
# =============================================================================


class TestFisherBaseAnchored:
    def test_base_anchored_produces_valid_model(self):
        base = TinyLM()
        expert1 = TinyLM()
        expert2 = TinyLM()
        with torch.no_grad():
            expert1.linear.weight.copy_(base.linear.weight + 0.5)
            expert2.linear.weight.copy_(base.linear.weight - 0.3)

        data = make_calibration_data(n_samples=2)
        fisher1, _ = build_exact_fisher(expert1, data, max_samples=2)
        fisher2, _ = build_exact_fisher(expert2, data, max_samples=2)
        base_fisher, _ = build_exact_fisher(base, data, max_samples=2)
        curvature_bank = {"expert_0": fisher1, "expert_1": fisher2}

        config = MergeConfig(
            method=MergeMethod.FISHER_BASE_ANCHORED,
            fisher_gamma=0.5,
            base_precision_weight=0.5,
        )
        result = merge_fisher_base_anchored(
            base, [expert1, expert2], config, curvature_bank, base_fisher
        )

        assert result.method == "fisher_base_anchored"
        assert "BASE_ANCHOR" in result.trace.operators
        assert result.trace.base_precision_weight == 0.5

    def test_base_anchor_zero_reduces_to_expert_only(self):
        """λ₀=0 should reduce to expert-only Fisher merge."""
        base = TinyLM()
        expert1 = TinyLM()
        expert2 = TinyLM()
        with torch.no_grad():
            expert1.linear.weight.copy_(base.linear.weight + 1.0)
            expert2.linear.weight.copy_(base.linear.weight + 3.0)

        curvature_bank = {
            "expert_0": {n: torch.ones_like(p) for n, p in base.named_parameters()},
            "expert_1": {n: torch.ones_like(p) for n, p in base.named_parameters()},
        }
        base_fisher = {n: torch.ones_like(p) for n, p in base.named_parameters()}

        config = MergeConfig(
            method=MergeMethod.FISHER_BASE_ANCHORED,
            fisher_gamma=0.0,
            base_precision_weight=0.0,
        )
        result = merge_fisher_base_anchored(
            base, [expert1, expert2], config, curvature_bank, base_fisher
        )

        # With λ₀=0, γ=0, uniform Fisher: should be same as expert-only
        # Δ* = (1*Δ₁ + 1*Δ₂) / (0 + 1 + 1) = (Δ₁ + Δ₂) / 2
        merged_weight = dict(result.merged_model.named_parameters())["linear.weight"]
        base_weight = dict(base.named_parameters())["linear.weight"]
        expected_delta = (1.0 + 3.0) / 2
        assert torch.allclose(merged_weight, base_weight + expected_delta, atol=1e-4)


# =============================================================================
# Fisher stabilization tests
# =============================================================================


class TestFisherStabilization:
    def test_floor(self):
        f = torch.tensor([0.0, 0.1, 0.5, 1.0])
        stabilized = stabilize_fisher(f, FisherStabilization.FLOOR, floor_eps=0.01)
        assert abs(stabilized[0].item() - 0.01) < 1e-5
        assert abs(stabilized[1].item() - 0.1) < 1e-5

    def test_log_compress(self):
        f = torch.tensor([0.0, 1.0, 10.0])
        stabilized = stabilize_fisher(f, FisherStabilization.LOG_COMPRESS, log_alpha=1.0)
        assert stabilized[0].item() == 0.0
        assert stabilized[1].item() > 0

    def test_none(self):
        f = torch.tensor([0.0, 0.5, 1.0])
        stabilized = stabilize_fisher(f, FisherStabilization.NONE)
        assert torch.allclose(stabilized, f)
