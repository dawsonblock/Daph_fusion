"""Research-contract tests for v3 merge methods.

Most important test class. Verifies that the reported method equals the
executed method. Prevents another fake-AGX situation.

Example:
    result = merge_experts(..., MergeConfig(method="regmean"))
    assert result.trace.operator == "REGMEAN"
    assert result.trace.activation_covariance_used
    assert not result.trace.fisher_used
    assert not result.trace.dare_used
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from daph_exfusion.merge.types import (
    MergeConfig, MergeMethod, OperatorTrace,
)
from daph_exfusion.merge.task_arithmetic import merge_task_arithmetic, merge_frozen
from daph_exfusion.merge.fisher_dense import (
    merge_fisher_dense, merge_fisher_base_anchored, build_exact_fisher,
)
from daph_exfusion.merge.regmean import merge_regmean
from daph_exfusion.merge.subspace import merge_subspace
from daph_exfusion.merge.legacy import op_dare, op_ties


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


def make_calibration_data(vocab=20, n=4, seq=8):
    data = []
    for _ in range(n):
        ids = torch.randint(0, vocab, (1, seq))
        data.append({"input_ids": ids, "attention_mask": torch.ones(1, seq), "labels": ids})
    return data


def make_experts(base, n=2, delta_scale=0.5):
    experts = []
    for i in range(n):
        e = TinyLM()
        with torch.no_grad():
            for name, p in e.named_parameters():
                bp = dict(base.named_parameters())[name]
                p.copy_(bp + delta_scale * (i + 1) * torch.randn_like(p))
        experts.append(e)
    return experts


# =============================================================================
# Research-contract: Task Arithmetic
# =============================================================================


class TestTaskArithmeticContract:
    def test_trace_has_task_arithmetic(self):
        base = TinyLM()
        experts = make_experts(base)
        config = MergeConfig(method=MergeMethod.TASK_ARITHMETIC)
        result = merge_task_arithmetic(base, experts, config)
        result.trace.assert_operator("TASK_ARITHMETIC")

    def test_trace_no_fisher(self):
        base = TinyLM()
        experts = make_experts(base)
        config = MergeConfig(method=MergeMethod.TASK_ARITHMETIC)
        result = merge_task_arithmetic(base, experts, config)
        result.trace.assert_no_operator("EMPIRICAL_FISHER")
        assert not result.trace.fisher_used

    def test_trace_no_dare(self):
        base = TinyLM()
        experts = make_experts(base)
        config = MergeConfig(method=MergeMethod.TASK_ARITHMETIC)
        result = merge_task_arithmetic(base, experts, config)
        assert not result.trace.dare_used

    def test_trace_no_ties(self):
        base = TinyLM()
        experts = make_experts(base)
        config = MergeConfig(method=MergeMethod.TASK_ARITHMETIC)
        result = merge_task_arithmetic(base, experts, config)
        assert not result.trace.ties_used

    def test_lambda_zero_returns_base(self):
        base = TinyLM()
        experts = make_experts(base)
        config = MergeConfig(method=MergeMethod.TASK_ARITHMETIC, task_scale=0.0)
        result = merge_task_arithmetic(base, experts, config)
        for name, p in result.merged_model.named_parameters():
            bp = dict(base.named_parameters())[name]
            assert torch.allclose(p, bp, atol=1e-5)

    def test_single_expert_lambda_one_returns_expert(self):
        base = TinyLM()
        expert = make_experts(base, n=1)[0]
        config = MergeConfig(
            method=MergeMethod.TASK_ARITHMETIC,
            task_scale=1.0,
            lambdas=(1.0,),
        )
        result = merge_task_arithmetic(base, [expert], config)
        for name, p in result.merged_model.named_parameters():
            ep = dict(expert.named_parameters())[name]
            assert torch.allclose(p, ep, atol=1e-4)


# =============================================================================
# Research-contract: Fisher Dense
# =============================================================================


class TestFisherDenseContract:
    def test_trace_has_fisher(self):
        base = TinyLM()
        experts = make_experts(base)
        data = make_calibration_data()
        bank = {}
        for i, e in enumerate(experts):
            f, _ = build_exact_fisher(e, data, max_samples=4)
            bank[f"expert_{i}"] = f
        config = MergeConfig(method=MergeMethod.FISHER_DENSE)
        result = merge_fisher_dense(base, experts, config, bank)
        result.trace.assert_operator("EMPIRICAL_FISHER")
        result.trace.assert_operator("DENSE_PRECISION_MERGE")

    def test_trace_fisher_estimator_exact(self):
        base = TinyLM()
        experts = make_experts(base)
        data = make_calibration_data()
        bank = {}
        for i, e in enumerate(experts):
            f, _ = build_exact_fisher(e, data, max_samples=4)
            bank[f"expert_{i}"] = f
        config = MergeConfig(method=MergeMethod.FISHER_DENSE)
        result = merge_fisher_dense(base, experts, config, bank)
        assert result.trace.fisher_estimator == "exact_per_sample"

    def test_trace_no_dare(self):
        base = TinyLM()
        experts = make_experts(base)
        data = make_calibration_data()
        bank = {}
        for i, e in enumerate(experts):
            f, _ = build_exact_fisher(e, data, max_samples=4)
            bank[f"expert_{i}"] = f
        config = MergeConfig(method=MergeMethod.FISHER_DENSE)
        result = merge_fisher_dense(base, experts, config, bank)
        assert not result.trace.dare_used
        assert not result.trace.ties_used

    def test_trace_no_activation_covariance(self):
        base = TinyLM()
        experts = make_experts(base)
        data = make_calibration_data()
        bank = {}
        for i, e in enumerate(experts):
            f, _ = build_exact_fisher(e, data, max_samples=4)
            bank[f"expert_{i}"] = f
        config = MergeConfig(method=MergeMethod.FISHER_DENSE)
        result = merge_fisher_dense(base, experts, config, bank)
        assert not result.trace.activation_covariance_used


# =============================================================================
# Research-contract: Fisher Base-Anchored
# =============================================================================


class TestFisherBaseAnchoredContract:
    def test_trace_has_base_anchor(self):
        base = TinyLM()
        experts = make_experts(base)
        data = make_calibration_data()
        bank = {}
        for i, e in enumerate(experts):
            f, _ = build_exact_fisher(e, data, max_samples=4)
            bank[f"expert_{i}"] = f
        base_f, _ = build_exact_fisher(base, data, max_samples=4)
        config = MergeConfig(method=MergeMethod.FISHER_BASE_ANCHORED, base_precision_weight=0.5)
        result = merge_fisher_base_anchored(base, experts, config, bank, base_f)
        result.trace.assert_operator("BASE_ANCHOR")
        assert result.trace.base_precision_weight == 0.5


# =============================================================================
# Research-contract: RegMean
# =============================================================================


class TestRegMeanContract:
    def test_trace_has_regmean(self):
        base = nn.Linear(8, 4)
        experts = []
        for i in range(2):
            e = nn.Linear(8, 4)
            with torch.no_grad():
                e.weight.copy_(base.weight + 0.5 * (i + 1))
            experts.append(e)
        activation_bank = {
            "expert_0": {"weight": torch.rand(8) + 0.1},
            "expert_1": {"weight": torch.rand(8) + 0.1},
        }
        config = MergeConfig(method=MergeMethod.REGMEAN)
        result = merge_regmean(base, experts, config, activation_bank)
        result.trace.assert_operator("REGMEAN")

    def test_trace_activation_covariance_used(self):
        base = nn.Linear(8, 4)
        experts = []
        for i in range(2):
            e = nn.Linear(8, 4)
            with torch.no_grad():
                e.weight.copy_(base.weight + 0.5 * (i + 1))
            experts.append(e)
        activation_bank = {
            "expert_0": {"weight": torch.rand(8) + 0.1},
            "expert_1": {"weight": torch.rand(8) + 0.1},
        }
        config = MergeConfig(method=MergeMethod.REGMEAN)
        result = merge_regmean(base, experts, config, activation_bank)
        assert result.trace.activation_covariance_used

    def test_trace_no_fisher(self):
        base = nn.Linear(8, 4)
        experts = []
        for i in range(2):
            e = nn.Linear(8, 4)
            with torch.no_grad():
                e.weight.copy_(base.weight + 0.5 * (i + 1))
            experts.append(e)
        activation_bank = {
            "expert_0": {"weight": torch.rand(8) + 0.1},
            "expert_1": {"weight": torch.rand(8) + 0.1},
        }
        config = MergeConfig(method=MergeMethod.REGMEAN)
        result = merge_regmean(base, experts, config, activation_bank)
        assert not result.trace.fisher_used

    def test_trace_no_dare(self):
        base = nn.Linear(8, 4)
        experts = []
        for i in range(2):
            e = nn.Linear(8, 4)
            with torch.no_grad():
                e.weight.copy_(base.weight + 0.5 * (i + 1))
            experts.append(e)
        activation_bank = {
            "expert_0": {"weight": torch.rand(8) + 0.1},
            "expert_1": {"weight": torch.rand(8) + 0.1},
        }
        config = MergeConfig(method=MergeMethod.REGMEAN)
        result = merge_regmean(base, experts, config, activation_bank)
        assert not result.trace.dare_used


# =============================================================================
# Research-contract: Subspace merge
# =============================================================================


class TestSubspaceContract:
    def test_trace_has_svd_diagnostic(self):
        base = nn.Linear(8, 4)
        experts = []
        for i in range(2):
            e = nn.Linear(8, 4)
            with torch.no_grad():
                e.weight.copy_(base.weight + 0.5 * (i + 1))
            experts.append(e)
        config = MergeConfig(method=MergeMethod.TASK_ARITHMETIC)  # subspace uses TA config
        result = merge_subspace(base, experts, config)
        result.trace.assert_operator("SVD_DIAGNOSTIC")


# =============================================================================
# Research-contract: Frozen
# =============================================================================


class TestFrozenContract:
    def test_frozen_returns_base(self):
        base = TinyLM()
        config = MergeConfig(method=MergeMethod.FROZEN)
        result = merge_frozen(base, config)
        result.trace.assert_operator("FROZEN")
        for name, p in result.merged_model.named_parameters():
            bp = dict(base.named_parameters())[name]
            assert torch.allclose(p, bp)


# =============================================================================
# Research-contract: Legacy methods
# =============================================================================


class TestLegacyContract:
    def test_dare_trace_has_dare(self):
        from daph_exfusion.merge.pipeline_v3 import merge_experts
        base = TinyLM()
        experts = make_experts(base)
        config = MergeConfig(
            method=MergeMethod.DARE,
            legacy_sparse={"dare_drop_rate": 0.2},
        )
        result = merge_experts(base, experts, config)
        assert result.trace.dare_used
        assert "DARE" in result.trace.operators

    def test_ties_trace_has_ties(self):
        from daph_exfusion.merge.pipeline_v3 import merge_experts
        base = TinyLM()
        experts = make_experts(base)
        config = MergeConfig(
            method=MergeMethod.TIES_MAGNITUDE,
            legacy_sparse={"ties_trim_fraction": 0.2},
        )
        result = merge_experts(base, experts, config)
        assert result.trace.ties_used
        assert not result.trace.dare_used

    def test_dare_ties_trace_has_both(self):
        from daph_exfusion.merge.pipeline_v3 import merge_experts
        base = TinyLM()
        experts = make_experts(base)
        config = MergeConfig(
            method=MergeMethod.DARE_TIES,
            legacy_sparse={"dare_drop_rate": 0.2, "ties_trim_fraction": 0.2},
        )
        result = merge_experts(base, experts, config)
        assert result.trace.dare_used
        assert result.trace.ties_used


# =============================================================================
# Research-contract: Release gates
# =============================================================================


class TestReleaseGatesContract:
    def test_paper_ready_false_by_default(self):
        from daph_exfusion.validation.release_gates import ReleaseGates
        gates = ReleaseGates()
        assert not gates.paper_ready

    def test_paper_ready_true_when_all_pass(self):
        from daph_exfusion.validation.release_gates import ReleaseGates
        gates = ReleaseGates(
            full_tests_pass=True,
            checkpoints_verified=True,
            split_integrity_verified=True,
            expert_qualification_verified=True,
            algorithm_trace_verified=True,
            sample_statistics_verified=True,
            no_test_leakage=True,
            results_reproduced=True,
        )
        assert gates.paper_ready

    def test_fisher_verified_requires_exact(self):
        from daph_exfusion.validation.release_gates import verify_fisher
        trace = OperatorTrace(
            method="fisher_dense",
            fisher_used=True,
            fisher_estimator="exact_per_sample",
        )
        assert verify_fisher(trace, calibration_hash_valid=True, pseudo_labels=False)

    def test_fisher_not_verified_for_microbatch(self):
        from daph_exfusion.validation.release_gates import verify_fisher
        trace = OperatorTrace(
            method="fisher_dense",
            fisher_used=True,
            fisher_estimator="microbatch_gradient_square",
        )
        assert not verify_fisher(trace, calibration_hash_valid=True, pseudo_labels=False)

    def test_fisher_not_verified_for_pseudo_labels(self):
        from daph_exfusion.validation.release_gates import verify_fisher
        trace = OperatorTrace(
            method="fisher_dense",
            fisher_used=True,
            fisher_estimator="exact_per_sample",
        )
        assert not verify_fisher(trace, calibration_hash_valid=True, pseudo_labels=True)

    def test_algorithm_trace_verified(self):
        from daph_exfusion.validation.release_gates import verify_algorithm_trace
        trace = OperatorTrace(
            method="regmean",
            operators=["REGMEAN", "TASK_ARITHMETIC_FALLBACK"],
        )
        assert verify_algorithm_trace(trace, "regmean")

    def test_algorithm_trace_not_verified_for_mismatch(self):
        from daph_exfusion.validation.release_gates import verify_algorithm_trace
        trace = OperatorTrace(
            method="regmean",
            operators=["DARE"],  # Wrong operator!
        )
        assert not verify_algorithm_trace(trace, "regmean")
