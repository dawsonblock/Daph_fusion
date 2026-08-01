"""Regression contracts for the corrected ExFusion v3 production path."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from daph_exfusion.merge.fisher_dense import MissingCurvatureError
from daph_exfusion.merge.fisher_task_arithmetic import merge_fisher_task_arithmetic
from daph_exfusion.merge.task_arithmetic import merge_task_arithmetic
from daph_exfusion.merge.task_search import search_ta2, search_ta3
from daph_exfusion.merge.types import MergeConfig, MergeMethod


def _linear(value: float) -> nn.Linear:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def _snapshot(model: nn.Module):
    return {
        name: (param.detach().clone(), param.device)
        for name, param in model.named_parameters()
    }


def _assert_unchanged(model: nn.Module, snapshot) -> None:
    for name, param in model.named_parameters():
        expected, device = snapshot[name]
        assert param.device == device
        assert torch.equal(param.detach().cpu(), expected.detach().cpu())


def test_task_arithmetic_exact_formula_and_no_input_mutation():
    base = _linear(1.0)
    expert_a = _linear(3.0)  # delta +2
    expert_b = _linear(0.0)  # delta -1
    snapshots = [_snapshot(m) for m in (base, expert_a, expert_b)]

    config = MergeConfig(
        method=MergeMethod.TASK_ARITHMETIC,
        task_scale=0.5,
        lambdas=(0.75, 0.25),
    )
    result = merge_task_arithmetic(base, [expert_a, expert_b], config)

    # 1 + .5 * (.75*2 + .25*(-1)) = 1.625
    assert torch.allclose(result.merged_model.weight, torch.full_like(result.merged_model.weight, 1.625))
    assert result.merged_model is not base
    for model, snapshot in zip((base, expert_a, expert_b), snapshots):
        _assert_unchanged(model, snapshot)


def test_fisher_task_arithmetic_exact_formula_and_no_input_mutation():
    base = _linear(1.0)
    expert_a = _linear(3.0)  # delta +2
    expert_b = _linear(0.0)  # delta -1
    snapshots = [_snapshot(m) for m in (base, expert_a, expert_b)]

    curvature = {
        "expert_0": {"weight": torch.full_like(base.weight, 4.0)},
        "expert_1": {"weight": torch.full_like(base.weight, 1.0)},
    }
    config = MergeConfig(
        method=MergeMethod.FISHER_DENSE,
        task_scale=1.0,
        lambdas=(0.5, 0.5),
        fisher_gamma=1.0,
        fisher_floor_eps=1e-12,
    )
    result = merge_fisher_task_arithmetic(base, [expert_a, expert_b], config, curvature)

    # (0.5*4*2 + 0.5*1*(-1)) / (0.5*4 + 0.5*1) = 1.4; base + 1.4 = 2.4
    assert torch.allclose(result.merged_model.weight, torch.full_like(result.merged_model.weight, 2.4), atol=1e-6)
    assert result.trace.operators == ["EMPIRICAL_FISHER", "FISHER_TASK_ARITHMETIC"]
    for model, snapshot in zip((base, expert_a, expert_b), snapshots):
        _assert_unchanged(model, snapshot)


def test_fisher_missing_curvature_fails_closed():
    base = _linear(1.0)
    expert = _linear(2.0)
    config = MergeConfig(method=MergeMethod.FISHER_DENSE)
    with pytest.raises(MissingCurvatureError):
        merge_fisher_task_arithmetic(base, [expert], config, curvature_bank={})


def test_ta2_search_does_not_swallow_missing_curvature():
    base = _linear(1.0)
    expert = _linear(2.0)

    def evaluator(_model):
        return {
            "mean_retention": 1.0,
            "min_retention": 1.0,
            "general_regression": 0.0,
        }

    with pytest.raises(MissingCurvatureError):
        search_ta2(
            base,
            [expert],
            evaluator,
            curvature_bank={},
            resolution=1.0,
            scales=[1.0],
            gamma_grid=[1.0],
        )


def test_ta3_preserves_full_family_solution_and_hash():
    base = _linear(0.0)
    expert_a = _linear(1.0)
    expert_b = _linear(3.0)

    def evaluator(model):
        # Prefer weight 3.0, i.e. expert_b, and make every candidate feasible.
        value = float(model.weight.detach().mean())
        score = 1.0 - abs(value - 3.0) / 10.0
        return {
            "mean_retention": score,
            "min_retention": score,
            "general_regression": 0.0,
            "per_domain_retention": {"toy": score},
        }

    result = search_ta3(
        base,
        [expert_a, expert_b],
        evaluator,
        resolution=1.0,
        scales=[1.0],
        tau=0.0,
        delta=1.0,
        frozen_families=(),
    )
    serialized = result.to_dict()

    assert result.best is not None
    assert result.family_order == ["other"]
    assert result.family_lambdas["other"] == (0.0, 1.0)
    assert result.family_scales["other"] == 1.0
    assert len(result.solution_hash) == 64
    assert serialized["family_lambdas"]["other"] == [0.0, 1.0]
    assert serialized["solution_hash"] == result.solution_hash
