import pytest
import torch
import torch.nn as nn

from daph_hybrid_exfusion_v2_3 import MergeMode, merge_expert_family
from research_metrics import calculate_retention, compute_expert_advantage


def test_parameter_average_equals_mean():
    base = nn.Linear(4, 4, bias=False)
    e1 = nn.Linear(4, 4, bias=False)
    e2 = nn.Linear(4, 4, bias=False)

    with torch.no_grad():
        base.weight.fill_(0.0)
        e1.weight.fill_(2.0)
        e2.weight.fill_(4.0)

    experts = [e1, e2]
    weights = torch.tensor([1.0, 1.0])

    res = merge_expert_family(
        experts,
        base,
        weights,
        policies={"merge_mode": MergeMode.PARAMETER_AVERAGE.value},
    )
    assert torch.allclose(res["weight"], torch.full((4, 4), 3.0))


def test_task_arithmetic_equals_sum():
    base = nn.Linear(4, 4, bias=False)
    e1 = nn.Linear(4, 4, bias=False)
    e2 = nn.Linear(4, 4, bias=False)

    with torch.no_grad():
        base.weight.fill_(0.0)
        e1.weight.fill_(2.0)
        e2.weight.fill_(4.0)

    experts = [e1, e2]
    weights = torch.tensor([1.0, 1.0])

    res = merge_expert_family(
        experts, base, weights, policies={"merge_mode": MergeMode.TASK_ARITHMETIC.value}
    )
    assert torch.allclose(res["weight"], torch.full((4, 4), 6.0))


def test_task_arithmetic_is_n_times_average():
    base = nn.Linear(4, 4, bias=False)
    e1 = nn.Linear(4, 4, bias=False)
    e2 = nn.Linear(4, 4, bias=False)

    with torch.no_grad():
        base.weight.fill_(0.0)
        e1.weight.fill_(1.5)
        e2.weight.fill_(2.5)

    experts = [e1, e2]
    weights = torch.tensor([1.0, 1.0])

    avg_res = merge_expert_family(
        experts,
        base,
        weights,
        policies={"merge_mode": MergeMode.PARAMETER_AVERAGE.value},
    )
    ta_res = merge_expert_family(
        experts, base, weights, policies={"merge_mode": MergeMode.TASK_ARITHMETIC.value}
    )
    assert torch.allclose(ta_res["weight"], 2.0 * avg_res["weight"])


def test_invalid_retention_when_expert_worse_than_base():
    # Base loss = 5.0, Expert loss = 6.0 (expert is worse)
    res = calculate_retention(base_loss=5.0, expert_loss=6.0, merged_loss=5.5)
    assert not res.valid
    assert res.value is None
    assert res.reason == "expert_does_not_outperform_base"


def test_retention_equals_one_for_expert():
    # Merged loss matches expert loss = 3.0, Base = 5.0
    res = calculate_retention(base_loss=5.0, expert_loss=3.0, merged_loss=3.0)
    assert res.valid
    assert abs(res.value - 1.0) < 1e-5


def test_retention_equals_zero_for_base():
    # Merged loss matches base loss = 5.0, Expert = 3.0
    res = calculate_retention(base_loss=5.0, expert_loss=3.0, merged_loss=5.0)
    assert res.valid
    assert abs(res.value - 0.0) < 1e-5
