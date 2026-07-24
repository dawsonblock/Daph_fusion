"""Phase 7: Merge-baseline semantics tests.

Verifies that each merge mode implements its correct mathematical
semantics and that softmax is NOT applied where it shouldn't be.
"""
import pytest
import torch
import torch.nn as nn

from daph_hybrid_exfusion_v2_3 import (
    MergeMode,
    merge_expert_family,
    task_arithmetic,
    mean_task_vector,
    softmax_weighted_merge,
    convex_weighted_merge,
)


def _make_base_and_experts(values=(2.0, 4.0)):
    base = nn.Linear(4, 4, bias=False)
    experts = []
    with torch.no_grad():
        base.weight.fill_(0.0)
        for v in values:
            e = nn.Linear(4, 4, bias=False)
            e.weight.fill_(v)
            experts.append(e)
    return base, experts


def test_task_arithmetic_is_sum():
    deltas = [{"w": torch.full((4, 4), 2.0)}, {"w": torch.full((4, 4), 4.0)}]
    result = task_arithmetic(deltas)
    assert torch.allclose(result["w"], torch.full((4, 4), 6.0))


def test_mean_task_vector_is_mean():
    deltas = [{"w": torch.full((4, 4), 2.0)}, {"w": torch.full((4, 4), 4.0)}]
    result = mean_task_vector(deltas)
    assert torch.allclose(result["w"], torch.full((4, 4), 3.0))


def test_weighted_arithmetic_does_not_softmax():
    """WEIGHTED_TASK_ARITHMETIC must use raw coefficients, not softmax."""
    base, experts = _make_base_and_experts((2.0, 4.0))
    # Raw coefficients [2.0, 3.0] -> merged = 2*2 + 3*4 = 4+12 = 16
    # Softmax would give ~[0.27, 0.73] -> merged = 0.27*2 + 0.73*4 ≈ 3.46
    weights = torch.tensor([2.0, 3.0])
    res = merge_expert_family(
        experts, base, weights,
        policies={"merge_mode": MergeMode.WEIGHTED_TASK_ARITHMETIC.value},
    )
    assert torch.allclose(res["weight"], torch.full((4, 4), 16.0)), (
        f"Weighted task arithmetic applied softmax; expected 16.0, got {res['weight'][0,0].item()}"
    )


def test_softmax_merge_does_softmax():
    """LOGIT_WEIGHTED must apply softmax to the logits."""
    base, experts = _make_base_and_experts((2.0, 4.0))
    weights = torch.tensor([0.0, 0.0])  # softmax([0,0]) = [0.5, 0.5]
    res = merge_expert_family(
        experts, base, weights,
        policies={"merge_mode": MergeMode.LOGIT_WEIGHTED.value},
    )
    # 0.5*2 + 0.5*4 = 3.0
    assert torch.allclose(res["weight"], torch.full((4, 4), 3.0))


def test_equal_raw_coefficients_not_confused_with_mean():
    """Raw coefficients [1, 1] with WEIGHTED_TASK_ARITHMETIC should give
    sum (2+4=6), NOT mean (3). This distinguishes it from PARAMETER_AVERAGE."""
    base, experts = _make_base_and_experts((2.0, 4.0))
    weights = torch.tensor([1.0, 1.0])
    res = merge_expert_family(
        experts, base, weights,
        policies={"merge_mode": MergeMode.WEIGHTED_TASK_ARITHMETIC.value},
    )
    # 1*2 + 1*4 = 6 (sum), not 3 (mean)
    assert torch.allclose(res["weight"], torch.full((4, 4), 6.0))


def test_parameter_average_is_mean():
    base, experts = _make_base_and_experts((2.0, 4.0))
    weights = torch.tensor([1.0, 1.0])
    res = merge_expert_family(
        experts, base, weights,
        policies={"merge_mode": MergeMode.PARAMETER_AVERAGE.value},
    )
    assert torch.allclose(res["weight"], torch.full((4, 4), 3.0))


def test_task_arithmetic_via_merge_is_sum():
    base, experts = _make_base_and_experts((2.0, 4.0))
    weights = torch.tensor([1.0, 1.0])
    res = merge_expert_family(
        experts, base, weights,
        policies={"merge_mode": MergeMode.TASK_ARITHMETIC.value},
    )
    assert torch.allclose(res["weight"], torch.full((4, 4), 6.0))


def test_convex_weighted_merge_validates_convexity():
    deltas = [{"w": torch.ones(4)}, {"w": torch.ones(4) * 2}]
    # Valid convex weights
    res = convex_weighted_merge(deltas, torch.tensor([0.3, 0.7]))
    assert torch.allclose(res["w"], torch.ones(4) * 1.7)
    # Invalid: negative
    with pytest.raises(ValueError, match="non-negative"):
        convex_weighted_merge(deltas, torch.tensor([-0.1, 1.1]))
    # Invalid: doesn't sum to 1
    with pytest.raises(ValueError, match="sum to 1"):
        convex_weighted_merge(deltas, torch.tensor([0.3, 0.8]))
