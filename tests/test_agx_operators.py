"""Phase 10-11: AGX operators and layer-descriptor tests.

Verifies that every AGX operator has a real mathematical contract
(no stubs) and that layer descriptors are computed correctly.
"""
import pytest
import torch

from daph_exfusion.geometry.operators import (
    op_raw,
    op_normalized,
    op_dare,
    op_delta_dropout,
    op_project,
    op_ties,
    op_fisher_weighted,
    transform_single_delta,
    transform_expert_set,
    normalize_raw,
    normalize_unit_frobenius,
    normalize_base_relative,
    normalize_median_expert_relative,
    normalize_clipped_norm,
    SINGLE_EXPERT_OPS,
    CROSS_EXPERT_OPS,
)


def test_op_raw_is_identity():
    delta = torch.randn(4, 4)
    assert torch.equal(op_raw(delta), delta)


def test_op_normalized_sets_unit_norm():
    delta = torch.randn(4, 4) * 10
    result = op_normalized(delta, target_scale=1.0)
    assert abs(result.norm().item() - 1.0) < 1e-4


def test_op_dare_preserves_expectation():
    torch.manual_seed(0)
    delta = torch.randn(2000) * 3.0 + 1.0
    p = 0.3
    accum = torch.zeros(2000)
    for _ in range(300):
        accum += op_dare(delta, drop_probability=p)
    mean = accum / 300
    rel_err = (mean - delta).abs().mean() / delta.abs().mean()
    assert rel_err.item() < 0.06


def test_op_delta_dropout_scales_by_one_minus_p():
    torch.manual_seed(0)
    delta = torch.randn(2000) * 3.0 + 1.0
    p = 0.3
    accum = torch.zeros(2000)
    for _ in range(300):
        accum += op_delta_dropout(delta, drop_probability=p)
    mean = accum / 300
    expected = (1 - p) * delta
    rel_err = (mean - expected).abs().mean() / expected.abs().mean()
    assert rel_err.item() < 0.06


def test_op_project_is_idempotent():
    """Projecting twice must give the same result as projecting once."""
    torch.manual_seed(0)
    delta = torch.randn(10, 4)
    # Random orthonormal subspace of dimension 2
    U, _ = torch.linalg.qr(torch.randn(4, 2))
    projected_once = op_project(delta, conflict_subspace=U)
    projected_twice = op_project(projected_once, conflict_subspace=U)
    assert torch.allclose(projected_once, projected_twice, atol=1e-4)


def test_op_project_changes_conflicting_delta():
    """Projection must change the delta if it has a component in the subspace."""
    torch.manual_seed(0)
    delta = torch.randn(5, 4)
    U, _ = torch.linalg.qr(torch.randn(4, 2))
    projected = op_project(delta, conflict_subspace=U)
    # The projection should change the delta (unless delta is orthogonal to U)
    assert not torch.allclose(delta, projected, atol=1e-4)


def test_op_project_none_subspace_is_identity():
    delta = torch.randn(4, 4)
    assert torch.equal(op_project(delta, conflict_subspace=None), delta)


def test_op_ties_cross_expert_sign_election():
    """TIES must elect the sign with greater total magnitude."""
    # 3 experts: 2 positive, 1 negative on a parameter
    deltas = [
        torch.tensor([[3.0, 1.0]]),
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[-5.0, 1.0]]),
    ]
    # For param 0: total_pos = 5, total_neg = 5 -> elect positive (>=)
    # For param 1: all positive -> elect positive
    merged = op_ties(deltas, trim_fraction=0.0)
    # Param 0: only positive entries (3, 2) contribute, averaged = 2.5
    assert abs(merged[0, 0].item() - 2.5) < 1e-4
    # Param 1: all contribute, averaged = 1.0
    assert abs(merged[0, 1].item() - 1.0) < 1e-4


def test_op_ties_trims_small_magnitudes():
    """TIES trim must zero out the smallest-magnitude elements."""
    deltas = [torch.tensor([[0.01, 10.0, 0.01, 10.0]])]
    merged = op_ties(deltas, trim_fraction=0.5)
    # 50% trim: the 0.01 values should be zeroed
    assert merged[0, 0].item() == 0.0
    assert merged[0, 2].item() == 0.0
    assert merged[0, 1].item() == 10.0
    assert merged[0, 3].item() == 10.0


def test_op_fisher_weights_by_curvature():
    """Fisher merge must weight higher-curvature experts more."""
    deltas = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]
    fisher = [torch.tensor([0.01, 0.01]), torch.tensor([100.0, 100.0])]
    merged = op_fisher_weighted(deltas, fisher, gamma=0.5)
    # Expert 1 has much higher Fisher -> weight ~1, expert 0 -> ~0
    # merged ≈ 1*1 + 1*1 * (0.01^0.5 / (0.01^0.5 + 100^0.5)) ≈ mostly expert 1
    assert merged[0].item() > 0.9  # close to expert 1's value


def test_agx_fisher_not_equivalent_raw():
    """FISHER must produce a different result from RAW."""
    deltas = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    fisher = [torch.tensor([1.0, 1.0]), torch.tensor([10.0, 10.0])]
    fisher_merged = op_fisher_weighted(deltas, fisher, gamma=0.5)
    raw_merged = sum(deltas)  # RAW = sum of deltas
    assert not torch.allclose(fisher_merged, raw_merged, atol=1e-4)


def test_agx_dare_expectation():
    """AGX DARE operator must preserve expected delta."""
    torch.manual_seed(42)
    delta = torch.randn(500) * 2.0
    p = 0.25
    accum = torch.zeros(500)
    for _ in range(200):
        accum += transform_single_delta(delta, "DARE", dare_drop=p)
    mean = accum / 200
    assert (mean - delta).abs().mean() / delta.abs().mean() < 0.06


def test_transform_expert_set_ties():
    deltas = [torch.randn(4, 4), torch.randn(4, 4)]
    merged = transform_expert_set(deltas, "TIES", trim_fraction=0.2)
    assert merged.shape == deltas[0].shape


def test_transform_expert_set_fisher_requires_fisher():
    deltas = [torch.randn(4, 4), torch.randn(4, 4)]
    with pytest.raises(ValueError, match="fisher_diagonals"):
        transform_expert_set(deltas, "FISHER")


def test_cross_expert_ops_set():
    assert "TIES" in CROSS_EXPERT_OPS
    assert "FISHER" in CROSS_EXPERT_OPS
    assert "RAW" not in CROSS_EXPERT_OPS


def test_single_expert_ops_set():
    assert "RAW" in SINGLE_EXPERT_OPS
    assert "DARE" in SINGLE_EXPERT_OPS
    assert "PROJECT" in SINGLE_EXPERT_OPS
    assert "NORMALIZED" in SINGLE_EXPERT_OPS
    assert "TIES" not in SINGLE_EXPERT_OPS


# Phase 10: layer descriptors


def test_normalize_raw_is_identity():
    delta = torch.randn(4, 4)
    assert torch.equal(normalize_raw(delta), delta)


def test_normalize_unit_frobenius():
    delta = torch.randn(4, 4) * 5
    result = normalize_unit_frobenius(delta)
    assert abs(result.norm().item() - 1.0) < 1e-4


def test_normalize_base_relative_returns_scalar():
    delta = torch.randn(4, 4) * 2
    base = torch.randn(4, 4) * 10
    ratio = normalize_base_relative(delta, base)
    assert ratio.dim() == 0  # scalar
    expected = delta.norm() / base.norm()
    assert abs(ratio.item() - expected.item()) < 1e-4


def test_normalize_median_expert_relative_returns_scalar():
    delta = torch.randn(4, 4) * 3
    norms = [2.0, 3.0, 4.0]  # median = 3.0
    ratio = normalize_median_expert_relative(delta, norms)
    assert ratio.dim() == 0
    expected = delta.norm() / 3.0
    assert abs(ratio.item() - expected.item()) < 1e-4


def test_normalize_clipped_norm_no_clip_when_below():
    torch.manual_seed(0)
    delta = torch.randn(2, 2) * 0.1  # small norm, definitely < 1.0
    assert delta.norm().item() < 1.0
    result = normalize_clipped_norm(delta, max_norm=1.0)
    assert torch.equal(result, delta)


def test_normalize_clipped_norm_clips_when_above():
    delta = torch.randn(4, 4) * 10  # norm >> 1.0
    result = normalize_clipped_norm(delta, max_norm=1.0)
    assert result.norm().item() <= 1.0 + 1e-4
