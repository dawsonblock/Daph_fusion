"""Phase 3: CKA representation-drift metric correctness tests.

Verifies the repaired token-observation layout, padding handling, and
degenerate-case guarding.
"""
import pytest
import torch

from daph_exfusion.geometry.representations import (
    MetricResult,
    compute_linear_cka,
    compute_linear_cka_value,
)


def test_cka_identity_batch1_is_one():
    """CKA(x, x) must be 1.0 even with B=1 (the bug case)."""
    x = torch.randn(1, 8, 16)  # B=1, L=8, H=16
    result = compute_linear_cka(x, x)
    assert result.valid
    assert abs(result.value - 1.0) < 1e-4


def test_cka_identity_multibatch_is_one():
    x = torch.randn(4, 16, 32)
    result = compute_linear_cka(x, x)
    assert result.valid
    assert abs(result.value - 1.0) < 1e-4


def test_cka_identity_2d_input():
    x = torch.randn(50, 16)
    result = compute_linear_cka(x, x)
    assert result.valid
    assert abs(result.value - 1.0) < 1e-4


def test_cka_ignores_padding():
    """Padded positions must be excluded from CKA computation."""
    B, L, H = 2, 8, 16
    x = torch.randn(B, L, H)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]],
        dtype=torch.float32,
    )
    # Pad with arbitrary garbage that would corrupt CKA if included
    x_padded = x.clone()
    x_padded[0, 4:] = 1000.0
    x_padded[1, 6:] = 1000.0

    # With mask, identity should hold (padding ignored)
    result_masked = compute_linear_cka(x, x_padded, attention_mask=mask)
    assert result_masked.valid
    assert abs(result_masked.value - 1.0) < 1e-4, (
        f"Padded positions leaked into CKA: {result_masked.value}"
    )

    # Without mask, identity should NOT hold (padding is included)
    result_unmasked = compute_linear_cka(x, x_padded)
    assert result_unmasked.valid
    assert abs(result_unmasked.value - 1.0) > 1e-3


def test_cka_detects_noisy_features():
    """Adding noise to features should reduce CKA below 1.

    Note: linear CKA is invariant to orthogonal transforms (including
    feature permutations), so we use additive noise to create a real
    representational difference.
    """
    torch.manual_seed(0)
    x = torch.randn(32, 8, 16)
    y = x + 0.5 * torch.randn_like(x)  # add noise
    result = compute_linear_cka(x, y)
    assert result.valid
    assert result.value < 1.0 - 1e-4, f"CKA should be < 1 for noisy features; got {result.value}"


def test_cka_invalid_for_single_observation():
    """A single token observation (B=1, L=1) must return invalid, not 0.0."""
    x = torch.randn(1, 1, 16)
    result = compute_linear_cka(x, x)
    assert not result.valid
    assert result.value is None
    assert "insufficient" in (result.reason or "")


def test_cka_invalid_for_all_padded():
    """All-padded batch has 0 observations -> invalid."""
    x = torch.randn(2, 8, 16)
    mask = torch.zeros(2, 8, dtype=torch.float32)
    result = compute_linear_cka(x, x, attention_mask=mask)
    assert not result.valid
    assert "insufficient" in (result.reason or "")


def test_cka_zero_variance_is_invalid():
    """Constant tensors have zero variance -> invalid."""
    x = torch.full((4, 8, 16), 3.0)
    y = torch.full((4, 8, 16), 5.0)
    result = compute_linear_cka(x, y)
    assert not result.valid
    assert "zero_variance" in (result.reason or "")


def test_cka_orthogonal_features_low_cka():
    """Orthogonal (uncorrelated) feature sets should yield CKA near 0."""
    torch.manual_seed(1)
    x = torch.randn(64, 16)
    y = torch.randn(64, 16)
    result = compute_linear_cka(x, y)
    assert result.valid
    assert result.value < 0.3, f"Orthogonal features CKA unexpectedly high: {result.value}"


def test_cka_mismatched_observations_raises():
    x = torch.randn(10, 16)
    y = torch.randn(20, 16)
    with pytest.raises(ValueError):
        compute_linear_cka(x, y)


def test_cka_value_helper_returns_none_for_invalid():
    x = torch.randn(1, 1, 16)
    assert compute_linear_cka_value(x, x) is None
