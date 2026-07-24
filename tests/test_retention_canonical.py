"""Phase 6: Canonical retention metric tests."""
import math
import pytest

from research_metrics import (
    calculate_retention,
    compute_expert_advantage,
    RetentionResult,
)


def test_retention_not_clipped_above_one():
    """R > 1.0 (merged outperformed specialist) must be preserved, not clipped."""
    res = calculate_retention(base_loss=5.0, expert_loss=3.0, merged_loss=2.0)
    assert res.valid
    assert res.value == 1.5  # (5-2)/(5-3) = 1.5
    assert res.interpretation == "merged_outperformed_specialist"


def test_retention_negative_when_merged_worse_than_base():
    res = calculate_retention(base_loss=5.0, expert_loss=3.0, merged_loss=6.0)
    assert res.valid
    assert res.value == -0.5  # (5-6)/(5-3) = -0.5
    assert res.interpretation == "merged_worse_than_base"


def test_retention_one_when_merged_equals_expert():
    res = calculate_retention(base_loss=5.0, expert_loss=3.0, merged_loss=3.0)
    assert res.valid
    assert abs(res.value - 1.0) < 1e-6
    assert res.interpretation == "merged_matched_specialist"


def test_retention_zero_when_merged_equals_base():
    res = calculate_retention(base_loss=5.0, expert_loss=3.0, merged_loss=5.0)
    assert res.valid
    assert abs(res.value) < 1e-6
    assert res.interpretation == "merged_matched_base"


def test_retention_invalid_when_expert_worse():
    res = calculate_retention(base_loss=5.0, expert_loss=6.0, merged_loss=5.5)
    assert not res.valid
    assert res.value is None
    assert res.reason == "expert_does_not_outperform_base"


def test_retention_invalid_when_expert_equals_base():
    res = calculate_retention(base_loss=5.0, expert_loss=5.0, merged_loss=4.0)
    assert not res.valid
    assert res.reason == "expert_does_not_outperform_base"


def test_retention_partial():
    res = calculate_retention(base_loss=10.0, expert_loss=4.0, merged_loss=7.0)
    assert res.valid
    assert abs(res.value - 0.5) < 1e-6  # (10-7)/(10-4) = 0.5
    assert res.interpretation == "partial_retention"
