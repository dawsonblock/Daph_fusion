"""Phase 19-20: Statistical validation + holdout guard tests."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from daph_exfusion.validation.statistics import (
    FIXED_SEEDS,
    BASELINE_METHODS,
    SeedResult,
    bootstrap_ci,
    aggregate_seed_results,
    run_multi_seed_experiment,
)
from daph_exfusion.validation.holdout import (
    FrozenConfig,
    compute_config_hash,
    freeze_config,
    verify_config_frozen,
    TestSplitGuard,
)


def test_fixed_seeds():
    assert FIXED_SEEDS == (11, 23, 37, 51, 73)
    assert len(FIXED_SEEDS) == 5


def test_baseline_methods_complete():
    expected = {"base", "task_arithmetic", "mean_merge", "TIES", "DARE",
                "Fisher", "DARE_TIES", "ExFusion", "AGX", "expert_specialists"}
    assert set(BASELINE_METHODS) == expected


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(5.0, 1.0, 100)
    lower, upper = bootstrap_ci(data, n_resamples=1000, rng=rng)
    assert lower < 5.0 < upper
    assert upper - lower < 1.0  # tight CI for n=100


def test_bootstrap_ci_empty():
    lower, upper = bootstrap_ci(np.array([]))
    assert lower == 0.0 and upper == 0.0


def test_aggregate_seed_results():
    results = [
        SeedResult(seed=11, method="AGX",
                   per_domain_nll={"math": 2.0, "planning": 3.0},
                   per_domain_retention={"math": 0.8, "planning": 0.6},
                   base_regression=0.05, repr_drift=0.1, runtime_s=10, vram_mb=100),
        SeedResult(seed=23, method="AGX",
                   per_domain_nll={"math": 2.1, "planning": 3.1},
                   per_domain_retention={"math": 0.85, "planning": 0.65},
                   base_regression=0.06, repr_drift=0.12, runtime_s=11, vram_mb=102),
    ]
    stats = aggregate_seed_results(results, n_bootstrap=100)
    assert stats.method == "AGX"
    assert abs(stats.mean_retention["math"] - 0.825) < 1e-4
    assert abs(stats.mean_nll["math"] - 2.05) < 1e-4
    assert stats.n_valid == 2


def test_run_multi_seed_experiment():
    def evaluator(method, seed):
        return SeedResult(
            seed=seed, method=method,
            per_domain_nll={"math": 2.0 + seed * 0.01},
            per_domain_retention={"math": 0.8 + seed * 0.001},
            base_regression=0.05, repr_drift=0.1, runtime_s=10, vram_mb=100,
        )
    results = run_multi_seed_experiment(("AGX",), (11, 23), evaluator)
    assert len(results["AGX"]) == 2
    assert results["AGX"][0].seed == 11


# Phase 20: holdout


def test_freeze_config_writes_file():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "final_config.json"
        frozen = freeze_config(
            {"operator": "TIES", "lambdas": [0.3, 0.4]},
            output_path=path,
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["config_hash"] == frozen.config_hash
        assert data["test_split_used_during_search"] is False


def test_verify_config_frozen_passes():
    config = {"a": 1, "b": 2}
    h = compute_config_hash(config)
    frozen = FrozenConfig(
        release="test", config_hash=h, frozen_at="now",
        config=config, test_split_used_during_search=False,
    )
    verify_config_frozen(frozen)  # should not raise


def test_verify_config_frozen_detects_tampering():
    frozen = FrozenConfig(
        release="test", config_hash="wrong_hash", frozen_at="now",
        config={"a": 1}, test_split_used_during_search=False,
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_config_frozen(frozen)


def test_verify_config_frozen_detects_test_contamination():
    config = {"a": 1}
    h = compute_config_hash(config)
    frozen = FrozenConfig(
        release="test", config_hash=h, frozen_at="now",
        config=config, test_split_used_during_search=True,
    )
    with pytest.raises(RuntimeError, match="contaminated"):
        verify_config_frozen(frozen)


def test_test_split_guard_blocks_test_access():
    guard = TestSplitGuard()
    guard.enter_search_mode()
    with pytest.raises(RuntimeError, match="Test split accessed"):
        guard.check_access("test")
    assert guard.test_was_accessed_during_search


def test_test_split_guard_allows_other_splits():
    guard = TestSplitGuard()
    guard.enter_search_mode()
    guard.check_access("validation")  # should not raise
    guard.check_access("calibration")  # should not raise


def test_test_split_guard_allows_test_after_search():
    guard = TestSplitGuard()
    guard.enter_search_mode()
    guard.exit_search_mode()
    guard.check_access("test")  # should not raise after search ends
