"""Statistical validation utilities (Phase 19).

Fixed seeds: 11, 23, 37, 51, 73
For each method: base, experts, task_arithmetic, mean_merge, TIES, DARE,
Fisher, DARE-TIES, ExFusion, AGX

Report: mean, std, 95% bootstrap CI, per-domain NLL, per-domain retention,
worst-domain retention, base regression, representation drift, runtime, VRAM

Bootstrap: 10,000 resamples, sample-level (not seed-mean-level).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# Fixed seeds for reproducibility
FIXED_SEEDS: Tuple[int, ...] = (11, 23, 37, 51, 73)

# Methods to evaluate
BASELINE_METHODS: Tuple[str, ...] = (
    "base",
    "expert_specialists",
    "task_arithmetic",
    "mean_merge",
    "TIES",
    "DARE",
    "Fisher",
    "DARE_TIES",
    "ExFusion",
    "AGX",
)


@dataclass
class SeedResult:
    """Results from a single seed run."""
    seed: int
    method: str
    per_domain_nll: Dict[str, float]
    per_domain_retention: Dict[str, Optional[float]]
    base_regression: float
    repr_drift: float
    runtime_s: float
    vram_mb: float
    valid: bool = True


@dataclass
class MethodStatistics:
    """Aggregated statistics for a method across seeds."""
    method: str
    mean_retention: Dict[str, float]  # per-domain mean
    std_retention: Dict[str, float]   # per-domain std
    ci_retention: Dict[str, Tuple[float, float]]  # per-domain 95% CI
    mean_nll: Dict[str, float]
    std_nll: Dict[str, float]
    worst_domain_retention_mean: float
    base_regression_mean: float
    repr_drift_mean: float
    runtime_mean: float
    vram_mean: float
    n_seeds: int
    n_valid: int


def bootstrap_ci(
    samples: np.ndarray,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Uses sample-level resampling (resample individual observations, not
    just seed means). Returns (lower, upper) bounds of the CI.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if len(samples) == 0:
        return (0.0, 0.0)
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    boot_means = np.array([
        rng.choice(samples, size=n, replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return (lower, upper)


def aggregate_seed_results(
    seed_results: List[SeedResult],
    n_bootstrap: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> MethodStatistics:
    """Aggregate seed results into MethodStatistics with bootstrap CIs."""
    if rng is None:
        rng = np.random.default_rng(42)

    method = seed_results[0].method if seed_results else "unknown"
    valid_results = [r for r in seed_results if r.valid]
    n_valid = len(valid_results)

    # Collect per-domain retention arrays
    domains = set()
    for r in valid_results:
        domains.update(r.per_domain_retention.keys())
    domains = sorted(domains)

    mean_retention: Dict[str, float] = {}
    std_retention: Dict[str, float] = {}
    ci_retention: Dict[str, Tuple[float, float]] = {}
    mean_nll: Dict[str, float] = {}
    std_nll: Dict[str, float] = {}

    for d in domains:
        ret_vals = [
            r.per_domain_retention[d] for r in valid_results
            if r.per_domain_retention.get(d) is not None
            and np.isfinite(r.per_domain_retention[d])
        ]
        if ret_vals:
            arr = np.array(ret_vals)
            mean_retention[d] = float(arr.mean())
            std_retention[d] = float(arr.std())
            ci_retention[d] = bootstrap_ci(arr, n_bootstrap, rng=rng)
        else:
            mean_retention[d] = float("nan")
            std_retention[d] = float("nan")
            ci_retention[d] = (float("nan"), float("nan"))

        nll_vals = [
            r.per_domain_nll.get(d, float("nan")) for r in valid_results
            if np.isfinite(r.per_domain_nll.get(d, float("nan")))
        ]
        if nll_vals:
            arr = np.array(nll_vals)
            mean_nll[d] = float(arr.mean())
            std_nll[d] = float(arr.std())
        else:
            mean_nll[d] = float("nan")
            std_nll[d] = float("nan")

    # Worst-domain retention
    worst_per_seed = []
    for r in valid_results:
        rets = [v for v in r.per_domain_retention.values() if v is not None and np.isfinite(v)]
        if rets:
            worst_per_seed.append(min(rets))
    worst_mean = float(np.mean(worst_per_seed)) if worst_per_seed else float("nan")

    # Other metrics
    base_reg = [r.base_regression for r in valid_results if np.isfinite(r.base_regression)]
    drift = [r.repr_drift for r in valid_results if np.isfinite(r.repr_drift)]
    runtime = [r.runtime_s for r in valid_results if np.isfinite(r.runtime_s)]
    vram = [r.vram_mb for r in valid_results if np.isfinite(r.vram_mb)]

    return MethodStatistics(
        method=method,
        mean_retention=mean_retention,
        std_retention=std_retention,
        ci_retention=ci_retention,
        mean_nll=mean_nll,
        std_nll=std_nll,
        worst_domain_retention_mean=worst_mean,
        base_regression_mean=float(np.mean(base_reg)) if base_reg else float("nan"),
        repr_drift_mean=float(np.mean(drift)) if drift else float("nan"),
        runtime_mean=float(np.mean(runtime)) if runtime else float("nan"),
        vram_mean=float(np.mean(vram)) if vram else float("nan"),
        n_seeds=len(seed_results),
        n_valid=n_valid,
    )


def run_multi_seed_experiment(
    methods: Tuple[str, ...],
    seeds: Tuple[int, ...],
    evaluator: Callable[[str, int], SeedResult],
) -> Dict[str, List[SeedResult]]:
    """Run a multi-seed experiment across all methods.

    Args:
        methods: Method names to evaluate.
        seeds: Random seeds to use.
        evaluator: Function(method_name, seed) -> SeedResult

    Returns:
        Dict mapping method name to list of SeedResults (one per seed).
    """
    results: Dict[str, List[SeedResult]] = {}
    for method in methods:
        method_results: List[SeedResult] = []
        for seed in seeds:
            result = evaluator(method, seed)
            method_results.append(result)
        results[method] = method_results
    return results
