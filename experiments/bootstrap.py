"""
Bootstrap Confidence Intervals & Paired Difference Testing (Phases 27-28).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def compute_bootstrap_ci(
    metrics: List[float],
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    arr = np.array(metrics)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    boot_means = []
    n = len(arr)
    for _ in range(num_resamples):
        sample = np.random.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    mean_val = float(np.mean(arr))

    return mean_val, lower, upper
