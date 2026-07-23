"""
Optuna & Derivative-Free Optimization Engines (Phase 15).
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List

from daph_exfusion.search.candidate import LayerMergeConfig, MergeCandidate


def generate_random_layerwise_candidate(
    num_layers: int,
    num_experts: int,
) -> MergeCandidate:
    operators = ["RAW", "NORMALIZED", "TIES", "DARE", "FISHER", "PROJECT"]
    layer_configs = {}
    for l in range(num_layers):
        op = random.choice(operators)
        lambdas = tuple(
            round(random.uniform(0.05, 0.45), 3) for _ in range(num_experts)
        )
        layer_configs[l] = LayerMergeConfig(
            operator=op,
            lambdas=lambdas,
            ties_trim=0.2,
            dare_drop=0.2,
            fisher_gamma=0.5,
        )
    return MergeCandidate(layer_configs=layer_configs)
