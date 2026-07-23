"""
Geometry Normalization Transformations (Phase 10).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor


def normalize_layerwise_task_vectors(
    task_vectors: List[Dict[str, Tensor]],
    base_model: torch.nn.Module,
    epsilon: float = 1e-8,
) -> List[Dict[str, Tensor]]:
    normalized = [{} for _ in task_vectors]

    keys = task_vectors[0].keys() if task_vectors else []
    for k in keys:
        norms = [torch.norm(tv[k]).item() for tv in task_vectors]
        mean_norm = sum(norms) / len(norms) if norms else 0.0

        for idx, tv in enumerate(task_vectors):
            vec = tv[k]
            v_norm = torch.norm(vec).item()
            if v_norm > 0:
                scale = mean_norm / (v_norm + epsilon)
                normalized[idx][k] = vec * scale
            else:
                normalized[idx][k] = vec.clone()

    return normalized
