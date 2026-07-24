"""
Adaptive Geometry Merge Candidate Data & Hash (Phase 11, Phase 15).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LayerMergeConfig:
    operator: str
    lambdas: Tuple[float, ...]
    ties_trim: float = 0.2
    dare_drop: float = 0.0
    fisher_gamma: float = 0.5
    sign_mode: str = "magnitude"  # "magnitude" or "majority"


@dataclass(frozen=True)
class MergeCandidate:
    layer_configs: Dict[int, LayerMergeConfig]

    def compute_hash(self) -> str:
        serialized = json.dumps(
            {str(k): asdict(v) for k, v in sorted(self.layer_configs.items())},
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
