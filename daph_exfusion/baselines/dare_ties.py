"""Legacy DARE-TIES operator (v2.5 sparse merge trunk).

DARE → TIES composition. In v3 retained only as a controlled baseline.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from daph_exfusion.baselines.dare import op_dare
from daph_exfusion.baselines.ties import op_ties


def op_dare_ties(
    deltas: List[Tensor],
    drop_probability: float = 0.2,
    trim_fraction: float = 0.2,
    sign_mode: str = "magnitude",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """DARE → TIES: drop → trim → elect → merge."""
    dare_deltas = [
        op_dare(d, drop_probability=drop_probability, generator=generator)
        for d in deltas
    ]
    return op_ties(dare_deltas, trim_fraction=trim_fraction, sign_mode=sign_mode)
