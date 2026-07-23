"""
Checkpoint & State Dict Compatibility Diagnostics (Phase 2).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def audit_checkpoint_compatibility(
    base_model: nn.Module,
    expert_model: nn.Module,
) -> Dict[str, Any]:
    base_params = dict(base_model.named_parameters())
    expert_params = dict(expert_model.named_parameters())

    missing_in_expert = set(base_params.keys()) - set(expert_params.keys())
    extra_in_expert = set(expert_params.keys()) - set(base_params.keys())

    shape_mismatches = {}
    for k in set(base_params.keys()) & set(expert_params.keys()):
        if base_params[k].shape != expert_params[k].shape:
            shape_mismatches[k] = {
                "base": tuple(base_params[k].shape),
                "expert": tuple(expert_params[k].shape),
            }

    compatible = not missing_in_expert and not extra_in_expert and not shape_mismatches

    return {
        "compatible": compatible,
        "missing_keys": sorted(list(missing_in_expert)),
        "extra_keys": sorted(list(extra_in_expert)),
        "shape_mismatches": shape_mismatches,
    }
