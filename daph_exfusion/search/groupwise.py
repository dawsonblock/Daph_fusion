"""Hierarchical/groupwise AGX search (Phase 12).

Instead of brute-forcing every parameter, search over layer GROUPS first.
Suggested groups: token embeddings, early/middle/late attention, early/
middle/late FFN, norms, LM head, SSM A/B-C/dt/D.

Parameters within a group are initially TIED (same operator, same lambdas).
Only untie individual layers if validation shows the group is sensitive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from daph_exfusion.search.candidate import LayerMergeConfig, MergeCandidate


# Default layer groups for a distilgpt2-style model
DEFAULT_LAYER_GROUPS: Dict[str, List[int]] = {
    "token_embeddings": [0],
    "early_attention": [1, 2],
    "middle_attention": [3, 4],
    "late_attention": [5, 6],
    "early_ffn": [7, 8],
    "middle_ffn": [9, 10],
    "late_ffn": [11, 12],
    "norms": [13],
    "lm_head": [14],
}


@dataclass(frozen=True)
class GroupMergeConfig:
    """Merge configuration for a layer group (tied across layers in group)."""
    operator: str
    lambdas: Tuple[float, ...]
    ties_trim: float = 0.2
    dare_drop: float = 0.2
    fisher_gamma: float = 0.5
    target_scale: float = 1.0


@dataclass
class GroupCandidate:
    """A candidate defined by per-group configs (tied within each group)."""
    group_configs: Dict[str, GroupMergeConfig]

    def to_layer_candidate(
        self,
        layer_groups: Dict[str, List[int]],
    ) -> MergeCandidate:
        """Expand group configs into per-layer configs."""
        layer_configs: Dict[int, LayerMergeConfig] = {}
        for group_name, layer_indices in layer_groups.items():
            gc = self.group_configs.get(group_name)
            if gc is None:
                # Default to RAW if group not configured
                gc = GroupMergeConfig(
                    operator="RAW",
                    lambdas=tuple(0.1 for _ in range(4)),
                )
            for layer_idx in layer_indices:
                layer_configs[layer_idx] = LayerMergeConfig(
                    operator=gc.operator,
                    lambdas=gc.lambdas,
                    ties_trim=gc.ties_trim,
                    dare_drop=gc.dare_drop,
                    fisher_gamma=gc.fisher_gamma,
                )
        return MergeCandidate(layer_configs=layer_configs)


def generate_random_group_candidate(
    layer_groups: Dict[str, List[int]],
    num_experts: int,
    operators: Optional[List[str]] = None,
) -> GroupCandidate:
    """Generate a random candidate with per-group tied parameters."""
    import random

    if operators is None:
        operators = ["RAW", "NORMALIZED", "TIES", "DARE", "FISHER", "PROJECT"]

    group_configs: Dict[str, GroupMergeConfig] = {}
    for group_name in layer_groups:
        op = random.choice(operators)
        lambdas = tuple(
            round(random.uniform(0.05, 0.45), 3) for _ in range(num_experts)
        )
        group_configs[group_name] = GroupMergeConfig(
            operator=op,
            lambdas=lambdas,
            ties_trim=round(random.uniform(0.1, 0.3), 2),
            dare_drop=round(random.uniform(0.1, 0.3), 2),
            fisher_gamma=round(random.uniform(0.3, 0.7), 2),
        )
    return GroupCandidate(group_configs=group_configs)


def classify_layer_group(
    param_name: str,
    layer_idx: int,
    num_layers: int,
) -> str:
    """Classify a parameter into a layer group based on its name and index."""
    name_lower = param_name.lower()
    if "embed" in name_lower and "position" not in name_lower:
        return "token_embeddings"
    if "position" in name_lower or "pos_embed" in name_lower:
        return "positional_embeddings"
    if "lm_head" in name_lower or "output" in name_lower:
        return "lm_head"
    if "norm" in name_lower or "ln" in name_lower:
        return "norms"
    if "ssm" in name_lower or "mamba" in name_lower:
        if "a_log" in name_lower or "a" in name_lower:
            return "ssm_a"
        if "dt" in name_lower:
            return "ssm_dt"
        if "d" in name_lower and "dt" not in name_lower:
            return "ssm_d"
        return "ssm_bc"
    if "attention" in name_lower or "attn" in name_lower or "q_proj" in name_lower or "k_proj" in name_lower or "v_proj" in name_lower:
        third = num_layers // 3
        if layer_idx < third:
            return "early_attention"
        elif layer_idx < 2 * third:
            return "middle_attention"
        return "late_attention"
    if "ffn" in name_lower or "mlp" in name_lower or "intermediate" in name_lower or "fc" in name_lower:
        third = num_layers // 3
        if layer_idx < third:
            return "early_ffn"
        elif layer_idx < 2 * third:
            return "middle_ffn"
        return "late_ffn"
    return "other"
