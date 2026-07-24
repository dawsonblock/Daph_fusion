"""Adaptive Geometry Policy Network (Phase 17).

Replaces the rigid `lambda_head = Linear(hidden_dim, 3)` (fixed 3 lambdas)
with a permutation-invariant set encoder that supports arbitrary numbers
of experts (variable N).

Architecture:
  1. Encode each expert/layer independently: e_{i,l} = phi(x_{i,l})
  2. Aggregate permutation-invariantly: z_l = rho(sum_i phi(e_{i,l}))
  3. Predict: operator logits, one lambda per expert, operator parameters

Trained from validated search trajectories (candidate geometry -> measured
objectives), NOT synthetic heuristic labels.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ExpertEncoder(nn.Module):
    """Encodes per-expert, per-layer descriptors into a feature vector."""

    def __init__(self, in_features: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [N, in_features] (per-expert descriptors for one layer)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return h  # [N, hidden_dim]


class SetAggregator(nn.Module):
    """Permutation-invariant aggregation via attention (DeepSets-style)."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, expert_embeddings: Tensor) -> Tensor:
        # expert_embeddings: [N, hidden_dim]
        # Attention-weighted sum (permutation-invariant)
        attn_weights = F.softmax(self.attention(expert_embeddings).squeeze(-1), dim=0)
        aggregated = (attn_weights.unsqueeze(-1) * expert_embeddings).sum(dim=0)
        return F.relu(self.fc(aggregated))  # [hidden_dim]


class VariableNGeometryPolicy(nn.Module):
    """Geometry policy supporting arbitrary number of experts (variable N).

    For each layer, takes N expert descriptors and predicts:
      - operator logits (over the operator vocabulary)
      - one lambda per expert (N lambdas, not fixed at 3)
      - operator parameters (ties_trim, dare_drop, fisher_gamma)

    The architecture is permutation-invariant: reordering the expert
    inputs produces the same aggregated representation (up to the
    per-expert lambda outputs which track individual experts).
    """

    def __init__(
        self,
        expert_descriptor_dim: int = 8,
        hidden_dim: int = 64,
        num_operators: int = 7,
    ) -> None:
        super().__init__()
        self.expert_encoder = ExpertEncoder(expert_descriptor_dim, hidden_dim)
        self.aggregator = SetAggregator(hidden_dim)

        # Operator prediction from aggregated representation
        self.op_head = nn.Linear(hidden_dim, num_operators)

        # Per-expert lambda from individual expert embeddings + aggregated context
        self.lambda_head = nn.Linear(hidden_dim * 2, 1)

        # Operator parameters from aggregated representation
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # ties_trim, dare_drop, fisher_gamma
        )

    def forward(
        self,
        expert_descriptors: Tensor,
    ) -> Dict[str, Tensor]:
        """Forward pass for a single layer.

        Args:
            expert_descriptors: [N, D] tensor of per-expert descriptors
                (delta norm, cosine relationships, sign conflict, Fisher
                stats, layer type, depth, base-relative magnitude, etc.)

        Returns:
            Dict with:
              operator_logits: [num_operators]
              lambdas: [N] (one per expert)
              operator_params: [3] (ties_trim, dare_drop, fisher_gamma)
              aggregated_embedding: [hidden_dim] (for policy training)
        """
        N = expert_descriptors.shape[0]

        # Encode each expert independently
        expert_embeddings = self.expert_encoder(expert_descriptors)  # [N, hidden]

        # Aggregate permutation-invariantly
        z = self.aggregator(expert_embeddings)  # [hidden]

        # Operator logits from aggregated representation
        op_logits = self.op_head(z)  # [num_operators]

        # Per-expert lambdas: concat individual embedding with aggregated context
        z_expanded = z.unsqueeze(0).expand(N, -1)  # [N, hidden]
        lambda_input = torch.cat([expert_embeddings, z_expanded], dim=-1)  # [N, 2*hidden]
        lambdas = torch.sigmoid(self.lambda_head(lambda_input).squeeze(-1)) * 0.5  # [N]

        # Operator parameters
        op_params = self.param_head(z)  # [3]
        op_params = torch.sigmoid(op_params)  # bound to [0, 1]
        # Scale: ties_trim in [0, 0.5], dare_drop in [0, 0.5], fisher_gamma in [0, 1]
        op_params_scaled = torch.stack([
            op_params[0] * 0.5,    # ties_trim
            op_params[1] * 0.5,    # dare_drop
            op_params[2],          # fisher_gamma
        ])

        return {
            "operator_logits": op_logits,
            "lambdas": lambdas,
            "operator_params": op_params_scaled,
            "aggregated_embedding": z,
        }


# Backward-compatible alias for the old fixed-N policy
class AdaptiveGeometryPolicy(nn.Module):
    """Legacy fixed-N policy (kept for backward compatibility).

    Prefer VariableNGeometryPolicy for new code. This old policy has a
    fixed lambda_head = Linear(hidden_dim, 3) that only works for exactly
    3 experts.
    """

    def __init__(
        self, in_features: int = 8, hidden_dim: int = 32, num_operators: int = 6
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.op_head = nn.Linear(hidden_dim, num_operators)
        self.lambda_head = nn.Linear(hidden_dim, 3)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = F.relu(self.fc1(x))
        op_logits = self.op_head(h)
        lambdas = torch.sigmoid(self.lambda_head(h)) * 0.5
        return {
            "operator_logits": op_logits,
            "lambdas": lambdas,
        }
