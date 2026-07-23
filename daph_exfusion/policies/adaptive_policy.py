"""
Adaptive Geometry Policy Network (Phase 25).
Predicts merge operators and hyperparameters from block descriptors.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGeometryPolicy(nn.Module):

    def __init__(
        self, in_features: int = 8, hidden_dim: int = 32, num_operators: int = 6
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.op_head = nn.Linear(hidden_dim, num_operators)
        self.lambda_head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = F.relu(self.fc1(x))
        op_logits = self.op_head(h)
        lambdas = torch.sigmoid(self.lambda_head(h)) * 0.5
        return {
            "operator_logits": op_logits,
            "lambdas": lambdas,
        }
