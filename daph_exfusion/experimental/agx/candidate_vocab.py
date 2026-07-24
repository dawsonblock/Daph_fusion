"""Candidate-vocabulary symbolic routing (Phase 18.1).

Reduces symbolic execution complexity from O(BLVH) to O(BLKH) where
K << V is the candidate vocabulary size.

Instead of projecting hidden states against the full vocabulary V,
route symbolic decoding through a small candidate-token set K:

  1. Router predicts symbolic domain
  2. Domain produces candidate-token set K
  3. Project hidden state only against candidate embeddings
  4. Solver transforms candidate IDs

This avoids dense one-hot F.one_hot(solved_ids, V) and vocabulary-scale
projections.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CandidateVocabularyRouter:
    """Routes symbolic decoding through a small candidate token set.

    For each symbolic domain, maintains a set of candidate token IDs (K).
    During forward, projects hidden states only against candidate
    embeddings, reducing complexity from O(BLVH) to O(BLKH).
    """

    def __init__(
        self,
        embedding_layer: nn.Embedding,
        domain_candidates: Dict[str, List[int]],
        default_domain: str = "math",
    ) -> None:
        self.embedding = embedding_layer
        self.domain_candidates = domain_candidates
        self.default_domain = default_domain
        # Pre-compute candidate ID tensors per domain
        self._candidate_tensors: Dict[str, Tensor] = {
            domain: torch.tensor(ids, dtype=torch.long)
            for domain, ids in domain_candidates.items()
        }

    def get_candidates(self, domain: str) -> Tensor:
        """Return candidate token IDs for a domain."""
        if domain not in self._candidate_tensors:
            domain = self.default_domain
        return self._candidate_tensors[domain]

    def project_against_candidates(
        self,
        hidden_states: Tensor,
        domain: str,
    ) -> Tuple[Tensor, Tensor]:
        """Project hidden states against only the candidate embeddings.

        Args:
            hidden_states: [B, L, H]
            domain: symbolic domain name

        Returns:
            candidate_logits: [B, L, K] (logits over candidates only)
            candidate_ids: [K] (the candidate token IDs)
        """
        candidate_ids = self.get_candidates(domain).to(hidden_states.device)
        # Get candidate embeddings: [K, H]
        candidate_embeds = self.embedding(candidate_ids)

        # Project: [B, L, H] x [H, K] -> [B, L, K]
        candidate_logits = torch.matmul(
            hidden_states,
            candidate_embeds.t(),
        )

        return candidate_logits, candidate_ids

    def solve_and_embed(
        self,
        hidden_states: Tensor,
        domain: str,
        solver: Optional[callable] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Solve candidates and produce sparse embeddings.

        Instead of F.one_hot(solved_ids, V), we directly index the
        candidate embeddings, producing a [B, L, H] output without
        ever materializing a V-dimensional one-hot.

        Args:
            hidden_states: [B, L, H]
            domain: symbolic domain
            solver: optional callable(logits, candidate_ids) -> solved_indices

        Returns:
            solved_embeddings: [B, L, H] (embeddings of solved tokens)
            solved_ids: [B, L] (actual token IDs, not one-hot)
        """
        candidate_logits, candidate_ids = self.project_against_candidates(
            hidden_states, domain
        )

        if solver is not None:
            # Custom solver: returns indices into candidate set
            solved_indices = solver(candidate_logits, candidate_ids)
        else:
            # Default: argmax over candidates
            solved_indices = candidate_logits.argmax(dim=-1)  # [B, L]

        # Map candidate indices to actual token IDs
        solved_ids = candidate_ids[solved_indices]  # [B, L]

        # Direct embedding lookup (no one-hot needed)
        solved_embeddings = self.embedding(solved_ids)  # [B, L, H]

        return solved_embeddings, solved_ids

    @property
    def candidate_sizes(self) -> Dict[str, int]:
        """Number of candidates per domain."""
        return {d: len(ids) for d, ids in self.domain_candidates.items()}
