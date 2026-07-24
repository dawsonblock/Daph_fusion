"""
Expert Qualification Pipeline (Phase 1).
Validates that candidate expert models satisfy relative improvement I_i >= 0.05
over base model and possess compatible parameter topology and tokenizers.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from research_metrics import compute_domain_nll, compute_expert_advantage


class QualificationStatus(str, Enum):
    QUALIFIED = "QUALIFIED"
    MARGINAL = "MARGINAL"
    REJECTED = "REJECTED"
    INCOMPATIBLE = "INCOMPATIBLE"


@dataclass(frozen=True)
class ExpertQualification:
    expert_name: str
    expert_revision: str
    domain: str
    base_nll: float
    expert_nll: float
    relative_improvement: float
    architecture_compatible: bool
    tokenizer_compatible: bool
    state_dict_compatible: bool
    passed: bool
    rejection_reason: Optional[str]


class InvalidExperiment(Exception):
    """Raised when one or more experts fail preflight qualification."""

    pass


class QualificationError(InvalidExperiment):
    """Raised when an OFFICIAL experiment path encounters unqualified experts.

    This is the hard-fail exception for the official/research benchmark path.
    Debug-mode runs may proceed past qualification but must tag their
    artifacts as ``official: false``.
    """

    pass


class ExpertQualificationPipeline:
    """Preflight expert qualification pipeline."""

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
        min_expert_improvement: float = 0.05,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.min_expert_improvement = min_expert_improvement

    def check_topology_compatibility(
        self, expert_model: nn.Module
    ) -> Tuple[bool, bool, Optional[str]]:
        base_params = dict(self.base_model.named_parameters())
        expert_params = dict(expert_model.named_parameters())

        if set(base_params.keys()) != set(expert_params.keys()):
            return False, False, "Parameter key mismatch between base and expert."

        for name, bp in base_params.items():
            ep = expert_params[name]
            if bp.shape != ep.shape:
                return (
                    False,
                    False,
                    f"Shape mismatch in parameter '{name}': {bp.shape} vs {ep.shape}.",
                )

        base_cfg = getattr(self.base_model, "config", None)
        expert_cfg = getattr(expert_model, "config", None)
        base_vocab = (
            getattr(base_cfg, "vocab_size", None) if base_cfg is not None else None
        )
        expert_vocab = (
            getattr(expert_cfg, "vocab_size", None) if expert_cfg is not None else None
        )
        if (
            base_vocab is not None
            and expert_vocab is not None
            and base_vocab != expert_vocab
        ):
            return (
                True,
                False,
                f"Vocab size mismatch: base={base_vocab} vs expert={expert_vocab}.",
            )

        return True, True, None

    def qualify_expert(
        self,
        expert_name: str,
        expert_revision: str,
        expert_model: nn.Module,
        domain: str,
        qualification_texts: List[str],
    ) -> ExpertQualification:
        arch_ok, state_ok, compat_reason = self.check_topology_compatibility(
            expert_model
        )
        if not (arch_ok and state_ok):
            return ExpertQualification(
                expert_name=expert_name,
                expert_revision=expert_revision,
                domain=domain,
                base_nll=0.0,
                expert_nll=0.0,
                relative_improvement=0.0,
                architecture_compatible=arch_ok,
                tokenizer_compatible=True,
                state_dict_compatible=state_ok,
                passed=False,
                rejection_reason=compat_reason or "Topology incompatible.",
            )

        if not qualification_texts:
            return ExpertQualification(
                expert_name=expert_name,
                expert_revision=expert_revision,
                domain=domain,
                base_nll=0.0,
                expert_nll=0.0,
                relative_improvement=0.0,
                architecture_compatible=True,
                tokenizer_compatible=True,
                state_dict_compatible=True,
                passed=False,
                rejection_reason="Qualification texts empty.",
            )

        base_nll, _ = compute_domain_nll(
            self.base_model, self.tokenizer, qualification_texts, device=self.device
        )
        expert_nll, _ = compute_domain_nll(
            expert_model, self.tokenizer, qualification_texts, device=self.device
        )

        if base_nll <= 0:
            rel_imp = 0.0
        else:
            rel_imp = (base_nll - expert_nll) / base_nll

        # Finiteness guards: NaN/inf NLL or parameter norm disqualifies the
        # expert regardless of the relative-improvement threshold.
        if not (math.isfinite(base_nll) and math.isfinite(expert_nll)):
            return ExpertQualification(
                expert_name=expert_name,
                expert_revision=expert_revision,
                domain=domain,
                base_nll=float(base_nll),
                expert_nll=float(expert_nll),
                relative_improvement=float(rel_imp) if math.isfinite(rel_imp) else 0.0,
                architecture_compatible=True,
                tokenizer_compatible=True,
                state_dict_compatible=True,
                passed=False,
                rejection_reason=(
                    f"Non-finite NLL: base={base_nll}, expert={expert_nll}."
                ),
            )

        param_norm = sum(
            float(p.detach().float().norm().item()) ** 2
            for p in expert_model.parameters()
        ) ** 0.5
        if not math.isfinite(param_norm):
            return ExpertQualification(
                expert_name=expert_name,
                expert_revision=expert_revision,
                domain=domain,
                base_nll=float(base_nll),
                expert_nll=float(expert_nll),
                relative_improvement=float(rel_imp),
                architecture_compatible=True,
                tokenizer_compatible=True,
                state_dict_compatible=True,
                passed=False,
                rejection_reason=f"Non-finite parameter norm: {param_norm}.",
            )

        passed = rel_imp >= self.min_expert_improvement
        rejection_reason = (
            None
            if passed
            else f"Relative improvement {rel_imp:.4f} < threshold {self.min_expert_improvement}."
        )

        return ExpertQualification(
            expert_name=expert_name,
            expert_revision=expert_revision,
            domain=domain,
            base_nll=float(base_nll),
            expert_nll=float(expert_nll),
            relative_improvement=float(rel_imp),
            architecture_compatible=True,
            tokenizer_compatible=True,
            state_dict_compatible=True,
            passed=passed,
            rejection_reason=rejection_reason,
        )

    def validate_preflight(self, qualifications: List[ExpertQualification]) -> None:
        failed = [q for q in qualifications if not q.passed]
        if failed:
            reasons = "; ".join(
                [f"{q.expert_name} ({q.domain}): {q.rejection_reason}" for q in failed]
            )
            raise InvalidExperiment(
                f"Merge search prohibited: one or more source experts failed qualification. Details: {reasons}"
            )
