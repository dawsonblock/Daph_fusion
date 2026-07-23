import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from research_metrics import compute_domain_nll, compute_expert_advantage


class QualificationStatus(str, Enum):
    QUALIFIED = "QUALIFIED"
    MARGINAL = "MARGINAL"
    REJECTED = "REJECTED"
    INCOMPATIBLE = "INCOMPATIBLE"


@dataclass(frozen=True)
class QualificationResult:
    expert_id: str
    domain: str
    base_nll: float
    expert_nll: float
    gain: float
    status: QualificationStatus
    reason: str


class ExpertQualificationPipeline:
    """
    Phase 3 Expert Qualification Pipeline.
    Evaluates candidate expert models against the base model on qualification data.
    Ensures models are only approved if G_d = L_base - L_expert > min_gain.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
        min_gain: float = 0.1,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.min_gain = min_gain

    def qualify_expert(
        self,
        expert_id: str,
        expert_model: nn.Module,
        domain: str,
        qualification_texts: List[str],
    ) -> QualificationResult:
        if not qualification_texts:
            return QualificationResult(
                expert_id=expert_id,
                domain=domain,
                base_nll=0.0,
                expert_nll=0.0,
                gain=0.0,
                status=QualificationStatus.INCOMPATIBLE,
                reason="Empty qualification dataset provided.",
            )

        # 1. Evaluate base model NLL on qualification dataset
        base_nll, _ = compute_domain_nll(
            self.base_model,
            self.tokenizer,
            qualification_texts,
            device=self.device,
        )

        # 2. Evaluate candidate expert model NLL on qualification dataset
        expert_nll, _ = compute_domain_nll(
            expert_model,
            self.tokenizer,
            qualification_texts,
            device=self.device,
        )

        # 3. Calculate advantage gain G_d = L_base - L_expert
        gain = compute_expert_advantage(base_nll, expert_nll)

        # 4. Determine qualification status
        if gain > self.min_gain:
            status = QualificationStatus.QUALIFIED
            reason = f"Expert advantage gain ({gain:.4f}) exceeds threshold ({self.min_gain})."
        elif gain > 0.0:
            status = QualificationStatus.MARGINAL
            reason = f"Expert advantage gain ({gain:.4f}) is positive but below threshold ({self.min_gain})."
        else:
            status = QualificationStatus.REJECTED
            reason = f"Expert failed to outperform base model (gain = {gain:.4f})."

        return QualificationResult(
            expert_id=expert_id,
            domain=domain,
            base_nll=float(base_nll),
            expert_nll=float(expert_nll),
            gain=float(gain),
            status=status,
            reason=reason,
        )

    def save_qualification_report(
        self,
        results: List[QualificationResult],
        output_path: str = "data/qualification/expert_qualification.json",
    ) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        serialized = [
            {
                "expert_id": r.expert_id,
                "domain": r.domain,
                "base_nll": r.base_nll,
                "expert_nll": r.expert_nll,
                "gain": r.gain,
                "status": r.status.value,
                "reason": r.reason,
            }
            for r in results
        ]
        with open(output_path, "w") as f:
            json.dump(serialized, f, indent=2)
