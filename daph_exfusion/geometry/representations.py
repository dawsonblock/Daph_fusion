"""Representation-space metrics: CKA, drift, KL, MSE, diagnostics (Phase 9, 16-17).

CKA (Centered Kernel Alignment) measures representational similarity between
two activation tensors. The correct observation layout for [B, L, H] hidden
states is TOKEN-observation: each (batch, position) pair is one observation,
yielding a [B*L, H] matrix.

This module also provides:
  - hidden-state MSE
  - cosine similarity
  - KL divergence
  - representation diagnostics (correlation analysis with general regression)

The representation metric with strongest predictive value becomes a
candidate regularizer. CKA does not receive privileged status just because
it is convenient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# CKA (Centered Kernel Alignment)
# =============================================================================


@dataclass
class MetricResult:
    """Result of a representation-space metric (CKA, drift, etc.).

    `valid=False` means the metric could not be computed meaningfully
    (e.g. fewer than 2 observations after masking). Callers MUST check
    `valid` before using `value`; consuming an invalid result's `value`
    (which is None) is a bug.
    """

    valid: bool
    value: Optional[float] = None
    reason: Optional[str] = None


def _flatten_to_token_observations(
    x: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """Reshape [B, L, H] -> [B*L, H] (token-observation layout).

    If attention_mask [B, L] is provided, only active (mask>0) positions are
    kept. If x is already 2-D [N, H], it is returned as-is (after optional
    masking via a 1-D mask).
    """
    if x.dim() == 2:
        if attention_mask is not None:
            mask = attention_mask.reshape(-1).bool()
            if mask.shape[0] != x.shape[0]:
                raise ValueError(
                    f"1-D mask length {mask.shape[0]} != observations {x.shape[0]}"
                )
            return x[mask]
        return x

    if x.dim() != 3:
        raise ValueError(
            f"Expected 2-D [N,H] or 3-D [B,L,H]; got shape {tuple(x.shape)}"
        )

    B, L, H = x.shape
    if attention_mask is not None:
        if attention_mask.shape != (B, L):
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} != {(B, L)}"
            )
        mask = attention_mask.reshape(-1).bool()
        return x.reshape(-1, H)[mask]

    return x.reshape(-1, H)


def compute_linear_cka(
    x: Tensor,
    y: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> MetricResult:
    """Linear CKA between two activation tensors.

    Observation layout: TOKEN-level. For [B, L, H] inputs, each (batch,
    position) is one observation, yielding [B*L, H]. This is the correct
    layout for measuring representational similarity of hidden states.

    Padding: if attention_mask [B, L] is provided, only active positions are
    used as observations.

    Degenerate cases: if fewer than 2 observations remain after masking, or
    if either tensor has zero variance, returns MetricResult(valid=False)
    rather than a misleading 0.0.

    Returns:
        MetricResult with valid=True and value in [0, 1] when computable,
        or valid=False with a reason string otherwise.
    """
    x_obs = _flatten_to_token_observations(x, attention_mask)
    y_obs = _flatten_to_token_observations(y, attention_mask)

    if x_obs.shape[0] != y_obs.shape[0]:
        raise ValueError(
            f"x and y must have the same number of observations; "
            f"got {x_obs.shape[0]} vs {y_obs.shape[0]}"
        )

    n = x_obs.shape[0]
    if n < 2:
        return MetricResult(valid=False, reason="insufficient_observations")

    # Center
    x_c = x_obs - x_obs.mean(dim=0, keepdim=True)
    y_c = y_obs - y_obs.mean(dim=0, keepdim=True)

    xtx = torch.linalg.norm(x_c.t() @ x_c)
    yty = torch.linalg.norm(y_c.t() @ y_c)
    normalization = xtx * yty

    if normalization.item() == 0.0:
        return MetricResult(valid=False, reason="zero_variance")

    similarity = torch.linalg.norm(x_c.t() @ y_c) ** 2
    value = float((similarity / (normalization + 1e-8)).item())
    # CKA is theoretically in [0, 1]; clamp tiny numerical overshoot.
    value = max(0.0, min(1.0, value))
    return MetricResult(valid=True, value=value)


def compute_linear_cka_value(
    x: Tensor,
    y: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Optional[float]:
    """Return CKA value or None if invalid."""
    result = compute_linear_cka(x, y, attention_mask)
    return result.value


# =============================================================================
# Additional representation metrics (Phase 16-17)
# =============================================================================


def compute_hidden_state_mse(
    h1: Tensor, h2: Tensor, attention_mask: Optional[Tensor] = None
) -> float:
    """Hidden-state MSE between two activation tensors.

    Uses token-observation layout. Padding excluded.
    """
    h1_obs = _flatten_to_token_observations(h1, attention_mask)
    h2_obs = _flatten_to_token_observations(h2, attention_mask)
    if h1_obs.shape != h2_obs.shape:
        return float("inf")
    return float(((h1_obs - h2_obs) ** 2).mean().item())


def compute_cosine_similarity(
    h1: Tensor, h2: Tensor, attention_mask: Optional[Tensor] = None
) -> float:
    """Average cosine similarity between corresponding token observations."""
    h1_obs = _flatten_to_token_observations(h1, attention_mask)
    h2_obs = _flatten_to_token_observations(h2, attention_mask)
    if h1_obs.shape[0] == 0:
        return 0.0
    cos = F.cosine_similarity(h1_obs, h2_obs, dim=-1)
    return float(cos.mean().item())


def compute_kl_divergence(
    logits1: Tensor,
    logits2: Tensor,
    attention_mask: Optional[Tensor] = None,
    temperature: float = 1.0,
) -> float:
    """KL divergence between two logit distributions.

    KL(p1 || p2) where p1 = softmax(logits1/T), p2 = softmax(logits2/T).

    Uses token-observation layout. Padding excluded.
    """
    if logits1.dim() == 3:
        logits1_obs = _flatten_to_token_observations(logits1, attention_mask)
        logits2_obs = _flatten_to_token_observations(logits2, attention_mask)
    else:
        logits1_obs = logits1
        logits2_obs = logits2

    if logits1_obs.shape[0] == 0:
        return 0.0

    p1 = F.log_softmax(logits1_obs / temperature, dim=-1)
    p2 = F.log_softmax(logits2_obs / temperature, dim=-1)
    kl = F.kl_div(p2, p1, log_target=True, reduction="batchmean")
    return float(kl.item())


# =============================================================================
# Representation diagnostics (Phase 16-17)
# =============================================================================


@dataclass
class RepresentationDiagnostics:
    """Per-layer representation diagnostics between base/expert and merged."""
    layer_name: str
    cka_base_merged: Optional[float] = None
    cka_expert_merged: Optional[float] = None
    hidden_state_mse: Optional[float] = None
    cosine_similarity: Optional[float] = None
    kl_base_merged: Optional[float] = None
    kl_expert_merged: Optional[float] = None
    nll_delta: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def compute_all_diagnostics(
    h_base: Tensor,
    h_expert: Tensor,
    h_merged: Tensor,
    logits_base: Optional[Tensor] = None,
    logits_expert: Optional[Tensor] = None,
    logits_merged: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    layer_name: str = "",
) -> RepresentationDiagnostics:
    """Compute all representation diagnostics for a single layer.

    Args:
        h_base: Base model hidden states [B, L, H].
        h_expert: Expert model hidden states.
        h_merged: Merged model hidden states.
        logits_base: Optional base model logits [B, L, V].
        logits_expert: Optional expert logits.
        logits_merged: Optional merged logits.
        attention_mask: Optional [B, L] mask.
        layer_name: Name of the layer.

    Returns:
        RepresentationDiagnostics with all available metrics.
    """
    diag = RepresentationDiagnostics(layer_name=layer_name)

    # CKA
    cka_bm = compute_linear_cka(h_base, h_merged, attention_mask)
    if cka_bm.valid:
        diag.cka_base_merged = cka_bm.value

    cka_em = compute_linear_cka(h_expert, h_merged, attention_mask)
    if cka_em.valid:
        diag.cka_expert_merged = cka_em.value

    # MSE
    diag.hidden_state_mse = compute_hidden_state_mse(h_base, h_merged, attention_mask)

    # Cosine similarity
    diag.cosine_similarity = compute_cosine_similarity(h_base, h_merged, attention_mask)

    # KL divergence (if logits available)
    if logits_base is not None and logits_merged is not None:
        diag.kl_base_merged = compute_kl_divergence(logits_base, logits_merged, attention_mask)
    if logits_expert is not None and logits_merged is not None:
        diag.kl_expert_merged = compute_kl_divergence(logits_expert, logits_merged, attention_mask)

    return diag


@dataclass
class MetricCorrelation:
    """Correlation between a representation metric and general regression."""
    metric_name: str
    correlation: float       # Pearson or Spearman ρ
    p_value: float
    n_samples: int
    predictive: bool         # True if |correlation| > threshold


def compute_metric_correlations(
    diagnostics: List[RepresentationDiagnostics],
    general_regressions: List[float],
    threshold: float = 0.3,
) -> List[MetricCorrelation]:
    """Compute correlations between representation metrics and general regression.

    For each metric (CKA, MSE, cosine, KL), compute the correlation with
    the general-domain NLL regression. The metric with the strongest
    predictive value becomes a candidate regularizer.

    Args:
        diagnostics: List of per-method/per-layer diagnostics.
        general_regressions: List of general regression values (ΔL_general).
        threshold: Minimum |correlation| to be considered predictive.

    Returns:
        List of MetricCorrelation, sorted by absolute correlation.
    """
    if len(diagnostics) != len(general_regressions):
        raise ValueError(
            f"diagnostics count {len(diagnostics)} != regressions {len(general_regressions)}"
        )

    n = len(diagnostics)
    if n < 3:
        return []

    metric_names = [
        "cka_base_merged", "cka_expert_merged",
        "hidden_state_mse", "cosine_similarity",
        "kl_base_merged", "kl_expert_merged",
    ]

    results: List[MetricCorrelation] = []
    regressions = torch.tensor(general_regressions, dtype=torch.float32)

    for metric_name in metric_names:
        values = []
        for d in diagnostics:
            v = getattr(d, metric_name, None)
            if v is not None and v == v:  # not None and not NaN
                values.append(v)
            else:
                values.append(float("nan"))

        values_tensor = torch.tensor(values, dtype=torch.float32)
        valid_mask = ~torch.isnan(values_tensor)

        if valid_mask.sum() < 3:
            continue

        valid_vals = values_tensor[valid_mask]
        valid_reg = regressions[valid_mask]

        if valid_vals.std() < 1e-10 or valid_reg.std() < 1e-10:
            continue

        # Pearson correlation
        v_centered = valid_vals - valid_vals.mean()
        r_centered = valid_reg - valid_reg.mean()
        correlation = float(
            (v_centered * r_centered).sum() /
            (v_centered.norm() * r_centered.norm() + 1e-12)
        )

        results.append(MetricCorrelation(
            metric_name=metric_name,
            correlation=correlation,
            p_value=0.0,  # Would need scipy for proper p-value
            n_samples=int(valid_mask.sum().item()),
            predictive=abs(correlation) > threshold,
        ))

    results.sort(key=lambda x: abs(x.correlation), reverse=True)
    return results


def select_best_regularizer(
    correlations: List[MetricCorrelation],
) -> Optional[str]:
    """Select the representation metric with strongest predictive value.

    Returns the metric name to use as a regularizer, or None if no metric
    is predictive enough.
    """
    predictive = [c for c in correlations if c.predictive]
    if not predictive:
        return None
    return predictive[0].metric_name
