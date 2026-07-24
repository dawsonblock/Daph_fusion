"""Canonical Fisher diagonal API (Phase 9 unification).

One public entry point for empirical diagonal Fisher information:

    build_fisher_diagonal(
        model,
        dataset,
        mode="exact_per_sample",
        offload="cpu",
        ignore_padding=True,
    )

Modes:
    exact_per_sample:       F = (1/N) Σ_n g_n²  (rigorous; micro_batch_size=1)
    microbatch_approximation: F_mb = (1/M) Σ_m (mean_{n∈B_m} g_n)²  (approx)

The old duplicate implementations in daph_hybrid_exfusion_v2_3.py and
daph_exfusion/geometry/curvature.py are kept as internal delegates;
this module is the single public API callers should use.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


def build_fisher_diagonal(
    model: nn.Module,
    dataset: Any,
    mode: str = "exact_per_sample",
    offload: str = "cpu",
    ignore_padding: bool = True,
    micro_batch_size: Optional[int] = None,
    forward_fn: Optional[Any] = None,
    loss_fn: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
    use_mmap: bool = False,
    mmap_dir: Optional[str] = None,
) -> Dict[str, Tensor]:
    """Build the empirical diagonal Fisher information matrix.

    Args:
        model: The model to compute Fisher for.
        dataset: Calibration data (StructuredBatch, dict, or Tensor).
        mode: "exact_per_sample" (micro_batch_size=1, rigorous) or
              "microbatch_approximation" (micro_batch_size>1, faster but
              squares the MEAN gradient per micro-batch — NOT exact E[g²]).
        offload: "cpu" to accumulate squared gradients in CPU RAM,
                 "gpu" to accumulate on-device, "mmap" for disk-backed.
        ignore_padding: If True, padding tokens are excluded from the loss
                        via labels[attention_mask==0] = -100.
        micro_batch_size: Override the micro-batch size. If None, defaults
                          to 1 for exact mode and 4 for approximation mode.
        forward_fn: Optional custom forward function.
        loss_fn: Optional custom loss function.
        device: Device for model computation.
        use_mmap: If True and offload="mmap", use memory-mapped files.
        mmap_dir: Directory for mmap files (if use_mmap=True).

    Returns:
        Dict mapping parameter name to diagonal Fisher tensor.
    """
    if mode not in ("exact_per_sample", "microbatch_approximation"):
        raise ValueError(
            f"mode must be 'exact_per_sample' or 'microbatch_approximation'; got '{mode}'"
        )

    if micro_batch_size is None:
        micro_batch_size = 1 if mode == "exact_per_sample" else 4

    if mode == "exact_per_sample" and micro_batch_size != 1:
        raise ValueError(
            "exact_per_sample mode requires micro_batch_size=1; "
            f"got {micro_batch_size}. Use mode='microbatch_approximation' "
            f"for larger micro-batches."
        )

    # Route to the appropriate implementation
    if offload == "mmap":
        from daph_exfusion.geometry.curvature import (
            build_empirical_fisher_diagonals_offloaded,
        )
        return build_empirical_fisher_diagonals_offloaded(
            model,
            dataset,
            micro_batch_size=micro_batch_size,
            use_mmap=True,
            mmap_dir=mmap_dir,
        )
    elif offload == "cpu" and forward_fn is None and loss_fn is None:
        # Use the offloaded-CPU path from curvature.py
        from daph_exfusion.geometry.curvature import (
            build_empirical_fisher_diagonals_offloaded,
        )
        return build_empirical_fisher_diagonals_offloaded(
            model,
            dataset,
            micro_batch_size=micro_batch_size,
            use_mmap=False,
        )
    else:
        # Use the full-featured path from v2_3 (handles forward_fn, loss_fn)
        from daph_hybrid_exfusion_v2_3 import build_empirical_fisher_diagonals
        return build_empirical_fisher_diagonals(
            model,
            dataset,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
            device=device,
            micro_batch_size=micro_batch_size,
            offload_to_cpu=(offload == "cpu"),
        )
