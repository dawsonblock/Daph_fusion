"""Canonical merge pipeline for DAPH ExFusion.

Single entry point for all model merging. Every experiment, AGX candidate,
CLI, and integration test must go through ``merge_experts`` so that algorithm
names and implementations never diverge.
"""
from daph_exfusion.merge.pipeline import (
    ExpertMergeState,
    MergeConfig,
    MergeResult,
    merge_experts,
)

__all__ = ["ExpertMergeState", "MergeConfig", "MergeResult", "merge_experts"]
