# DAPH / ExFusion Implementation-Grade Repair & Research Roadmap

## Overview & Program Governance

This document establishes the official implementation-grade repair and research validation roadmap for **DAPH NeSy-MoE** and **ExFusion Hybrid**. To convert DAPH/ExFusion into a trustworthy, reproducible, and mathematically rigorous research platform, all architectural development follows the strict staged sequence:

$$\boxed{\text{Freeze} \rightarrow \text{Correct} \rightarrow \text{Validate} \rightarrow \text{Measure} \rightarrow \text{Ablate} \rightarrow \text{Optimize} \rightarrow \text{Extend}}$$

---

## Release Sequence & Status Matrix

| Release | Version Focus | Key Deliverables & Scope | Status |
| :--- | :--- | :--- | :--- |
| **Release A** | `v2.3.1-correctness-hotfix` | Legacy artifact freeze (`Phase 0`), `MergeMode` math fixes (`Phase 1`), Retention metric guards (`Phase 2`), Regression suite (`test_v2_3_1_correctness.py`). | **COMPLETED / ACTIVE** |
| **Release B** | `v2.4.0-research-validity` | Expert qualification pipeline (`Phase 3`), 4-layer dataset split (`Phase 4`), Multi-seed RNG runner (`Phase 5`), Validation $\lambda$ selection (`Phase 6`), Full merge ablations (`Phase 7`). | **PLANNED** |
| **Release C** | `v2.5.0-runtime-validity` | Full/chunked/streaming SSM equivalence (`Phase 10`), Explicit path dispatch (`Phase 12`), Router calibration & oracle baselines (`Phase 13`), Benchmark harness rewrite (`Phase 20`). | **PLANNED** |
| **Release D** | `v3.0.0-neurosymbolic` | Real AST/arithmetic/Boolean/SAT parsers (`Phases 15–17`), Tokenizer-safe bridge (`Phase 18`), Verified STE backward surrogates (`Phase 19`). | **PLANNED** |

---

## Immediate P0 Implementation Queue (Release A Completed Scope)

- [x] **P0-01 Legacy Artifact Freeze**: Archived immutable codebase state, environment metadata, SHA-256 file manifest, model manifest, and experiment config in `artifacts/legacy_v2_3/`.
- [x] **P0-02 Baseline Math Correction**: Introduced `MergeMode` enum (`parameter_average`, `task_arithmetic`, `weighted_task_arithmetic`, `logit_weighted`, `full`, `weighted_average`) in `daph_hybrid_exfusion_v2_3.py` ensuring $\Delta_{TA} = N \cdot \Delta_{avg}$ and separating unnormalized logits from explicit coefficients.
- [x] **P0-03 Metric System Overhaul**: Created `research_metrics.py` with `calculate_retention` returning `RetentionResult(valid=False)` when an expert fails to beat base ($G_d \le \epsilon$), preventing invalid retention percentages. Added `compute_pareto_metrics` for multi-objective degradation monitoring.
- [x] **P0-04 Deterministic RNG System**: Implemented `seed_everything(seed)` in `research_metrics.py` returning an explicit `torch.Generator` passed directly to stochastic functions.
- [x] **P0-05 4-Layer Dataset Architecture**: Created directory structure (`data/qualification`, `data/calibration`, `data/validation`, `data/test`) to isolate model qualification, Fisher estimation, hyperparameter selection, and single-pass test evaluation.
- [x] **P0-06 Regression Test Suite**: Added `test_v2_3_1_correctness.py` containing 6 unit tests covering parameter averaging, task arithmetic, $N \times$ average scaling, and invalid retention guards.
