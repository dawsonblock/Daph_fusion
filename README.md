# DAPH ExFusion v3 — Dense Model Merging

> **Production Task Arithmetic merge pipeline with architecture-aware optimization.**
> _Optimized weighted task vectors with constrained general-capability preservation._

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

ExFusion v3 merges multiple fine-tuned specialist models into a single model
that retains specialist capabilities while preserving general language ability.

The production path is **optimized Task Arithmetic**:

```
θ* = θ₀ + α Σᵢ λᵢ Δᵢ
```

where:
- `θ₀` is the base model
- `Δᵢ = θᵢ - θ₀` are task vectors
- `λᵢ` are expert coefficients (optimized)
- `α` is the global scale (optimized)

The search optimizes `(λ₁, ..., λ_N, α)` on a simplex grid with a constrained
objective:

```
max  R_mean(λ, α)
s.t. R_min(λ, α) ≥ τ       (minimum specialist retention)
     G_regression(λ, α) ≤ δ  (maximum general regression)
```

Defaults: `τ = 0.70`, `δ = 0.25`.

---

## Production Merge Modes

| Mode | Formula | Search Space | When to Use |
|------|---------|-------------|-------------|
| **TA-0** | `θ* = θ₀ + α/N Σᵢ Δᵢ` | `α` only | Uniform baseline |
| **TA-1** | `θ* = θ₀ + α Σᵢ λᵢ Δᵢ` | `(λ₁,...,λ_N, α)` | **Default** — weighted TA |
| **TA-2** | `θ*_k = θ₀ + α Σᵢ wᵢ F_{i,k}^γ Δ_{i,k} / Z` | `(w₁,...,w_N, α, γ)` | When TA-1 is exhausted |
| **TA-3** | `θ*_{f,k} = θ₀ + α_f Σᵢ w_{i,f} Δ_{i,k}` | Per-family `(w, α)` | Architecture-aware refinement |

### Search Space Size

For 3 experts at resolution 0.1:
- Simplex grid: 66 λ-combinations
- × 5 α values = **330 configurations**
- Trivial at DistilGPT2 scale

Refinement at 0.025 resolution around the best candidate is automatic.

---

## Quick Start

```python
from daph_exfusion.merge import (
    MergeConfig, MergeMethod, merge_experts_v3,
    search_task_arithmetic,
)

# Simple merge (TA-0)
config = MergeConfig(
    method=MergeMethod.TASK_ARITHMETIC,
    task_scale=0.5,
)
result = merge_experts_v3(base_model, experts, config)

# Optimized merge (TA-1) — grid search over (λ, α)
def evaluator(merged_model):
    # Compute domain retention and general regression
    return {
        "mean_retention": 0.85,
        "min_retention": 0.78,
        "general_regression": 0.15,
        "per_domain_retention": {"math": 0.88, "planning": 0.82, "coding": 0.85},
    }

search_result = search_task_arithmetic(
    base_model, experts, evaluator,
    mode="TA-1",
    resolution=0.1,
    tau=0.70,   # min specialist retention
    delta=0.25, # max general regression
)

print(f"Best: λ={search_result.best.lambdas}, α={search_result.best.scale}")
print(f"Mean retention: {search_result.best.mean_retention:.4f}")
```

---

## Architecture

```
daph_exfusion/
    merge/
        types.py              — Canonical types (MergeConfig, MergeResult, OperatorTrace)
        task_arithmetic.py    — TA merge operator + scale search
        task_search.py        — Simplex grid search (TA-0 through TA-3)
        fisher_dense.py       — Exact empirical Fisher + dense Fisher merge (TA-2)
        pipeline_v3.py        — Single dispatch entry point
        pipeline.py           — v2.5 backward-compatible API
    geometry/
        spectral.py           — SVD diagnostics, spectral gate
        interactions.py       — Fisher interaction matrix, curvature cosine
        activations.py        — Activation covariance bank
        profiler.py           — Geometry profiler (per-group diagnostics)
        representations.py    — CKA, KL, MSE, correlation analysis
    curvature/
        bank.py               — CurvatureBank with provenance snapshots
    validation/
        holdout.py            — Train/validation/test split integrity
        provenance.py         — Experiment manifest
        release_gates.py      — Automatic paper_ready, fisher_verified
        statistics.py         — Bootstrap confidence intervals
    baselines/
        dare.py               — DARE (legacy baseline)
        ties.py               — TIES (legacy baseline)
        dare_ties.py          — DARE-TIES (legacy baseline)
    experimental/
        agx/                  — AGX tournament search (frozen)
        regmean/              — RegMean and RegMean++ (frozen)
        kfac/                 — K-FAC structured merge (frozen)
        surgery/              — Representation surgery (frozen)
        trust_region/         — Trust-region constrained merge (frozen)
        subspace/             — TSV/subspace merge (frozen)
        coefficient_opt/      — Differentiable coefficient optimization (frozen)
    cli/
        main.py               — daph-merge CLI
```

### Design Principles

1. **Single entry point**: All merging goes through `merge_experts()`
2. **Operator trace provenance**: Every merge records exactly which operators ran
3. **Fail-closed Fisher**: Missing curvature data raises `MissingCurvatureError`,
   not silent fallback
4. **Constrained optimization**: Model selection respects both specialist retention
   AND general capability preservation
5. **Experimental isolation**: Premature research modules are frozen under
   `experimental/` until the production TA path is exhausted

---

## Experimental Results

Current validation results (DistilGPT2, 3 experts: math, planning, coding):

| Method | Mean Retention | Worst Domain | General Regression |
|--------|---------------|-------------|-------------------|
| **Task Arithmetic** | **0.795** | 0.752 | 0.557 |
| Weighted TA | 0.795 | 0.752 | 0.557 |
| DARE | 0.793 | 0.755 | 0.573 |
| TIES magnitude | 0.789 | 0.762 | 0.325 |
| DARE-TIES | 0.768 | 0.743 | 0.294 |
| TIES-Fisher | 0.767 | 0.739 | 0.247 |
| Fisher | 0.651 | 0.587 | 0.046 |
| AGX-H | 0.584 | 0.546 | 0.001 |

**Key finding**: Task Arithmetic wins on specialist retention. The optimization
problem is not "maximize retention" but "retain ~80% specialist gains without
destroying general capability."

### Known Issue: Coding Validation→Test Collapse

Coding retention drops from 75.2% (validation) to 45.4% (test).
Root causes identified:
- 8.5-12% prompt-code mismatches in the training data
- Test NLL degradation (merged: 2.745 vs validation: 2.278)

This is the current research priority — see `scripts/investigate_coding_collapse.py`.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -q

# Run only v3 tests
python -m pytest tests/test_v3_*.py -q

# Run with verbose output
python -m pytest tests/ -v
```

Test categories:
- `test_v3_types.py` — Canonical types, enums, task vectors, family classification
- `test_v3_fisher.py` — Exact Fisher, dense merge, base-anchored, stabilization
- `test_v3_regmean.py` — RegMean solver, eligibility, integration
- `test_v3_task_search.py` — Simplex grid, TA-0 through TA-3, constraints
- `test_v3_research_contracts.py` — Operator trace contracts, release gates
- `test_v3_geometry.py` — Spectral, interactions, profiler, activations
- `test_cka_correctness.py` — CKA token-observation layout
- `test_validation_holdout.py` — Split integrity
- `test_experimental_truth.py` — Experimental validity checks

---

## CLI

```bash
# Run a merge experiment
daph-merge run config.yaml

# Compute Fisher diagonals
daph-merge fisher --estimator exact_per_sample --samples 512

# Collect activation covariance
daph-merge activations --mode diagonal

# Run AGX search (experimental)
daph-merge search config.yaml

# Verify release gates
daph-merge verify
```

---

## Provenance & Release Gates

Every merge produces an `OperatorTrace` recording:
- Method name
- Operators executed (e.g., `["TASK_ARITHMETIC"]`)
- Whether Fisher/activation covariance/DARE/TIES was used
- Fisher estimator type (e.g., `exact_per_sample`)
- Config hash for reproducibility

Release gates are automatic:
```python
from daph_exfusion.validation.release_gates import ReleaseGates

gates = ReleaseGates(
    full_tests_pass=True,
    checkpoints_verified=True,
    split_integrity_verified=True,
    # ...
)
print(gates.paper_ready)        # True only if ALL gates pass
print(gates.paper_ready_reason) # Human-readable explanation
```

---

## Legacy

v2.5 sparse merge results are archived under `legacy/v2_5/` with relabeled
method names. The old "Fisher" was delta-squared weighting, "ExFusion" was
DARE-delta2-trim, and "AGX" was a heuristic proxy. Do not cite v2.5 results
as validated.

---

## License

Apache 2.0
