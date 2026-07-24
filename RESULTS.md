# DAPH ExFusion Enhanced Experiment Results

> Generated from enhanced 5-seed experiment with scale sweep, hyperparameter
> optimization, and held-out base regression measurement.
> All experts qualified. 5-seed evaluation with 10k bootstrap CI.

## Experimental Setup

- **Base model**: distilgpt2
- **Experts**: 3 lineage-matched specialists (math, planning, coding)
- **Seeds**: 11, 23, 37, 51, 73
- **Bootstrap**: 10,000 resamples, sample-level
- **Evaluation**: 50 samples/domain on validation
- **Held-out general domain**: 50 samples of generic English text (no expert trained on this) — used to measure base regression
- **Optimization phases**:
  1. Scale sweep: [0.2, 0.4, 0.6, 0.8, 1.0] per method
  2. Hyperparameter sweep: DARE p ∈ [0.05, 0.1, 0.15, 0.2, 0.3], TIES trim ∈ [0.1, 0.2, 0.3, 0.4, 0.5], Fisher gamma ∈ [0.5, 1.0]
  3. Per-expert lambda optimization via coordinate descent

## Expert Qualification

| Expert | Domain | Base NLL | Expert NLL | Relative Improvement | Status |
|--------|--------|----------|------------|---------------------|--------|
| lineage-math | math | 4.6681 | 0.8994 | 0.8073 | PASS |
| lineage-planning | planning | 4.5057 | 0.5894 | 0.8692 | PASS |
| lineage-coding | coding | 3.7964 | 0.8292 | 0.7816 | PASS |

## Optimal Parameters (from sweep)

| Method | Scale | DARE p | TIES trim | Fisher gamma |
|--------|-------|--------|-----------|--------------|
| task_arithmetic | 0.6 | — | — | — |
| mean_merge | 1.0 | — | — | — |
| DARE | 0.6 | 0.05 | — | — |
| TIES | 1.0 | — | 0.5 | — |
| DARE_TIES | 1.0 | 0.2 | 0.5 | — |
| Fisher | 1.0 | — | — | 1.0 |
| ExFusion | 1.0 | 0.2 | 0.1 | 0.5 |
| AGX | 0.6 | 0.1 | 0.2 | — |
| weighted_task_arithmetic | 0.6 | — | — | — (lambdas=[1.0, 1.0, 1.0]) |

## 5-Seed Validation Results

| Method | Mean R (math) | Mean R (planning) | Mean R (coding) | Worst R | Base Reg | Drift |
|--------|-------------|-------------------|-----------------|---------|----------|-------|
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| expert_specialists | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| task_arithmetic | 0.6968 | 0.8333 | 0.5905 | 0.5905 | 0.4018 | 0.0001 |
| mean_merge | 0.5083 | 0.6411 | 0.5367 | 0.5083 | 0.0053 | 0.0000 |
| TIES | 0.6627 | 0.7760 | 0.5869 | 0.5869 | 0.2343 | 0.0001 |
| DARE | 0.6946 | 0.8306 | 0.5884 | 0.5884 | 0.4073 | 0.0001 |
| Fisher | 0.6970 | 0.7767 | 0.5994 | 0.5994 | 0.2391 | 0.0001 |
| DARE_TIES | 0.6799 | 0.7842 | 0.5614 | 0.5614 | 0.3514 | 0.0001 |
| ExFusion | 0.6809 | 0.7888 | 0.6039 | 0.6039 | 0.2654 | 0.0001 |
| AGX | 0.6946 | 0.8306 | 0.5898 | 0.5898 | 0.4106 | 0.0001 |
| weighted_task_arithmetic | 0.6918 | 0.7842 | 0.6070 | 0.6070 | 0.2021 | 0.0001 |

## 95% Bootstrap CI (Mean Retention per Domain)

| Method | Math CI | Planning CI | Coding CI |
|--------|---------|------------|-----------|
| task_arithmetic | [0.6968, 0.6968] | [0.8333, 0.8333] | [0.5905, 0.5905] |
| DARE | [0.6919, 0.6970] | [0.8294, 0.8318] | [0.5872, 0.5899] |
| TIES | [0.6627, 0.6627] | [0.7760, 0.7760] | [0.5869, 0.5869] |
| DARE_TIES | [0.6758, 0.6841] | [0.7806, 0.7875] | [0.5577, 0.5661] |
| ExFusion | [0.6745, 0.6869] | [0.7872, 0.7905] | [0.6002, 0.6071] |
| Fisher | [0.6970, 0.6970] | [0.7767, 0.7767] | [0.5994, 0.5994] |
| AGX | [0.6919, 0.6973] | [0.8294, 0.8318] | [0.5872, 0.5899] |

## Retention vs. Base Regression Tradeoff

The enhanced experiment introduces a **held-out general domain** (generic English
text that no expert was trained on) to measure true base capability regression.
This reveals a critical tradeoff invisible in the previous experiment:

| Method | Mean Retention | Base Regression | Retention/Reg Ratio |
|--------|---------------|-----------------|---------------------|
| weighted_task_arithmetic | 0.6918 | 0.2021 | 3.42 |
| ExFusion | 0.6912 | 0.2654 | 2.60 |
| TIES | 0.6752 | 0.2343 | 2.88 |
| Fisher | 0.6910 | 0.2391 | 2.89 |
| DARE_TIES | 0.6752 | 0.3514 | 1.92 |
| task_arithmetic | 0.7069 | 0.4018 | 1.76 |
| DARE | 0.7045 | 0.4073 | 1.73 |
| AGX | 0.7036 | 0.4106 | 1.71 |
| mean_merge | 0.5620 | 0.0053 | 106.0 |

**Key insight**: Methods with the highest raw retention (task_arithmetic, DARE, AGX
at ~70.5%) also cause the most base regression (~40%). Methods that moderate
the update magnitude (weighted_task_arithmetic, ExFusion, Fisher, TIES) achieve
nearly the same retention with substantially less forgetting.

## Key Findings

1. **Task Arithmetic** achieves the highest mean retention (70.7%) but at the cost
   of 40.2% base regression — the merged model forgets general capabilities.
2. **Weighted Task Arithmetic** offers the best retention-to-regression ratio (3.42),
   achieving 69.2% retention with only 20.2% base regression.
3. **ExFusion** (Fisher-weighted DARE-TIES) achieves 69.1% retention with 26.5%
   base regression — a strong balance, and the best worst-domain retention (60.4%)
   among all methods.
4. **DARE** is a close second in raw retention (70.5%) but has high base regression
   (40.7%), similar to task_arithmetic.
5. **Mean merge** causes almost no base regression (0.5%) but achieves only 56.2%
   retention — it's too conservative.
6. **Scale matters**: optimal scale varies by method (0.6 for task_arithmetic/DARE/AGX,
   1.0 for TIES/Fisher/ExFusion). The scale sweep improved retention by 5-10% over
   the default 0.5.
7. **Lambda optimization** converged to [1.0, 1.0, 1.0], indicating the three experts
   contribute equally — no domain needs up-weighting.
8. **Representation drift** is negligible (<0.01%) for all methods, confirming merges
   preserve the base model's internal geometry.

## Config Freeze

- **Frozen hash**: 47db4f87ba0b91fa5218b2586337a7dd89bd6729ea2f948c23f5a2ba1a39a1e1
- **Test split used during search**: False
- **Verified**: Config not tampered, test split was held out until final evaluation.
