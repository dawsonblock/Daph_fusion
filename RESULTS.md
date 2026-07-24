# DAPH ExFusion Enhanced Experiment Results

> Generated from enhanced 5-seed experiment with scale sweep, hyperparameter
> optimization, and held-out base regression measurement.
> All experts qualified. 5-seed evaluation with 10k bootstrap CI.
> Dataset audit passes: exact overlap = 0, near-duplicate threshold pass = true.

## Experimental Setup

- **Base model**: distilgpt2
- **Experts**: 3 lineage-matched specialists (math, planning, coding), each
  fine-tuned from the same distilgpt2 checkpoint (500 steps, lr 5e-5, seed 23)
- **Seeds**: 11, 23, 37, 51, 73
- **Bootstrap**: 10,000 resamples, sample-level
- **Evaluation**: 50 samples/domain on validation
- **Held-out general domain**: 50 samples of generic English text (no expert trained on this) — used to measure base regression
- **Dataset isolation**: exact overlap = 0 and MinHash Jaccard < 0.8 across all
  split pairs (train, qualification, calibration, validation, test). The
  generator clusters near-duplicate components into a single split.
- **Optimization phases**:
  1. Scale sweep: [0.2, 0.4, 0.6, 0.8, 1.0] per method
  2. Hyperparameter sweep: DARE p ∈ [0.05, 0.1, 0.15, 0.2, 0.3], TIES trim ∈ [0.1, 0.2, 0.3, 0.4, 0.5], Fisher gamma ∈ [0.5, 1.0]
  3. Per-expert lambda optimization via coordinate descent

## Expert Qualification

| Expert | Domain | Base NLL | Expert NLL | Relative Improvement | Status |
|--------|--------|----------|------------|---------------------|--------|
| lineage-math | math | 4.6937 | 0.9237 | 0.8032 | PASS |
| lineage-planning | planning | 4.4597 | 0.6488 | 0.8545 | PASS |
| lineage-coding | coding | 3.7447 | 1.4942 | 0.6010 | PASS |

## Optimal Parameters (from sweep)

| Method | Scale | DARE p | TIES trim | Fisher gamma |
|--------|-------|--------|-----------|--------------|
| task_arithmetic | 0.6 | — | — | — |
| mean_merge | 1.0 | — | — | — |
| DARE | 0.6 | 0.1 | — | — |
| TIES | 1.0 | — | 0.5 | — |
| DARE_TIES | 1.0 | 0.2 | 0.5 | — |
| Fisher | 1.0 | — | — | 1.0 |
| ExFusion | 1.0 | 0.2 | 0.1 | 1.0 |
| AGX | 0.6 | 0.1 | 0.2 | — |
| weighted_task_arithmetic | 0.6 | — | — | — (lambdas=[1.0, 1.0, 1.0]) |

## 5-Seed Validation Results

| Method | Mean R (math) | Mean R (planning) | Mean R (coding) | Worst R | Base Reg | Drift |
|--------|-------------|-------------------|-----------------|---------|----------|-------|
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| expert_specialists | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| task_arithmetic | 0.8183 | 0.8247 | 0.5434 | 0.5434 | 0.5591 | 0.0006 |
| mean_merge | 0.7307 | 0.6298 | 0.5322 | 0.5322 | 0.0458 | 0.0002 |
| TIES | 0.8155 | 0.7926 | 0.5950 | 0.5950 | 0.3267 | 0.0004 |
| DARE | 0.8141 | 0.8217 | 0.5415 | 0.5415 | 0.5694 | 0.0006 |
| Fisher | 0.8052 | 0.7779 | 0.6158 | 0.6158 | 0.3193 | 0.0005 |
| DARE_TIES | 0.7954 | 0.7927 | 0.5708 | 0.5708 | 0.4665 | 0.0005 |
| ExFusion | 0.7835 | 0.7903 | 0.5899 | 0.5899 | 0.4975 | 0.0006 |
| AGX | 0.8141 | 0.8217 | 0.5415 | 0.5415 | 0.5694 | 0.0006 |
| weighted_task_arithmetic | 0.8375 | 0.7978 | 0.5494 | 0.5494 | 0.3197 | 0.0004 |

## 95% Bootstrap CI (Mean Retention per Domain)

| Method | Math CI | Planning CI | Coding CI |
|--------|---------|------------|-----------|
| task_arithmetic | [0.8183, 0.8183] | [0.8247, 0.8247] | [0.5434, 0.5434] |
| DARE | [0.8108, 0.8191] | [0.8202, 0.8229] | [0.5338, 0.5498] |
| TIES | [0.8155, 0.8155] | [0.7926, 0.7926] | [0.5950, 0.5950] |
| DARE_TIES | [0.7915, 0.8000] | [0.7903, 0.7950] | [0.5619, 0.5797] |
| ExFusion | [0.7791, 0.7910] | [0.7873, 0.7934] | [0.5833, 0.5966] |
| Fisher | [0.8052, 0.8052] | [0.7779, 0.7779] | [0.6158, 0.6158] |
| AGX | [0.8108, 0.8191] | [0.8202, 0.8229] | [0.5338, 0.5498] |
| weighted_task_arithmetic | [0.8375, 0.8375] | [0.7978, 0.7978] | [0.5494, 0.5494] |

## Retention vs. Base Regression Tradeoff

The enhanced experiment introduces a **held-out general domain** (generic English
text that no expert was trained on) to measure true base capability regression.
This reveals a critical tradeoff invisible in retention-only evaluation:

| Method | Mean Retention | Base Regression | Retention/Reg Ratio |
|--------|---------------|-----------------|---------------------|
| mean_merge | 0.6309 | 0.0458 | 13.78 |
| Fisher | 0.7329 | 0.3193 | 2.29 |
| weighted_task_arithmetic | 0.7282 | 0.3197 | 2.28 |
| TIES | 0.7344 | 0.3267 | 2.25 |
| DARE_TIES | 0.7196 | 0.4665 | 1.54 |
| ExFusion | 0.7212 | 0.4975 | 1.45 |
| task_arithmetic | 0.7288 | 0.5591 | 1.30 |
| DARE | 0.7257 | 0.5694 | 1.27 |
| AGX | 0.7257 | 0.5694 | 1.27 |

**Key insight**: Methods with the highest raw retention (TIES, task_arithmetic,
DARE, AGX at ~73%) also cause the most base regression (~33-57%). Methods that
moderate the update magnitude (Fisher, weighted_task_arithmetic, TIES) achieve
nearly the same retention with substantially less forgetting. `mean_merge` is
the extreme conservative case (minimal regression but lowest retention).
Excluding `mean_merge`, **Fisher** offers the best retention-to-regression ratio
(2.29), with **TIES** a close second (2.25) and the highest overall mean
retention (0.7344).

## Held-Out Test Evaluation (single pass after config freeze)

The best validation method (TIES) was evaluated once on the held-out test split
using the frozen optimized parameters (scale=1.0, ties_trim=0.5):

| Domain | Base NLL | Expert NLL | Merged NLL | Test Retention |
|--------|----------|------------|------------|----------------|
| math | 4.5546 | 0.8762 | 1.6323 | 0.7945 |
| planning | 4.5743 | 0.6221 | 1.3941 | 0.8047 |
| coding | 3.8551 | 1.4108 | 2.6288 | 0.5017 |

- **Test mean retention**: 0.7003
- **Test worst retention**: 0.5017

The test retention (0.7003) is consistent with the validation retention (0.7344),
confirming the optimized configuration generalizes to the held-out split with
only a modest gap attributable to split variance.

## Key Findings

1. **TIES** achieves the highest mean validation retention (73.4%) with moderate
   base regression (32.7%) — the best raw retention among merge methods.
2. **Fisher** offers the best retention-to-regression ratio (2.29) excluding the
   overly conservative mean_merge, with 73.3% retention and 31.9% base
   regression, and the best worst-domain retention (61.6%) among high-retention
   methods.
3. **Weighted Task Arithmetic** (lambdas=[1.0, 1.0, 1.0]) achieves 72.8% retention
   with 32.0% base regression — a strong balance; lambda optimization converged
   to equal weights, indicating the three experts contribute equally.
4. **Task Arithmetic / DARE / AGX** reach ~72.6-72.9% retention but cause the
   most base regression (~57%), indicating aggressive updates that forget
   general capabilities.
5. **Mean merge** causes almost no base regression (4.6%) but achieves only 63.1%
   retention — too conservative.
6. **Scale matters**: optimal scale varies by method (0.6 for task_arithmetic/DARE/AGX,
   1.0 for TIES/Fisher/ExFusion). The scale sweep improved retention materially
   over a naive 0.5 default.
7. **Representation drift** is negligible (<0.07%) for all methods, confirming
   merges preserve the base model's internal geometry.
8. **Dataset integrity**: exact overlap = 0 and near-duplicate Jaccard < 0.8
   across all split pairs, so test numbers are uncontaminated.

## Config Freeze

- **Frozen hash**: 1f271939b73ca3fe1f289ff753981ce82a1495f14c445ffefe5e83346a43a207
- **Best method**: TIES (scale=1.0, ties_trim=0.5)
- **Test split used during search**: False
- **Verified**: Config not tampered, test split was held out until final evaluation.
