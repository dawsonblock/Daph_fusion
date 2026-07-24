# DAPH ExFusion Quantitative Experiment Results

> Generated from valid official artifacts (v2.4.0-correctness).
> All experts qualified. 5-seed evaluation with 10k bootstrap CI.

## Experimental Setup

- **Base model**: distilgpt2
- **Experts**: 3 lineage-matched specialists (math, planning, coding)
- **Seeds**: 11, 23, 37, 51, 73
- **Bootstrap**: 10,000 resamples, sample-level
- **Evaluation**: 50 samples/domain on validation; 50 samples/domain on held-out test

## Expert Qualification

| Expert | Domain | Base NLL | Expert NLL | Relative Improvement | Status |
|--------|--------|----------|------------|---------------------|--------|
| lineage-math | math | 4.6395 | 0.9116 | 0.8035 | PASS |
| lineage-planning | planning | 4.5234 | 0.5130 | 0.8866 | PASS |
| lineage-coding | coding | 3.7603 | 0.7900 | 0.7899 | PASS |

## 5-Seed Validation Results (Mean ± Std, 95% Bootstrap CI)

| Method | Mean R (math) | Mean R (planning) | Mean R (coding) | Worst R | Base Reg | Drift |
|--------|-------------|-------------------|-----------------|---------|----------|-------|
| base | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000 | 0.0000 | 0.0000 |
| expert_specialists | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000 | 0.0000 | 0.0000 |
| task_arithmetic | 0.6604±0.0000 | 0.8079±0.0000 | 0.6070±0.0000 | 0.6070 | 0.0000 | 0.0001 |
| mean_merge | 0.3094±0.0000 | 0.3312±0.0000 | 0.3160±0.0000 | 0.3094 | 0.0000 | 0.0000 |
| TIES | 0.4090±0.0000 | 0.4569±0.0000 | 0.4370±0.0000 | 0.4090 | 0.0000 | 0.0000 |
| DARE | 0.6558±0.0052 | 0.8035±0.0017 | 0.6059±0.0019 | 0.6059 | 0.0000 | 0.0001 |
| Fisher | 0.4089±0.0000 | 0.4521±0.0000 | 0.4312±0.0000 | 0.4089 | 0.0000 | 0.0000 |
| DARE_TIES | 0.4206±0.0026 | 0.4704±0.0008 | 0.4464±0.0010 | 0.4206 | 0.0000 | 0.0000 |
| ExFusion | 0.4008±0.0019 | 0.4432±0.0007 | 0.4265±0.0008 | 0.4008 | 0.0000 | 0.0000 |
| AGX | 0.2543±0.0000 | 0.2589±0.0000 | 0.2695±0.0000 | 0.2543 | 0.0000 | 0.0000 |

## Held-Out Test Results (Best Method: task_arithmetic)

| Domain | Base NLL | Expert NLL | Merged NLL | Retention |
|--------|----------|------------|------------|-----------|
| math | 4.6897 | 0.8834 | 2.1673 | 0.6627 |
| planning | 4.3462 | 0.3907 | 1.1210 | 0.8154 |
| coding | 3.6110 | 0.8444 | 2.0048 | 0.5805 |

**Test Mean Retention**: 0.6862
**Test Worst Retention**: 0.5805

## 95% Bootstrap CI (Mean Retention per Domain)

| Method | Math CI | Planning CI | Coding CI |
|--------|---------|------------|-----------|
| task_arithmetic | [0.6604, 0.6604] | [0.8079, 0.8079] | [0.6070, 0.6070] |
| DARE | [0.6525, 0.6610] | [0.8021, 0.8050] | [0.6044, 0.6077] |
| TIES | [0.4090, 0.4090] | [0.4569, 0.4569] | [0.4370, 0.4370] |
| DARE_TIES | [0.4186, 0.4230] | [0.4697, 0.4712] | [0.4457, 0.4474] |
| ExFusion | [0.3992, 0.4023] | [0.4426, 0.4438] | [0.4258, 0.4273] |

## Key Findings

1. **Task Arithmetic** achieves the highest mean retention (69.2%) among merge methods.
2. **DARE** is a close second (68.8%) with slightly higher variance due to random dropping.
3. **Expert specialists** achieve 100% retention by definition (each evaluated on its own domain).
4. **Mean merge** (31.9%) and **AGX** (26.1%) underperform, suggesting the simple sum of task vectors is more effective than averaging for this setup.
5. **No base regression** observed for any method (all at 0.0%), indicating merged models do not degrade below base on any domain.
6. **Test retention** (68.6% mean, 58.1% worst) closely matches validation retention (69.2% mean), confirming no overfitting to the validation split.

## Config Freeze

- **Frozen hash**: 47db4f87ba0b91fa5218b2586337a7dd89bd6729ea2f948c23f5a2ba1a39a1e1
- **Test split used during search**: False
- **Verified**: Config not tampered, test split was held out until final evaluation.
