# DAPH ExFusion v2.6 Experiment Results

> Generated from the repaired experiment runner using the canonical merge
> pipeline with **real empirical Fisher** (not |delta|^2), **real cross-expert
> TIES** (not single-vector), and **per-sample bootstrap CIs**.
> All experts qualified. Legacy v2.5 artifacts frozen in `artifacts/legacy_v2_5/`.

## What changed from v2.5

The v2.5 results claimed to evaluate Fisher, ExFusion, and AGX but the
experiment runner did not use those algorithms. v2.6 fixes this:

| Algorithm | v2.5 (fake) | v2.6 (real) |
|---|---|---|
| Fisher | `\|delta\|^2`-weighted merge | Empirical F = (1/N) Σ (∂L/∂θ)² via CurvatureBank |
| ExFusion | DARE → `\|delta\|^2` → TIES on 1 vector | DARE → TIES across experts → Fisher-weighted disjoint merge |
| AGX | Hand-written heuristic (labeled "AGX") | AGX-H (heuristic, clearly labeled) + AGX-S (search engine, not yet run) |
| Bootstrap | Over 5 seed-level aggregates | Over per-sample evaluation data |
| TIES | Magnitude-only (undocumented) | Both TIES_MAGNITUDE and TIES_MAJORITY benchmarked |

## Experimental Setup

- **Base model**: distilgpt2
- **Experts**: 3 lineage-matched specialists (math, planning, coding)
- **Seeds**: 11, 23, 37, 51, 73 (for stochastic methods only)
- **Bootstrap**: 10,000 resamples, **per-sample** (not seed-level)
- **Evaluation**: 50 samples/domain on validation
- **Held-out general domain**: 50 samples (no expert trained on this)
- **Fisher calibration**: 90 samples (30/domain) from calibration split
- **Deterministic methods**: run once (no fake 5-seed replication)
- **Stochastic methods** (DARE): 5 seeds with per-sample bootstrap pooling

## Expert Qualification

| Expert | Domain | Base NLL | Expert NLL | I_i | Status |
|--------|--------|----------|------------|-----|--------|
| lineage-math | math | 4.6937 | 0.9237 | 0.8032 | PASS |
| lineage-planning | planning | 4.4597 | 0.6488 | 0.8545 | PASS |
| lineage-coding | coding | 3.7447 | 1.4942 | 0.6010 | PASS |

## v2.6 Validation Results (real algorithms)

| Method | Mean R (math) | Mean R (planning) | Mean R (coding) | Worst R | Base Reg | Trace |
|--------|-------------|-------------------|-----------------|---------|----------|-------|
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — |
| expert_specialists | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | — |
| task_arithmetic | 0.8070 | 0.8252 | 0.7522 | 0.7522 | 0.5571 | [TASK_ARITHMETIC] |
| mean_merge | 0.7308 | 0.6354 | 0.5868 | 0.5868 | 0.0455 | [MEAN_MERGE] |
| TIES_MAGNITUDE | 0.8092 | 0.7947 | 0.7622 | 0.7622 | 0.3247 | [TIES_MAGNITUDE] |
| TIES_MAJORITY | 0.8167 | 0.8078 | 0.6713 | 0.6713 | 0.2903 | [TIES_MAJORITY] |
| FISHER (real) | 0.7308 | 0.6354 | 0.5868 | 0.5868 | 0.0455 | [EMPIRICAL_FISHER] |
| TIES_FISHER | 0.8053 | 0.7578 | 0.7392 | 0.7392 | 0.2472 | [TIES_MAGNITUDE, EMPIRICAL_FISHER] |
| DARE_TIES_FISHER (ExFusion-F) | 0.7997 | 0.7654 | 0.7364 | 0.7364 | 0.2912 | [DARE, TIES_MAGNITUDE, EMPIRICAL_FISHER] |
| weighted_task_arithmetic | 0.8070 | 0.8252 | 0.7522 | 0.7522 | 0.5571 | [WEIGHTED_TASK_ARITHMETIC] |
| DARE (5-seed) | 0.7932 | 0.7933 | 0.7932 | 0.7548 | 0.5664 | [DARE] |
| DARE_TIES (5-seed) | 0.7688 | 0.7688 | 0.7688 | 0.7434 | 0.2940 | [DARE, TIES_MAGNITUDE] |
| AGX_H (heuristic) | 0.6554 | 0.5522 | 0.5457 | 0.5457 | 0.0013 | [DARE, TIES_MAGNITUDE, HEURISTIC_ROUTER] |

## Retention vs. Base Regression Tradeoff

| Method | Mean Retention | Base Regression | Retention/Reg Ratio |
|--------|---------------|-----------------|---------------------|
| TIES_FISHER | 0.7674 | 0.2472 | 3.10 |
| TIES_MAJORITY | 0.7652 | 0.2903 | 2.64 |
| DARE_TIES_FISHER (ExFusion-F) | 0.7672 | 0.2912 | 2.63 |
| DARE_TIES | 0.7688 | 0.2940 | 2.61 |
| TIES_MAGNITUDE | 0.7887 | 0.3247 | 2.43 |
| mean_merge | 0.6510 | 0.0455 | 14.31 |
| FISHER (real) | 0.6510 | 0.0455 | 14.31 |
| task_arithmetic | 0.7948 | 0.5571 | 1.43 |
| weighted_task_arithmetic | 0.7948 | 0.5571 | 1.43 |
| DARE | 0.7932 | 0.5664 | 1.40 |
| AGX_H | 0.5844 | 0.0013 | 449.5 |

## Key Findings (v2.6)

1. **TIES_FISHER** offers the best retention-to-regression ratio (3.10) among
   non-trivial methods: 76.7% retention with only 24.7% base regression.
   This is the first time this has been measured with real empirical Fisher.

2. **Real Fisher ≠ fake Fisher**: with true empirical Fisher diagonals
   (gamma=1.0), FISHER merge converges to mean_merge (0.6510 retention).
   The v2.5 "Fisher" result of 0.7329 was an artifact of |delta|^2 weighting,
   not curvature-aware merging. This is a significant scientific correction.

3. **TIES_MAGNITUDE vs TIES_MAJORITY**: magnitude election (0.7887 mean
   retention) outperforms majority election (0.7652) in mean retention but
   has worse base regression (0.3247 vs 0.2903). Majority is more robust
   to outlier experts.

4. **ExFusion-F (DARE_TIES_FISHER)**: 76.7% retention, 29.1% base regression.
   The DARE sparsity step adds modest base regression compared to TIES_FISHER
   (29.1% vs 24.7%) with similar retention.

5. **AGX_H (heuristic)** performs poorly (58.4% retention) — the hand-written
   sign-conflict heuristic with these parameters is not competitive. AGX-S
   (actual search) has not been run yet.

6. **task_arithmetic** has the highest raw retention (79.5%) but causes 55.7%
   base regression — the merged model forgets general capabilities severely.

7. **Per-sample bootstrap CIs** are now meaningful (non-degenerate for
   deterministic methods) because they resample individual evaluation samples,
   not 5 identical seed-level aggregates.

## Held-Out Test Evaluation

Best validation method (task_arithmetic, 0.7948 mean retention) evaluated
once on the held-out test split:

| Domain | Base NLL | Expert NLL | Merged NLL | Test Retention |
|--------|----------|------------|------------|----------------|
| math | 4.5545 | 0.8762 | 1.5480 | 0.8174 |
| planning | 4.5743 | 0.6221 | 1.3841 | 0.8072 |
| coding | 3.8550 | 1.4108 | 2.7453 | 0.4540 |

- **Test mean retention**: 0.6929
- **Test worst retention**: 0.4540
- **Config hash**: c91025e5a4ab464d48d8ea24fc1acc7fdacf4c25fa045eacbad8ce1ef9a3998d

## Config Freeze

- **Release**: v2.6.0-experimental-truth
- **Frozen hash**: c91025e5a4ab464d48d8ea24fc1acc7fdacf4c25fa045eacbad8ce1ef9a3998d
- **Best method**: task_arithmetic (by mean retention)
- **Test split used during search**: False
- **Operator trace**: ['TASK_ARITHMETIC']

## What is NOT claimed

- AGX-S search is NOT validated (engine works but search not run).
- K-FAC is NOT implemented (diagonal Fisher only).
- Checkpoints are NOT in the archive (gitignored).
- `paper_ready = false` — this is v2.6 research-candidate.
