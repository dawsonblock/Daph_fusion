# DAPH ExFusion — Current Status

> **Release: v2.4.0-correctness**
> Authoritative status: see `artifacts/release_status.json`

## What currently works (post-repair)

### P0 — Runtime correctness: PASS

- **Sparse Mamba/SSM**: fixed cross-batch leakage and temporal compression.
  The SSM path now runs on the full `[B,L,H]` sequence with the active mask
  passed into the SSM (freeze or decay semantics), never through sparse
  gather/scatter. See `tests/test_sparse_mamba_correctness.py`.
- **PointwiseSparseDispatch vs SequencePathExecutor**: structural separation
  enforced. Pointwise ops (FFN, CheapPath) may use sparse gather; recurrent
  ops (Mamba, attention) must use the full-sequence executor.
- **FP16/BF16 symbolic priors**: `clamp_symbolic_priors` prevents
  `BIAS_FORCE=1e5` from overflowing FP16 into inf/NaN. Mixed-precision test
  matrix passes for FP32, FP16, BF16.
- **CKA**: token-observation layout `[B,L,H]→[B*L,H]` with padding masking
  and degenerate-case guarding via `MetricResult(valid=False)`.

### P1 — Experimental validity: PARTIAL

- **Expert qualification**: fail-closed for official runs. `QualificationError`
  raised; `DAPH_ALLOW_UNQUALIFIED_EXPERTS` env var no longer overrides the
  official path. Debug mode proceeds but tags artifacts `official=false`.
- **Finiteness guards**: NaN/inf NLL and parameter norms disqualify experts.
- **Dataset isolation audit**: `daph_exfusion/data/dataset_audit.py` checks
  exact and near-duplicate overlap across all 5 splits. Current data has
  exact overlap = 0 but near-duplicate templated samples exist (must be
  re-generated before release).
- **Retention**: single canonical `calculate_retention` with no auto-clip.
  Values >1.0 preserved with `interpretation="merged_outperformed_specialist"`.
- **Artifact validation**: results JSON includes `all_experts_qualified` and
  `all_metrics_valid` flags.

### P2 — Merge algorithm integrity: PASS

- **Merge-baseline semantics**: separate functions for task arithmetic
  (sum, no softmax), mean, softmax-weighted, and convex-weighted merges.
  `_normalize_expert_weights` no longer pre-softmaxes weighted-task-arithmetic.
- **DARE vs delta dropout**: `apply_dare_preprocessing` defaults to
  `rescale=True` (standard DARE). Separate `apply_delta_dropout` for the
  non-rescaled variant. AGX operator names `DARE` and `DELTA_DROPOUT` are
  unambiguous.
- **Fisher**: unified API `build_fisher_diagonal` with `exact_per_sample`
  and `microbatch_approximation` modes. Padding fixed (labels set to -100
  before causal shift).
- **AGX operators**: all 7 operators (RAW, NORMALIZED, DARE, DELTA_DROPOUT,
  TIES, FISHER, PROJECT) have real mathematical contracts. No stubs.
  TIES and FISHER are cross-expert operators via `transform_expert_set`.

### P3 — AGX implementation: PASS (infrastructure)

- **Groupwise search**: `daph_exfusion/search/groupwise.py` with layer-group
  classification and tied-parameter expansion.
- **Pareto scoring**: multi-objective Pareto frontier with feasibility
  filtering and scalar utility.
- **Successive halving**: 4-stage evaluation (quick screen → full val →
  multi-seed → final) to reduce compute.
- **Real surrogate**: `TreeSurrogatePredictor` (ExtraTrees/RF/GBM) that
  actually uses `y` in `fit()`, with CV diagnostics and usability gate.
- **Constrained acquisition**: `constrained_expected_improvement` with
  feasibility probability.
- **Variable-N geometry policy**: permutation-invariant set encoder
  supporting arbitrary expert count.
- **Candidate-vocabulary routing**: `CandidateVocabularyRouter` reduces
  symbolic execution from O(BLVH) to O(BLKH).

### P4 — Statistical validation: PASS (infrastructure)

- **Bootstrap CI**: 10,000 resamples, sample-level.
- **Fixed seeds**: (11, 23, 37, 51, 73).
- **Config freeze**: `freeze_config` + hash verification.
- **Test-split guard**: runtime guard preventing test-split access during
  search.

## What does NOT work yet (blockers for paper-ready)

1. **Lineage-matched experts not trained**: `scripts/train_lineage_experts.py`
   exists but has not been run (requires GPU + training data in
   `data/train/<domain>/`). The current data has no `train` split.
2. **Dataset near-duplicates**: existing data uses templated samples that
   trigger near-duplicate detection. Must be re-generated with diverse text.
3. **5-seed final experiment**: infrastructure ready but not executed
   (depends on qualified experts).
4. **Surrogate trained on real search history**: `TreeSurrogatePredictor`
   is implemented but has no search history to fit on yet.

## Release gate status

```
P0 runtime correctness        PASS
P1 experimental validity      PARTIAL (needs lineage experts + clean data)
P2 merge algorithm integrity  PASS
P3 AGX implementation         PASS (infrastructure; search not yet run)
P4 statistical validation     BLOCKED (needs P1 completion)
```

`paper_ready = false` until all five gates are green.
