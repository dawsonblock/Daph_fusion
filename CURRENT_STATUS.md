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

### P1 — Experimental validity: PASS

- **Expert qualification**: fail-closed for official runs. `QualificationError`
  raised; `DAPH_ALLOW_UNQUALIFIED_EXPERTS` env var no longer overrides the
  official path. Debug mode proceeds but tags artifacts `official=false`.
  All three lineage-matched specialists (math, planning, coding) trained from
  the same distilgpt2 checkpoint qualify with relative improvement I_i >= 0.60
  (threshold 0.05). See `artifacts/qualification_report.json`.
- **Finiteness guards**: NaN/inf NLL and parameter norms disqualify experts.
- **Dataset isolation audit**: `daph_exfusion/data/dataset_audit.py` checks
  exact and near-duplicate overlap across all 5 splits. Current data has
  exact overlap = 0 AND near_duplicate_threshold_pass = true (MinHash
  Jaccard < 0.8 across all split pairs). The generator
  (`scripts/generate_diverse_data.py`) now clusters near-duplicate components
  into the same split via union-find on MinHash signatures, so templated
  variants never span splits.
- **CI real-data audit**: `scripts/run_ci_gates.py` research gate now runs the
  canonical audit over the actual `data/` directory (not just synthetic temp
  data), failing the gate if `pass_release_gate` is false.
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

None. All five release gates are green and `paper_ready = true` in
`artifacts/release_status.json`. The previously open blockers (no lineage
experts, dataset near-duplicates, no train split, 5-seed experiment not run,
surrogate without search history) have been resolved:

1. **Lineage-matched experts trained**: 3 specialists (math, planning, coding)
   fine-tuned from the same distilgpt2 checkpoint (500 steps, lr 5e-5, seed 23)
   with structured `lineage_manifest.json` provenance.
2. **Dataset near-duplicates eliminated**: the generator now clusters
   near-duplicate components into a single split, and the dataset audit
   passes with `near_duplicate_threshold_pass = true`.
3. **Train split present**: 200 samples/domain across all 5 splits.
4. **5-seed final experiment executed**: seeds (11, 23, 37, 51, 73) with
   10,000-resample bootstrap CIs, scale sweep, hyperparameter optimization,
   per-expert lambda optimization, and held-out general-domain base
   regression measurement. See `artifacts/experiment_results.json` and
   `RESULTS.md`.
5. **Surrogate**: `TreeSurrogatePredictor` remains infrastructure-only; it
   becomes active for acquisition once enough AGX search trajectories are
   logged in `artifacts/geometry_history/`. This is not a release blocker.

## Release gate status

```
P0 runtime correctness        PASS
P1 experimental validity      PASS
P2 merge algorithm integrity  PASS
P3 AGX implementation         PASS
P4 statistical validation     PASS
CI runtime / research / merge / agx   ALL PASS
```

`paper_ready = true`. Final held-out test evaluation (single pass after config
freeze): best method TIES, mean retention 0.7003, worst retention 0.5017.
