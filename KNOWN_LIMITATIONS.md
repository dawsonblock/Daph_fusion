# DAPH ExFusion — Known Limitations

Open defects and limitations as of v2.5.0-enhanced.

## Blockers for paper-ready release

**None.** All five release gates (P0-P4) and all four CI gates
(runtime, research, merge, agx) pass. `artifacts/release_status.json` has
`paper_ready = true`. The previously open blockers have been resolved:

### 1. No lineage-matched experts trained — RESOLVED

Three same-lineage specialists (math, planning, coding) are fine-tuned from
the exact same distilgpt2 checkpoint (500 steps, lr 5e-5, seed 23) with
structured `lineage_manifest.json` provenance under `checkpoints/<domain>/`.
All three pass fail-closed qualification with relative improvement
I_i >= 0.60 (threshold 0.05). See `artifacts/qualification_report.json`.

### 2. Dataset near-duplicates — RESOLVED

`scripts/generate_diverse_data.py` now clusters near-duplicate records into
a single split via union-find on MinHash Jaccard signatures (threshold 0.8,
matching the audit), so templated variants never span splits. The dataset
audit reports `exact_overlap_total = 0` and
`near_duplicate_threshold_pass = true`.

### 3. No train split exists — RESOLVED

200 samples/domain are present across all 5 splits
(train, qualification, calibration, validation, test).

### 4. 5-seed final experiment not executed — RESOLVED

Seeds (11, 23, 37, 51, 73) with 10,000-resample sample-level bootstrap CIs,
scale sweep, DARE/TIES/Fisher hyperparameter optimization, per-expert lambda
optimization, and a held-out general-domain base-regression measurement.
See `artifacts/experiment_results.json`, `artifacts/method_statistics.json`,
and `RESULTS.md`. Final held-out test evaluation runs once after config freeze
(`artifacts/test_results.json`).

### 5. Surrogate has no search history — NOT A BLOCKER

`TreeSurrogatePredictor` is implemented and unit-tested. It becomes active
for `constrained_expected_improvement` acquisition once enough AGX search
trajectories are logged in `artifacts/geometry_history/`. Until then it is
infrastructure-only and does not block release.

## Architectural limitations

### SubwordSequenceBridge is CPU-only

`SubwordSequenceBridge` uses Python ThreadPoolExecutor + tokenizer
round-trips. It is classified as:
- `backend = "cpu_compatibility"`
- `compile_safe = false`
- `cuda_graph_safe = false`
- `production_default = false`

For production, use `CandidateVocabularyRouter` instead (reduces symbolic
execution from O(BLVH) to O(BLKH)).

### AGX search not yet run end-to-end

All AGX infrastructure (operators, groupwise search, Pareto, halving,
surrogate, acquisition, geometry policy) is implemented and unit-tested,
but no full end-to-end layerwise/groupwise geometry search has been executed
on the trained experts. The enhanced experiment uses a parameterized AGX
operator selection heuristic; a full `LayerwiseGeometrySearchEngine` run
under CKA representation-drift bounds remains future work.

### Triton selective-scan bindings are GPU-only

`dispatch_selective_scan` routes to `mamba_ssm.ops.selective_scan_fn` only
in CUDA environments. On CPU (and on this machine, which has no CUDA), the
fallback path is used. Mixed-precision (FP16/BF16) state management is
validated in the test suite but is not exercised here because the host lacks
a CUDA device.
