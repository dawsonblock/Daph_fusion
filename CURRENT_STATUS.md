# DAPH ExFusion — Current Status

> **Release: v2.6.0-experimental-truth**
> Authoritative status: see `artifacts/release_status.json`

## What changed in v2.6 (experimental-truth repair)

The v2.5 artifacts claimed `paper_ready=true` but the experiment runner did
not use the algorithms it claimed to evaluate. v2.6 fixes the
experiment-to-implementation disconnect:

1. **Canonical merge API** (`daph_exfusion/merge/pipeline.py`): every merge
   goes through `merge_experts()` with a `MergeConfig` and produces an
   `operator_trace` for provenance. No more inline "almost-TIES" or
   "almost-Fisher" implementations.
2. **Real empirical Fisher**: `CurvatureBank` computes true
   F = (1/N) Σ (∂L/∂θ)² via per-sample gradients on calibration data.
   The old `delta.abs().pow(2)` fake Fisher is removed from the experiment path.
3. **ExFusion = DARE → TIES → Fisher**: `op_ties_fisher` performs TIES sign
   election across experts, then Fisher-weighted disjoint merge among
   survivors. The old single-vector `op_ties([merged])` is eliminated.
4. **Both TIES variants**: `TIES_MAGNITUDE` (magnitude-based election) and
   `TIES_MAJORITY` (pure sign counting) are explicit, benchmarked separately.
5. **AGX cross-expert dispatch fixed**: `apply_layer_merge_operator` now
   calls `transform_expert_set()` for TIES/FISHER/TIES_FISHER, so candidates
   labeled "TIES" actually execute TIES, not RAW.
6. **AGX CKA attention-mask fixed**: `dict.get("attention_mask")` instead of
   `getattr(dict, "attention_mask", None)`.
7. **DARE RNG fixed**: persistent generator consumed continuously across
   parameters, not reset per-parameter.
8. **Per-sample bootstrap**: CIs now resample individual evaluation samples,
   not 5 seed-level aggregates. Deterministic methods run once; stochastic
   methods (DARE) run across 5 seeds.
9. **search_geometry.py**: real `LayerwiseGeometrySearchEngine` execution,
   not hardcoded results.
10. **Legacy artifacts frozen**: `artifacts/legacy_v2_5/` preserves the old
    (incorrect) results with relabeling documentation.

## What currently works

### P0 — Runtime correctness: PASS
- Sparse Mamba/SSM, pointwise/sequence separation, FP16/BF16 priors, CKA.

### P1 — Experimental validity: PARTIAL
- Expert qualification: fail-closed, all 3 experts pass (I_i ≥ 0.60).
- Dataset isolation: exact overlap = 0, near-duplicate threshold pass = true.
- CI real-data audit: passes on actual `data/` directory.
- **Fisher is real**: CurvatureBank with per-sample gradients.
- **ExFusion is real**: DARE → TIES → Fisher via op_ties_fisher.
- **Operator traces**: every merge records its execution path.
- **Blocker**: checkpoints not in archive (gitignored); results need
  independent regeneration.

### P2 — Merge algorithm integrity: PASS
- All 7+ operators (RAW, NORMALIZED, DARE, DELTA_DROPOUT, TIES_MAGNITUDE,
  TIES_MAJORITY, FISHER, TIES_FISHER, DARE_TIES, DARE_TIES_FISHER, PROJECT)
  have real mathematical contracts and are tested for equivalence.

### P3 — AGX implementation: PASS (infrastructure)
- LayerwiseGeometrySearchEngine with correct cross-expert dispatch.
- CKA attention mask fixed.
- search_geometry.py runs real search.
- **Blocker**: AGX-S search not yet run end-to-end on trained experts.

### P4 — Statistical validation: PASS
- Per-sample bootstrap (10,000 resamples).
- Deterministic methods run once (no fake 5-seed replication).
- Stochastic methods (DARE) run across 5 seeds with per-sample bootstrap.
- Config freeze + hash verification + test-split guard.

## What does NOT work yet (blockers for paper-ready)

1. **AGX-S search not executed**: the engine works but no full search has
   been run on the trained experts. AGX-H (heuristic) is benchmarked but
   performs poorly (0.5844 retention).
2. **Checkpoints not in archive**: `checkpoints/` is gitignored. The archive
   cannot independently regenerate results without the trained expert weights.
3. **Results need independent regeneration**: the current numbers were
   produced on one machine; they should be regenerated from published
   checkpoints with full provenance manifests.

## Release gate status

```
P0 runtime correctness        PASS
P1 experimental validity      PARTIAL (checkpoints not in archive)
P2 merge algorithm integrity  PASS
P3 AGX implementation         PASS (infrastructure; AGX-S not run)
P4 statistical validation     PASS
CI runtime / research / merge / agx   ALL PASS (158 tests)
```

`paper_ready = false` — this is v2.6 research-candidate, not paper-ready.
