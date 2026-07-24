# DAPH ExFusion — Known Limitations

Open defects and limitations as of v2.6.0-experimental-truth.

## Blockers for paper-ready release

### 1. AGX-S search not executed — OPEN

The `LayerwiseGeometrySearchEngine` is repaired (cross-expert dispatch,
CKA mask fix) and `search_geometry.py` runs the real engine. However, no
full end-to-end AGX-S search has been run on the trained experts. AGX-H
(heuristic) is benchmarked but performs poorly (0.5844 retention).

**To resolve**: run `python search_geometry.py --num-candidates 64` and
evaluate the selected candidate on the test split.

### 2. Checkpoints not in archive — OPEN

`checkpoints/` is gitignored (model weights are too large for git). The
archive contains lineage manifests but not the actual safetensors files.
This prevents independent regeneration of results from the ZIP alone.

**To resolve**: publish checkpoints to HuggingFace Hub and record the
model IDs + commit hashes in the provenance manifest.

### 3. Results need independent regeneration — OPEN

Current numbers were produced on one CPU-only machine. They should be
regenerated from published checkpoints with full provenance manifests
including git commit, torch version, training config, and split hashes.

## Resolved in v2.6

### Fake Fisher → real empirical Fisher — RESOLVED

The experiment runner now uses `CurvatureBank` which computes true
F = (1/N) Σ (∂L/∂θ)² via per-sample gradients on calibration data.
The old `delta.abs().pow(2)` fake Fisher is removed.

### ExFusion single-vector TIES → real cross-expert DARE-TIES-Fisher — RESOLVED

ExFusion now uses `op_ties_fisher`: DARE → TIES sign election across
experts → Fisher-weighted disjoint merge among survivors. The old
`op_ties([single_merged_vector])` that bypassed cross-expert election
is eliminated.

### AGX cross-expert ops silently RAW → real dispatch — RESOLVED

`apply_layer_merge_operator` now calls `transform_expert_set()` for
TIES/FISHER/TIES_FISHER, so candidates labeled "TIES" actually execute
TIES, not RAW task arithmetic.

### AGX CKA attention-mask dict bug — RESOLVED

Fixed `getattr(dict, "attention_mask", None)` → `dict.get("attention_mask")`.

### DARE RNG reset per-parameter → persistent generator — RESOLVED

The generator is now created once and consumed continuously across all
parameters, eliminating structured correlation between DARE masks.

### TIES semantics inconsistent → both variants explicit — RESOLVED

`TIES_MAGNITUDE` (magnitude-based election) and `TIES_MAJORITY` (pure
sign counting) are separate operators, benchmarked independently.

### Bootstrap over seeds not samples → per-sample bootstrap — RESOLVED

CIs now resample individual evaluation samples. Deterministic methods
run once; stochastic methods (DARE) run across 5 seeds with per-sample
bootstrap pooling.

### search_geometry.py hardcoded → real engine — RESOLVED

`search_geometry.py` now executes `LayerwiseGeometrySearchEngine`.

### No canonical merge API → daph_exfusion/merge/pipeline.py — RESOLVED

All merges go through `merge_experts()` with `MergeConfig` and produce
`operator_trace` for provenance.

### Release metadata v2.4/v2.5 inconsistent → v2.6 unified — RESOLVED

All artifacts now use `v2.6.0-experimental-truth`.

## Architectural limitations

### SubwordSequenceBridge is CPU-only

For production, use `CandidateVocabularyRouter` instead.

### K-FAC not implemented

Diagonal Fisher is used. K-FAC (structured curvature) is described in the
repair plan but not yet implemented.

### AGX layer-family-specific search

The infrastructure supports layer-family partitioning (attention, SSM,
FFN, norm, embeddings) but the current search uses uniform candidate
generation across all layers.

### Triton selective-scan bindings are GPU-only

Validated in tests but not exercised on this CPU-only host.
