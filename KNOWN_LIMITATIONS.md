# DAPH ExFusion — Known Limitations

Open defects and limitations as of v2.4.0-correctness.

## Blockers for paper-ready release

### 1. No lineage-matched experts trained

The `scripts/train_lineage_experts.py` script exists and is ready to run,
but no same-lineage specialists have been trained yet. The previous run
used off-the-shelf distilgpt2 fine-tunes (postbot/distilgpt2-emailgen,
FredZhang7/distilgpt2-stable-diffusion, misterkilgore/distilgpt2-psy-ita)
which failed qualification catastrophically (NaN NLL, -35% and -191%
relative improvement).

**To resolve**: run `python scripts/train_lineage_experts.py --base-model
distilgpt2 --domains math planning coding --train-data data/train/`
after creating diverse training data.

### 2. Dataset near-duplicates

The existing data in `data/` uses templated samples (e.g., "A highly
detailed digital painting of a fantasy landscape, 8k render, masterpiece
quality sample N") that trigger near-duplicate detection across splits.
Exact overlap is 0, but MinHash Jaccard similarity exceeds the 0.8
threshold.

**To resolve**: regenerate all data splits with diverse, non-templated text.

### 3. No train split exists

The dataset audit found 0 records in the `train` split. The existing data
only has qualification, calibration, validation, and test splits. A train
split is required for lineage expert training.

### 4. 5-seed final experiment not executed

The statistical validation infrastructure (`daph_exfusion/validation/`) is
ready, but the actual 5-seed experiment cannot run until lineage experts
are trained and qualified.

### 5. Surrogate has no search history

`TreeSurrogatePredictor` is implemented and tested, but has no real
search history to fit on. It will only become usable for acquisition after
enough AGX search candidates have been evaluated.

## Architectural limitations

### SubwordSequenceBridge is CPU-only

`SubwordSequenceBridge` uses Python ThreadPoolExecutor + tokenizer
round-trips. It is classified as:
- `backend = "cpu_compatibility"`
- `compile_safe = false`
- `cuda_graph_safe = false`
- `production_default = false`

For production, use `CandidateVocabularyRouter` instead.

### AGX search not yet run end-to-end

All AGX infrastructure (operators, groupwise search, Pareto, halving,
surrogate, acquisition, geometry policy) is implemented and unit-tested,
but no full end-to-end search has been executed. This requires qualified
experts first.
