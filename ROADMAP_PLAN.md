# DAPH / ExFusion Research Roadmap

v2.3.2 Experimental Validity → v2.4 Adaptive Geometry ExFusion

1. Program Objective

The next DAPH/ExFusion program answers two questions in sequence:

1. Can the experimental harness produce scientifically valid measurements of expert retention, interference, and general capability degradation?
2. Given trustworthy measurements, can ExFusion automatically discover a better merge geometry than fixed global task arithmetic, TIES, DARE, Fisher, or manually selected combinations?

The target system is:

\boxed{
\theta^*
=

\theta_0 +
\mathcal{G}^*
(
\Delta_1,\Delta_2,\ldots,\Delta_N
\mid
\mathcal{D}_{cal},
\mathcal{D}_{val}
)
}

where:

- \theta_0 = base model parameters
- \Delta_i=\theta_i-\theta_0 = expert task vectors
- \mathcal{G}^* = automatically discovered merge transformation
- \mathcal{D}_{cal} = calibration data
- \mathcal{D}_{val} = validation data

The system searches for the geometry without using the final test set.

The development sequence is:

\boxed{
\text{Validity}
\rightarrow
\text{Geometry Measurement}
\rightarrow
\text{Strong Baselines}
\rightarrow
\text{Adaptive Search}
\rightarrow
\text{Pareto Optimization}
\rightarrow
\text{Surrogate Search}
\rightarrow
\text{Learned Geometry Policy}
}

---

## PART I — v2.3.2 EXPERIMENTAL VALIDITY

### Phase 0 — Freeze the Existing Baseline

- Baseline reference commit: `BASELINE_COMMIT = a4222cb` (frozen under `artifacts/baselines/v2_3_1/`).
- Mandatory provenance logs: commit, dirty state, Python version, PyTorch version, device, model/tokenizer revisions, dataset hashes, seed, config hash.

### Phase 1 — Expert Qualification System

- Relative improvement threshold: $I_i = \frac{L_{base,i}-L_{expert,i}}{L_{base,i}} \ge 0.05$.
- Module: `experiments/qualification.py` and CLI `validate_experts.py`.
- Fail-closed preflight gate: search halted if any expert fails qualification.

### Phase 2 — Diagnose Pathological Experts

- Diagnostic suite under `diagnostics/`:
  - `checkpoint_compatibility.py`
  - `tokenizer_compatibility.py`
  - `loss_audit.py` (shift CE, padding mask -100 verification, token loss distribution percentiles p90/p95/p99)
  - `dataset_audit.py` (hash disjointness)

### Phase 3 — Source Proper Experts

- Lineage control: experts fine-tuned from exact same base checkpoint $\theta_0$.
- Topological validation: parameter key hashes, shape hashes, architecture hashes.

### Phase 4 — Build Proper Dataset Separation

- 4-layer isolated splits under `data/`: `qualification/`, `calibration/`, `validation/`, `test/`.
- Strict role isolation: Calibration for Fisher/curvature; Validation for search/hyperparameters; Test for single-pass final evaluation.

### Phase 5 — Correct Research Metrics

- Retention metric guarded by expert advantage ($L_{expert,i} < L_{base,i}$).
- 4 metrics per domain: Absolute gain $A_i$, Relative base gain $B_i$, Retention $R_i$, Regression $G_i$.

---

## PART II — GEOMETRY CHARACTERIZATION

### Phase 6 — Build the Geometry Analyzer

- `daph_exfusion/geometry/descriptors.py` & `hierarchy.py`.
- Global descriptors: norms $\|\Delta_i\|_2$, RMS, pairwise cosine similarity $\cos(\Delta_i,\Delta_j)$, sign conflict $C_{sign}(i,j)$.

### Phase 7 — Layerwise Geometry

- Layerwise descriptors $g_l = [n_i, r_i, c_{ij}, s_{ij}, m_{ij}, f_i, \kappa_i]$.

### Phase 8 — Blockwise Geometry

- Block resolution $g_{l,b}$ across attention q/k/v/o, mlp up/gate/down, ssm A/D/dt/B/C, norms, embeddings, lm_head.

### Phase 9 — Representation-Space Geometry

- Activation states $H_0^l, H_i^l$: CKA, activation norm drift, representation rank, output KL divergence (`daph_exfusion/geometry/representations.py`).

---

## PART III — STRONG BASELINES

### Phase 10 — Baseline Family

- Suite in `experiments/baselines.py`: Parameter Average, Raw Task Arithmetic, Unit-Norm Task Arithmetic, RMS-Normalized Arithmetic, Layer-Normalized Arithmetic, TIES, DARE, DARE-TIES, Fisher Merge, DARE-TIES-Fisher, ExFusion v2.3.

---

## PART IV — ADAPTIVE GEOMETRY EXFUSION (AGX)

### Phase 11 — AGX v1: Layerwise Coefficient Search

- Layerwise formulation $\theta_m^l = \theta_0^l + \sum_i \lambda_{i,l}\Delta_i^l$.

### Phase 12 — Automatic Merge-Operator Selection

- Operators per layer: RAW, NORMALIZED, TIES, DARE, DARE_TIES, FISHER, PROJECT.

### Phase 13 — Analytical Search-Space Pruning

- Descriptor-driven candidate pruning (high conflict $\rightarrow$ TIES/Projection; high curvature $\rightarrow$ scale limit).

### Phase 14 — Multi-Objective Search Function

- Objective $J_{balanced} = \operatorname{mean}(R_i) - \alpha \operatorname{Std}(R_i) - \beta D_g - \gamma P$.

### Phase 15 — Search Engine

- `daph_exfusion/search/`: `candidate.py`, `pareto.py`, `optimization.py`.

### Phase 16 — Coarse-to-Fine Search

- Search stages: Global $\rightarrow$ Layerwise $\rightarrow$ Blockwise.

### Phase 17 — Successive Halving / Early Stopping

- Sample budget stages: 32 $\rightarrow$ 128 $\rightarrow$ Full validation.

### Phase 18 — Pareto Frontier

- Multi-objective non-dominated front tracking in `daph_exfusion/search/pareto.py`.

---

## PART V — EFFICIENT GEOMETRY SEARCH

### Phase 19 — Surrogate Model

- Predictor $x_c \rightarrow \hat y_c$ in `daph_exfusion/search/surrogate.py`.

### Phase 20 — Active Candidate Selection

- Acquisition $A(c) = \hat J(c) + \eta U(c)$.

---

## PART VI — CURVATURE & REPRESENTATION

### Phase 21 — Fisher Validation

- Empirical Fisher diagonals in `daph_exfusion/geometry/curvature.py`.

### Phase 22 — Curvature-Constrained Search

- Curvature penalty $P_F = (\theta-\theta_0)^T F (\theta-\theta_0)$.

### Phase 23 — Representation Drift Constraint

- CKA drift penalty $D_{repr,l} = 1 - \operatorname{CKA}(H_0^l, H_m^l)$.

---

## PART VII — AUTOMATED POLICY & STATISTICS

### Phase 24 — Build a Search History Dataset

- `artifacts/geometry_history/`.

### Phase 25 — Learned Geometry Policy

- Policy network $\pi(g_{l,b})$ in `daph_exfusion/policies/adaptive_policy.py`.

### Phase 26–28 — Statistical Validation

- Multi-seed stochastic runs (5–10 seeds).
- 10,000-sample bootstrap confidence intervals (`experiments/bootstrap.py`).
- Paired bootstrap difference testing.

---

## RELEASE GATES & STATUS

- **v2.3.2 (Experimental Validity)**: Phases 0–10 — Baseline freeze, Expert qualification, Diagnostics, Lineage experts, 4-layer dataset splits, Metrics overhaul, Geometry analyzer, Strong baselines. **[ACTIVE / COMPLETE]**
- **v2.4.0 (Adaptive Geometry ExFusion v1)**: Phases 11–18 — Layerwise $\lambda$, Operator selection, Analytical pruning, Multi-objective Pareto search. **[READY FOR RUN]**
- **v2.5.0 (Efficient AGX)**: Phases 19–23 — Surrogate model, Active acquisition, Curvature/Representation constraints. **[STRUCTURED]**
- **v2.6.0 (Learned Geometry Policy)**: Phases 24–28 — History dataset, Policy network, Bootstrap statistical validation. **[STRUCTURED]**
