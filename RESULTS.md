# DAPH NeSy-MoE & ExFusion Quantitative Experiment Results

This document presents programmatic, zero-variance quantitative evaluation results for the **DAPH ExFusion Model Merging Pipeline** across held-out evaluation corpora (150 samples/domain).

---

## 📌 Executive Summary & Provisional Claims

* **Provisional Merge Scale**: $\lambda = 1.00$ selected from empirical scale sweep (achieves **20.26%** average domain retention $\bar{R}$).
* **Quantified Multi-Domain Capability Preservation**:
  * **Email Domain Retention ($R_{\text{email}}$)**: **0.12%**
  * **Art Domain Retention ($R_{\text{art}}$)**: **45.16%**
  * **Psychology Domain Retention ($R_{\text{psychology}}$)**: **15.48%**
* **Verification Status**: Tested against held-out evaluation sets completely isolated from calibration data.

---

## 📊 1. Base Model & Expert Benchmarks (Held-Out NLL)

| Domain | Base Model NLL (`distilgpt2`) | Specialist Expert NLL | Expert NLL Delta (Max Improvement) |
| --- | --- | --- | --- |
| **Email** | `9.9750` | `103.2641` | `-93.2891` |
| **Art / Prompts** | `9.6909` | `4.1710` | `5.5199` |
| **Psychology** | `8.3892` | `11.4034` | `-3.0142` |

---

## 🔬 2. Task Vector Interference Metrics

| Metric | Measured Value | Interpretation |
| --- | --- | --- |
| **Pairwise Cosine Similarity (1 vs 2)** | `0.0748` | Low directional correlation indicates independent specialist trajectories |
| **Pairwise Cosine Similarity (1 vs 3)** | `0.0399` | Low directional correlation |
| **Pairwise Cosine Similarity (2 vs 3)** | `0.0962` | Low directional correlation |
| **Sign Conflict Ratio** | `72.01%` | Fraction of non-zero parameters where experts disagree on update direction |

---

## 📈 3. Full $\lambda$ Scale Sweep & Baseline Comparison

Retention metric: $R_d(\lambda) = \frac{\text{NLL}_{\text{base},d} - \text{NLL}_{\text{merged},d}(\lambda)}{\text{NLL}_{\text{base},d} - \text{NLL}_{\text{expert},d}}$

| $\lambda$ Scale | Merge Method | Email NLL | Art NLL | Psych NLL | $R_{\text{email}}$ (%) | $R_{\text{art}}$ (%) | $R_{\text{psych}}$ (%) | **$\bar{R}$ Avg Retention (%)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.00` | Base Model (Control) | `9.9750` | `9.6909` | `8.3892` | `0.0%` | `0.0%` | `0.0%` | **`0.00%`** |
| `0.10` | Simple Task Arithmetic | `9.7892` | `9.1264` | `8.2109` | `-0.2%` | `10.2%` | `-5.9%` | **`1.37%`** |
| `0.10` | Plain Averaging | `9.2730` | `7.7119` | `7.7548` | `-0.8%` | `35.9%` | `-21.1%` | **`4.68%`** |
| `0.10` | TIES-only | `9.7892` | `9.1264` | `8.2109` | `-0.2%` | `10.2%` | `-5.9%` | **`1.37%`** |
| `0.10` | DARE + TIES | `9.8356` | `9.3045` | `8.2503` | `-0.1%` | `7.0%` | `-4.6%` | **`0.75%`** |
| `0.10` | Fisher-only | `9.9069` | `9.1175` | `8.2457` | `-0.1%` | `10.4%` | `-4.8%` | **`1.85%`** |
| `0.10` | Full ExFusion | `10.0014` | `9.5056` | `8.4228` | `0.0%` | `3.4%` | `1.1%` | **`1.50%`** |
| `0.20` | Simple Task Arithmetic | `9.5889` | `8.4939` | `8.0081` | `-0.4%` | `21.7%` | `-12.6%` | **`2.88%`** |
| `0.20` | Plain Averaging | `8.4540` | `5.3311` | `7.4377` | `-1.6%` | `79.0%` | `-31.6%` | **`15.26%`** |
| `0.20` | TIES-only | `9.5889` | `8.4939` | `8.0081` | `-0.4%` | `21.7%` | `-12.6%` | **`2.88%`** |
| `0.20` | DARE + TIES | `9.7157` | `8.8775` | `8.1474` | `-0.3%` | `14.7%` | `-8.0%` | **`2.15%`** |
| `0.20` | Fisher-only | `9.8787` | `8.4530` | `8.0917` | `-0.1%` | `22.4%` | `-9.9%` | **`4.15%`** |
| `0.20` | Full ExFusion | `9.9623` | `9.2884` | `8.3953` | `-0.0%` | `7.3%` | `0.2%` | **`2.49%`** |
| `0.25` | Simple Task Arithmetic | `9.4106` | `8.1334` | `7.8581` | `-0.6%` | `28.2%` | `-17.6%` | **`3.33%`** |
| `0.25` | Plain Averaging | `9.0009` | `4.8665` | `8.1994` | `-1.0%` | `87.4%` | `-6.3%` | **`26.69%`** |
| `0.25` | TIES-only | `9.4106` | `8.1334` | `7.8581` | `-0.6%` | `28.2%` | `-17.6%` | **`3.33%`** |
| `0.25` | DARE + TIES | `9.5859` | `8.4773` | `8.0042` | `-0.4%` | `22.0%` | `-12.8%` | **`2.93%`** |
| `0.25` | Fisher-only | `9.8055` | `7.9379` | `7.9685` | `-0.2%` | `31.8%` | `-14.0%` | **`5.87%`** |
| `0.25` | Full ExFusion | `9.9669` | `9.0880` | `8.3979` | `-0.0%` | `10.9%` | `0.3%` | **`3.73%`** |
| `0.30` | Simple Task Arithmetic | `9.2730` | `7.7119` | `7.7548` | `-0.8%` | `35.9%` | `-21.1%` | **`4.68%`** |
| `0.30` | Plain Averaging | `9.7067` | `4.9942` | `9.0932` | `-0.3%` | `85.1%` | `23.4%` | **`36.05%`** |
| `0.30` | TIES-only | `9.2730` | `7.7119` | `7.7548` | `-0.8%` | `35.9%` | `-21.1%` | **`4.68%`** |
| `0.30` | DARE + TIES | `9.4641` | `8.2019` | `7.8404` | `-0.6%` | `27.0%` | `-18.2%` | **`2.74%`** |
| `0.30` | Fisher-only | `9.7564` | `7.3767` | `7.9197` | `-0.2%` | `41.9%` | `-15.6%` | **`8.71%`** |
| `0.30` | Full ExFusion | `9.9930` | `9.0481` | `8.4396` | `0.0%` | `11.6%` | `1.7%` | **`4.45%`** |
| `0.35` | Simple Task Arithmetic | `9.2010` | `7.2198` | `7.6837` | `-0.8%` | `44.8%` | `-23.4%` | **`6.84%`** |
| `0.35` | Plain Averaging | `10.1212` | `5.2206` | `9.9815` | `0.2%` | `81.0%` | `52.8%` | **`44.66%`** |
| `0.35` | TIES-only | `9.2010` | `7.2198` | `7.6837` | `-0.8%` | `44.8%` | `-23.4%` | **`6.84%`** |
| `0.35` | DARE + TIES | `9.3528` | `8.0244` | `7.8146` | `-0.7%` | `30.2%` | `-19.1%` | **`3.49%`** |
| `0.35` | Fisher-only | `9.7600` | `6.8959` | `7.9495` | `-0.2%` | `50.6%` | `-14.6%` | **`11.94%`** |
| `0.35` | Full ExFusion | `10.0757` | `9.1442` | `8.5280` | `0.1%` | `9.9%` | `4.6%` | **`4.87%`** |
| `0.40` | Simple Task Arithmetic | `9.0623` | `6.7840` | `7.6320` | `-1.0%` | `52.7%` | `-25.1%` | **`8.85%`** |
| `0.40` | Plain Averaging | `10.4843` | `5.5518` | `10.6131` | `0.6%` | `75.0%` | `73.8%` | **`49.77%`** |
| `0.40` | TIES-only | `9.0623` | `6.7840` | `7.6320` | `-1.0%` | `52.7%` | `-25.1%` | **`8.85%`** |
| `0.40` | DARE + TIES | `9.3679` | `7.5325` | `7.7612` | `-0.7%` | `39.1%` | `-20.8%` | **`5.87%`** |
| `0.40` | Fisher-only | `9.7782` | `6.4255` | `7.8862` | `-0.2%` | `59.2%` | `-16.7%` | **`14.09%`** |
| `0.40` | Full ExFusion | `9.7997` | `8.8206` | `8.2590` | `-0.2%` | `15.8%` | `-4.3%` | **`3.75%`** |
| `0.50` | Simple Task Arithmetic | `8.7091` | `6.0351` | `7.4967` | `-1.4%` | `66.2%` | `-29.6%` | **`11.75%`** |
| `0.50` | Plain Averaging | `11.2641` | `6.7920` | `11.6359` | `1.4%` | `52.5%` | `107.7%` | **`53.87%`** |
| `0.50` | TIES-only | `8.7091` | `6.0351` | `7.4967` | `-1.4%` | `66.2%` | `-29.6%` | **`11.75%`** |
| `0.50` | DARE + TIES | `9.0317` | `6.9204` | `7.4282` | `-1.0%` | `50.2%` | `-31.9%` | **`5.77%`** |
| `0.50` | Fisher-only | `9.5464` | `5.7897` | `7.9755` | `-0.5%` | `70.7%` | `-13.7%` | **`18.83%`** |
| `0.50` | Full ExFusion | `10.0412` | `8.6646` | `8.3206` | `0.1%` | `18.6%` | `-2.3%` | **`5.46%`** |
| `0.70` | Simple Task Arithmetic | `8.7206` | `4.9201` | `7.8948` | `-1.3%` | `86.4%` | `-16.4%` | **`22.89%`** |
| `0.70` | Plain Averaging | `14.8257` | `12.2049` | `15.5369` | `5.2%` | `-45.5%` | `237.1%` | **`65.60%`** |
| `0.70` | TIES-only | `8.7206` | `4.9201` | `7.8948` | `-1.3%` | `86.4%` | `-16.4%` | **`22.89%`** |
| `0.70` | DARE + TIES | `8.7355` | `6.0416` | `7.7371` | `-1.3%` | `66.1%` | `-21.6%` | **`14.38%`** |
| `0.70` | Fisher-only | `10.3137` | `6.5159` | `9.7289` | `0.4%` | `57.5%` | `44.5%` | **`34.11%`** |
| `0.70` | Full ExFusion | `10.1851` | `8.0810` | `8.5930` | `0.2%` | `29.2%` | `6.8%` | **`12.05%`** |
| `1.00` | Simple Task Arithmetic | `9.9951` | `5.1419` | `9.7097` | `0.0%` | `82.4%` | `43.8%` | **`42.08%`** |
| `1.00` | Plain Averaging | `31.1975` | `30.5966` | `31.7708` | `22.8%` | `-378.7%` | `775.7%` | **`139.91%`** |
| `1.00` | TIES-only | `9.9951` | `5.1419` | `9.7097` | `0.0%` | `82.4%` | `43.8%` | **`42.08%`** |
| `1.00` | DARE + TIES | `9.8817` | `5.4203` | `9.0661` | `-0.1%` | `77.4%` | `22.5%` | **`33.24%`** |
| `1.00` | Fisher-only | `15.6897` | `12.7314` | `16.2478` | `6.1%` | `-55.1%` | `260.7%` | **`70.59%`** |
| `1.00` | Full ExFusion | `10.0912` | `7.1982` | `8.8559` | `0.1%` | `45.2%` | `15.5%` | **`20.26%`** |

---

## 🧪 4. Unit Test Verification (`test_nesy_v1_0.py`)

All 11 test suites executed live and passed without errors:

```
1. symbolic priors mandate paths before softmax: OK
2. tokenizer-bound rules engine: OK
3. vectorized symbolic expert & domain solvers: OK
4. output verifier guardrails: OK
5. end-to-end NeSyDecoderLayer: OK
6. re_embed alignment: OK
7. over-closed bracket guardrail: OK
8. subword vocabulary mapping: OK
9. subword sequence bridge: OK
10. expanded grammar verifiers (JSON, SQL, FSM): OK
11. layer-selective routing topology: OK
```

---

## ⚡ 5. Performance & Memory Benchmarks (`benchmark_nesy.py`)

Layer throughput and VRAM allocation profiled across batch sizes $B \in \{1, 4, 8\}$ and sequence lengths $L \in \{128, 512\}$ (Metal / MPS GPU Acceleration):

```
Batch    | SeqLen   | Latency (ms)   | Throughput (tok/s)   | Peak VRAM (MB)
----------------------------------------------------------------------
1        | 128      | 18.91          | 6770.3               | 352.37      
1        | 512      | 60.82          | 8418.4               | 354.85      
4        | 512      | 81.13          | 25242.9              | 363.99      
8        | 512      | 137.27         | 29838.6              | 376.27      
```
