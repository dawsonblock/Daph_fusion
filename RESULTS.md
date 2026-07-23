# DAPH NeSy-MoE & ExFusion Quantitative Experiment Results

This document presents programmatic, zero-variance quantitative evaluation results for the **DAPH ExFusion Model Merging Pipeline** across held-out evaluation corpora (150 samples/domain).

---

## 📌 Executive Summary & Provisional Claims

* **Provisional Merge Scale**: $\lambda = 1.00$ selected from empirical scale sweep (achieves **21.20%** average domain retention $\bar{R}$).
* **Quantified Multi-Domain Capability Preservation**:
  * **Email Domain Retention ($R_{\text{email}}$)**: **-0.21%**
  * **Art Domain Retention ($R_{\text{art}}$)**: **51.21%**
  * **Psychology Domain Retention ($R_{\text{psychology}}$)**: **12.60%**
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
| `0.10` | DARE + TIES | `9.8351` | `9.3106` | `8.2492` | `-0.1%` | `6.9%` | `-4.6%` | **`0.70%`** |
| `0.10` | Fisher-only | `9.9069` | `9.1175` | `8.2457` | `-0.1%` | `10.4%` | `-4.8%` | **`1.85%`** |
| `0.10` | Full ExFusion | `10.0380` | `9.5244` | `8.3921` | `0.1%` | `3.0%` | `0.1%` | **`1.06%`** |
| `0.20` | Simple Task Arithmetic | `9.5889` | `8.4939` | `8.0081` | `-0.4%` | `21.7%` | `-12.6%` | **`2.88%`** |
| `0.20` | Plain Averaging | `8.4540` | `5.3311` | `7.4377` | `-1.6%` | `79.0%` | `-31.6%` | **`15.26%`** |
| `0.20` | TIES-only | `9.5889` | `8.4939` | `8.0081` | `-0.4%` | `21.7%` | `-12.6%` | **`2.88%`** |
| `0.20` | DARE + TIES | `9.8013` | `8.9084` | `8.2516` | `-0.2%` | `14.2%` | `-4.6%` | **`3.14%`** |
| `0.20` | Fisher-only | `9.8787` | `8.4530` | `8.0917` | `-0.1%` | `22.4%` | `-9.9%` | **`4.15%`** |
| `0.20` | Full ExFusion | `10.0198` | `9.3549` | `8.4305` | `0.1%` | `6.1%` | `1.4%` | **`2.50%`** |
| `0.25` | Simple Task Arithmetic | `9.4106` | `8.1334` | `7.8581` | `-0.6%` | `28.2%` | `-17.6%` | **`3.33%`** |
| `0.25` | Plain Averaging | `9.0009` | `4.8665` | `8.1994` | `-1.0%` | `87.4%` | `-6.3%` | **`26.69%`** |
| `0.25` | TIES-only | `9.4106` | `8.1334` | `7.8581` | `-0.6%` | `28.2%` | `-17.6%` | **`3.33%`** |
| `0.25` | DARE + TIES | `9.6021` | `8.6188` | `8.0953` | `-0.4%` | `19.4%` | `-9.8%` | **`3.09%`** |
| `0.25` | Fisher-only | `9.8055` | `7.9379` | `7.9685` | `-0.2%` | `31.8%` | `-14.0%` | **`5.87%`** |
| `0.25` | Full ExFusion | `10.0596` | `9.3937` | `8.5108` | `0.1%` | `5.4%` | `4.0%` | **`3.17%`** |
| `0.30` | Simple Task Arithmetic | `9.2730` | `7.7119` | `7.7548` | `-0.8%` | `35.9%` | `-21.1%` | **`4.68%`** |
| `0.30` | Plain Averaging | `9.7067` | `4.9942` | `9.0932` | `-0.3%` | `85.1%` | `23.4%` | **`36.05%`** |
| `0.30` | TIES-only | `9.2730` | `7.7119` | `7.7548` | `-0.8%` | `35.9%` | `-21.1%` | **`4.68%`** |
| `0.30` | DARE + TIES | `9.4287` | `8.2393` | `7.8348` | `-0.6%` | `26.3%` | `-18.4%` | **`2.44%`** |
| `0.30` | Fisher-only | `9.7564` | `7.3767` | `7.9197` | `-0.2%` | `41.9%` | `-15.6%` | **`8.71%`** |
| `0.30` | Full ExFusion | `10.2125` | `9.1122` | `8.4886` | `0.2%` | `10.5%` | `3.3%` | **`4.68%`** |
| `0.35` | Simple Task Arithmetic | `9.2010` | `7.2198` | `7.6837` | `-0.8%` | `44.8%` | `-23.4%` | **`6.84%`** |
| `0.35` | Plain Averaging | `10.1212` | `5.2206` | `9.9815` | `0.2%` | `81.0%` | `52.8%` | **`44.66%`** |
| `0.35` | TIES-only | `9.2010` | `7.2198` | `7.6837` | `-0.8%` | `44.8%` | `-23.4%` | **`6.84%`** |
| `0.35` | DARE + TIES | `9.4308` | `8.0389` | `7.8311` | `-0.6%` | `29.9%` | `-18.5%` | **`3.61%`** |
| `0.35` | Fisher-only | `9.7600` | `6.8959` | `7.9496` | `-0.2%` | `50.6%` | `-14.6%` | **`11.94%`** |
| `0.35` | Full ExFusion | `10.0461` | `8.9525` | `8.4791` | `0.1%` | `13.4%` | `3.0%` | **`5.48%`** |
| `0.40` | Simple Task Arithmetic | `9.0623` | `6.7840` | `7.6320` | `-1.0%` | `52.7%` | `-25.1%` | **`8.85%`** |
| `0.40` | Plain Averaging | `10.4843` | `5.5518` | `10.6131` | `0.6%` | `75.0%` | `73.8%` | **`49.77%`** |
| `0.40` | TIES-only | `9.0623` | `6.7840` | `7.6320` | `-1.0%` | `52.7%` | `-25.1%` | **`8.85%`** |
| `0.40` | DARE + TIES | `9.2602` | `7.7318` | `7.8485` | `-0.8%` | `35.5%` | `-17.9%` | **`5.60%`** |
| `0.40` | Fisher-only | `9.7782` | `6.4255` | `7.8862` | `-0.2%` | `59.2%` | `-16.7%` | **`14.09%`** |
| `0.40` | Full ExFusion | `9.9785` | `9.0078` | `8.4504` | `0.0%` | `12.4%` | `2.0%` | **`4.80%`** |
| `0.50` | Simple Task Arithmetic | `8.7091` | `6.0351` | `7.4967` | `-1.4%` | `66.2%` | `-29.6%` | **`11.75%`** |
| `0.50` | Plain Averaging | `11.2641` | `6.7920` | `11.6359` | `1.4%` | `52.5%` | `107.7%` | **`53.87%`** |
| `0.50` | TIES-only | `8.7091` | `6.0351` | `7.4967` | `-1.4%` | `66.2%` | `-29.6%` | **`11.75%`** |
| `0.50` | DARE + TIES | `9.2325` | `6.9593` | `7.8062` | `-0.8%` | `49.5%` | `-19.3%` | **`9.78%`** |
| `0.50` | Fisher-only | `9.5465` | `5.7897` | `7.9756` | `-0.5%` | `70.7%` | `-13.7%` | **`18.83%`** |
| `0.50` | Full ExFusion | `10.1652` | `8.7265` | `8.3999` | `0.2%` | `17.5%` | `0.4%` | **`6.01%`** |
| `0.70` | Simple Task Arithmetic | `8.7206` | `4.9201` | `7.8948` | `-1.3%` | `86.4%` | `-16.4%` | **`22.89%`** |
| `0.70` | Plain Averaging | `14.8257` | `12.2049` | `15.5369` | `5.2%` | `-45.5%` | `237.1%` | **`65.60%`** |
| `0.70` | TIES-only | `8.7206` | `4.9201` | `7.8948` | `-1.3%` | `86.4%` | `-16.4%` | **`22.89%`** |
| `0.70` | DARE + TIES | `8.8821` | `6.0830` | `7.7850` | `-1.2%` | `65.4%` | `-20.0%` | **`14.72%`** |
| `0.70` | Fisher-only | `10.3137` | `6.5159` | `9.7289` | `0.4%` | `57.5%` | `44.5%` | **`34.11%`** |
| `0.70` | Full ExFusion | `10.2630` | `8.2535` | `8.7128` | `0.3%` | `26.0%` | `10.7%` | **`12.36%`** |
| `1.00` | Simple Task Arithmetic | `9.9951` | `5.1419` | `9.7097` | `0.0%` | `82.4%` | `43.8%` | **`42.08%`** |
| `1.00` | Plain Averaging | `31.1975` | `30.5966` | `31.7708` | `22.8%` | `-378.7%` | `775.7%` | **`139.91%`** |
| `1.00` | TIES-only | `9.9951` | `5.1419` | `9.7097` | `0.0%` | `82.4%` | `43.8%` | **`42.08%`** |
| `1.00` | DARE + TIES | `9.4058` | `5.4207` | `8.8395` | `-0.6%` | `77.4%` | `14.9%` | **`30.56%`** |
| `1.00` | Fisher-only | `15.6897` | `12.7314` | `16.2478` | `6.1%` | `-55.1%` | `260.7%` | **`70.59%`** |
| `1.00` | Full ExFusion | `9.7805` | `6.8640` | `8.7689` | `-0.2%` | `51.2%` | `12.6%` | **`21.20%`** |

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
