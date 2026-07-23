# DAPH NeSy-MoE & ExFusion Execution Results & Benchmarks

This document records the official live execution results for unit tests, performance benchmarks, and multi-model ExFusion pipeline runs.

---

## 🧪 1. Unit Test Verification (`test_nesy_v1_0.py`)

All 11 test suites executed live and passed without errors:

```
Running DAPH NeSy-MoE v1.1 Extended test suite...
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

All 11 NeSy-MoE v1.1 Extended tests passed (executed live).
```

---

## ⚡ 2. Performance & Memory Benchmarks (`benchmark_nesy.py`)

Layer throughput and VRAM allocation profiled across batch sizes $B \in \{1, 4, 8\}$ and sequence lengths $L \in \{128, 512\}$ (Metal / MPS GPU Acceleration):

```
Batch    | SeqLen   | Latency (ms)   | Throughput (tok/s)   | Peak VRAM (MB)
----------------------------------------------------------------------
1        | 128      | 18.91          | 6770.3               | 352.37      
1        | 512      | 60.82          | 8418.4               | 354.85      
4        | 512      | 81.13          | 25242.9              | 363.99      
8        | 512      | 137.27         | 29838.6              | 376.27      
```

---

## 🔀 3. Hugging Face 3-Model ExFusion Pipeline Results (`run_model_merge.py`)

### Merged Models
* **Base Model**: `distilbert/distilgpt2`
* **Expert 1**: `postbot/distilgpt2-emailgen`
* **Expert 2**: `FredZhang7/distilgpt2-stable-diffusion`
* **Expert 3**: `misterkilgore/distilgpt2-psy-ita`

### Execution Output & Generation Comparisons

```
================================================================================
1. DOWNLOADING & LOADING HUGGING FACE MODELS FROM HUB
================================================================================

[+] Loading Base Model: 'distilbert/distilgpt2'...
[+] Loading Expert 1: 'postbot/distilgpt2-emailgen'...
[+] Loading Expert 2: 'FredZhang7/distilgpt2-stable-diffusion'...
[+] Loading Expert 3: 'misterkilgore/distilgpt2-psy-ita'...

[✓] All 3 Hugging Face Expert Models and Base Model Loaded Successfully.

================================================================================
2. EXECUTING DAPH EXFUSION MODEL MERGING PIPELINE
   (DARE Preprocessing -> TIES v2 Sign Election -> Fisher Diagonal Weighting)
================================================================================

[+] Merging expert family deltas into target model container...
[✓] ExFusion Merging Complete. Applied merged deltas across 77 parameter tensors.

================================================================================
3. GENERATION COMPARISON ACROSS EXPERTS VS. UNIFIED MERGED MODEL
================================================================================

----------------------------------------------------------------------
Category: Email / Business Writing
Prompt: 'Dear Team,
I am writing to share an update regarding'
----------------------------------------------------------------------

 -> Base Model (distilgpt2):
    Dear Team,
    I am writing to share an update regarding a recent tweet by the President of the American Psychological Association, Dr. Steve Sailer, that suggests that we should not allow this phenomenon to continue.

 -> Expert 1 (Email Gen):
    Dear Team,
    I am writing to share an update regarding the update.
    Below are the changes in the revised version of the <COMPANY> agreement.
    I am also attaching a summary of the

 -> Expert 2 (Stable Diffusion Art):
    Dear Team,
    I am writing to share an update regarding in a library, hyperrealistic, 8k, uhd, 8k, award-winning, cinematic lighting, uhd, uhd,

 -> Expert 3 (Psychology & Dialogue):
    Dear Team,
    I am writing to share an update regarding c 1G ther, Bom militarylymban easy/omet T cause le thaw/an interface beenUPol cult Am1 R aff

 -> UNIFIED MERGED MODEL (DAPH ExFusion Merged Calibrated λ=0.35):
    Dear Team,
    I am writing to share an update regarding the status of the Sony vs. Sony E3-Concord.
    Please be in contact wnk.io and e-mail me

----------------------------------------------------------------------
Category: Art / Stable Diffusion Prompt
Prompt: 'A highly detailed digital painting of a futuristic city with'
----------------------------------------------------------------------

 -> Base Model (distilgpt2):
    A highly detailed digital painting of a futuristic city with no walls.

 -> Expert 1 (Email Gen):
    A highly detailed digital painting of a futuristic city with an eye toward its future. It will be available online by Tuesday, August <NUMBER>.

 -> Expert 2 (Stable Diffusion Art):
    A highly detailed digital painting of a futuristic city with a view of the city from a distance, by Studio Ghibli, Makoto Shinkai, by Artgerm, by beeple,

 -> Expert 3 (Psychology & Dialogue):
    A highly detailed digital painting of a futuristic city withmud Bomatterlyleyans why wland d limitsve Tank Finlin

 -> UNIFIED MERGED MODEL (DAPH ExFusion Merged Calibrated λ=0.35):
    A highly detailed digital painting of a futuristic city with its own unique lights. Four lights are in view, and all of the lights are on the way to a new home.

----------------------------------------------------------------------
Category: Psychology & Dialogue
Prompt: 'In psychological terms, emotional resilience is defined as'
----------------------------------------------------------------------

 -> Base Model (distilgpt2):
    In psychological terms, emotional resilience is defined as a personality trait, which is considered a mental trait, which is considered a psychological trait...

 -> Expert 1 (Email Gen):
    In psychological terms, emotional resilience is defined as the ability to feel the world around you emotionally.

 -> Expert 2 (Stable Diffusion Art):
    In psychological terms, emotional resilience is defined as a very beautiful 3d anime, featured on pixiv, anime aesthetic, official art, 8k, uhd

 -> Expert 3 (Psychology & Dialogue):
    In psychological terms, emotional resilience is defined as history soc w yetatter/ addslectentialig animan history

 -> UNIFIED MERGED MODEL (DAPH ExFusion Merged Calibrated λ=0.35):
    In psychological terms, emotional resilience is defined as being around an ever expanding and complex, and we are in an age of digital age. The term is not an emotional one. However, emotional resilience

================================================================================
SUCCESS: 3 Hugging Face Models Downloaded & Merged via DAPH ExFusion Pipeline!
================================================================================
```
