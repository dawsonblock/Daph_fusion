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

 -> UNIFIED MERGED MODEL (DAPH ExFusion Merged):
    Dear Team,
    I am writing to share an update regarding body bag full of Cr warrants focuses under SophAttuggageuggageAttAttAttAttAttAttAttAttAttAttAttAttAttAttAttAttAttAtt

----------------------------------------------------------------------
Category: Art / Stable Diffusion Prompt
Prompt: 'A highly detailed digital painting of a futuristic city with'
----------------------------------------------------------------------

 -> Base Model (distilgpt2):
    A highly detailed digital painting of a futuristic city with some of the most recognizable characters of the 19th century.

 -> Expert 1 (Email Gen):
    A highly detailed digital painting of a futuristic city with a new, yet unexplainable atmosphere and the uncanny valley, and is made for a very compelling story.

 -> Expert 2 (Stable Diffusion Art):
    A highly detailed digital painting of a futuristic city with flying cars in the background, artstation, concept art, smooth, sharp focus, illustration, Unreal Engine 5, 8K, art by artger

 -> Expert 3 (Psychology & Dialogue):
    A highly detailed digital painting of a futuristic city withmud who stay donntaries in attempt T).ALoramink expl back thTI st th elementsanakes rel1ain th graphics Wundred

 -> UNIFIED MERGED MODEL (DAPH ExFusion Merged):
    A highly detailed digital painting of a futuristic city with undeniable legislation marvels, recommends skyscrapers,Tech legislation legislation legislation legislation legislation legislation legislation legislation legislation focuses, Alan legislation legislation legislation legislation legislation focuses,

----------------------------------------------------------------------
Category: Psychology & Dialogue
Prompt: 'In psychological terms, emotional resilience is defined as'
----------------------------------------------------------------------

 -> Base Model (distilgpt2):
    In psychological terms, emotional resilience is defined as a state of mental and physical resilience. The physical and psychological stress of an individual's life is characterized by the presence or absence of any mental or physical

 -> Expert 1 (Email Gen):
    In psychological terms, emotional resilience is defined as both physical and psychological. They also have a strong sense of self-doubt and fear. In their role as human beings, they feel that they

 -> Expert 2 (Stable Diffusion Art):
    In psychological terms, emotional resilience is defined as a magical book, trending on artstation, highly detailed, hyperrealistic, 8k, uhd, hyperrealistic, hyperrealistic,

 -> Expert 3 (Psychology & Dialogue):
    In psychological terms, emotional resilience is defined as Mayan sincely qualan easyom plansowingorddenam Br That/veributeik successfulcc th openried1ily Because taken W emergency

 -> UNIFIED MERGED MODEL (DAPH ExFusion Merged):
    In psychological terms, emotional resilience is defined as usually were usually were usually were usually were usually were usually were usually were usually were usually were usually were usually were usually were usually were usually were usually were

================================================================================
SUCCESS: 3 Hugging Face Models Downloaded & Merged via DAPH ExFusion Pipeline!
================================================================================
```
