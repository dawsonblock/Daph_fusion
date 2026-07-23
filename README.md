# DAPH NeSy-MoE v1.1 Extended

> **Differentiable Adaptive Predictive Hybrid Neurosymbolic Mixture-of-Experts**
> _Unifying System 1 Neural Intuition with System 2 Symbolic Reasoning inside PyTorch Transformer & State-Space Architectures._

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/tests-14%2F14%20passing-brightgreen.svg)](<>)
[![Architecture](https://img.shields.io/badge/architecture-5--Path%20NeSy--MoE-purple.svg)](daph_nesy_v1_0.py)

---

## 📌 Table of Contents

- [Executive Overview](#-executive-overview)
- [Key Architectural Highlights](#-key-architectural-highlights)
- [System Architecture & Routing Flow](#-system-architecture--routing-flow)
- [Core Components Deep-Dive](#-core-components-deep-dive)
  - [1. NeSyMacroRouter](#1-nesymacrorouter)
  - [2. TokenizerBoundRulesEngine](#2-tokenizerboundrulesengine)
  - [3. VectorizedSymbolicExpert \& Domain Solvers](#3-vectorizedsymbolicexpert--domain-solvers)
  - [4. NeSyOutputVerifier](#4-nesyoutputverifier)
  - [5. NeSyDecoderLayer](#5-nesydecoderlayer)
  - [6. ExFusion Model Merging (DARE → TIES → Fisher)](#6-exfusion-model-merging-dare--ties--fisher)
- [Installation \& Prerequisites](#-installation--prerequisites)
- [Quick Start \& Code Examples](#-quick-start--code-examples)
  - [Basic Setup](#basic-setup)
  - [Registering Custom Solvers](#registering-custom-solvers)
  - [Grammar Logit Masking](#grammar-logit-masking)
- [Pretrained Model Runner (`run_model.py`)](#-pretrained-model-runner-run_modelpy)
- [Verification \& Test Suite](#-verification--testing)
- [Repository File Structure](#-repository-directory-structure)
- [Mathematical Foundations](#-mathematical-foundations)
- [License \& Citation](#-license--citation)

---

## 🧠 Executive Overview

Standard Deep Learning models excel at **System 1 tasks** (pattern recognition, fluent language generation, associative recall) but frequently fail or hallucinate during **System 2 tasks** (exact arithmetic, formal logic, Boolean satisfiability, strict syntax generation). Post-hoc decoding filters or external API tools often break end-to-end backpropagation and increase latency.

**DAPH NeSy-MoE v1.1 Extended** solves this by injecting symbolic logic directly **into the routing mathematics before softmax** and bridging non-differentiable solver execution back into continuous representations via a **Straight-Through Estimator (STE)** gradient bypass.

```
                    ┌──────────────────────────────────────────────┐
                    │            Input Sequence / Tokens           │
                    └──────────────────────┬───────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │    Predictive Difficulty Scoring    │
                        │        𝓓(x) ∈ [0.0, 1.0]           │
                        └──────────────────┬──────────────────┘
                                           │
               ┌───────────────────────────┼───────────────────────────┐
               ▼                           ▼                           ▼
      ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
      │  Path 0: SSM    │         │ Path 1: Attention│        │ Path 2: Hybrid  │
      │  (Mamba-2 O(N)) │         │ (Causal Multi-H)│        │ (SSM + Attn)    │
      └────────┬────────┘         └────────┬────────┘         └────────┬────────┘
               │                           │                           │
               └───────────────────────────┼───────────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   ▼                                               ▼
      ┌─────────────────────────┐                     ┌─────────────────────────┐
      │  Path 3: Dynamic Depth  │                     │  Path 4: Symbolic Expert│
      │  (Recurrent / Cheap)    │                     │  (Exact Tensor Solvers) │
      └────────────┬────────────┘                     └────────────┬────────────┘
                   │                                               │
                   └───────────────────────┬───────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │     Fused Output Representation & Meta       │
                    └──────────────────────────────────────────────┘
```

---

## ⚡ Key Architectural Highlights

- **5-Path Adaptive Macro-Routing**: Dynamically dispatches tokens/batches across **Mamba-2 SSM** (`0`), **Causal Attention** (`1`), **Full Hybrid** (`2`), **Recurrent Depth** (`3`), and **Vectorized Symbolic Expert** (`4`).
- **Additive Symbolic Logit Priors**: Modifies router logits _before_ softmax ($z_{\text{eff}} = z_{\text{neural}} + b_{\text{symbolic}}$) with hard mandates ($+10^5$), hard forbids ($-10^5$), or soft domain biases.
- **Straight-Through Estimator (STE) Gradient Bridge**: Discretizes neural hidden states to token IDs, executes parallel domain solvers, and re-embeds the discrete solution while preserving smooth gradient backpropagation through probabilities:
  $$\text{STE} = (\mathbf{1}_{\text{solved}} - \mathbf{p}).\text{detach}() + \mathbf{p}$$
- **Tokenizer-Bound Structural Rules**: Vectorized rule engine mapping math, logic, padding, JSON, SQL, and symbolic trigger tokens across arbitrary tokenizers (`HuggingFace`, `tiktoken`, explicit ID maps).
- **Extensible Domain Solvers**: Out-of-the-box support for exact integer arithmetic, Boolean SAT solving, AST bracket canonicalization, digit squaring, and custom user-defined solvers registered at runtime.
- **Inference-Time Grammar Guardrails**: `NeSyOutputVerifier` enforces balanced brackets and structural constraints via dynamic logit masking during autoregressive decoding.
- **DARE → TIES → Fisher Model Merging**: Merges task-specific experts into base checkpoints with density pruning, sign-consensus voting, and empirical Fisher information diagonal weighting (`daph_hybrid_exfusion_v2_3.py`).

---

## 📐 System Architecture & Routing Flow

```mermaid
graph TD
    A[Input Hidden States X & Token IDs] --> B[TokenizerBoundRulesEngine]
    A --> C[PredictiveDifficultyMacroRouter]

    B -->|Logit Priors b_symbolic| D[NeSyMacroRouter]
    C -->|Neural Logits z_neural| D

    D -->|z_eff = z_neural + b_symbolic| E[Softmax / Top-P Path Selection]

    E -->|Path 0| F[Mamba-2 State Space Model]
    E -->|Path 1| G[Causal Self-Attention]
    E -->|Path 2| H[Hybrid SSM + Attention]
    E -->|Path 3| I[Recurrent Cheap Path]
    E -->|Path 4| J[Vectorized Symbolic Expert]

    J --> J1[De-Embed Logits]
    J1 --> J2[Argmax Discrete Tokens]
    J2 --> J3[Tensor Domain Solver: Math/SAT/AST]
    J3 --> J4[STE Re-Embedding Bridge]
    J4 --> J5[Context Preservation Gate α]

    F --> K[Path Weighted Fusion]
    G --> K
    H --> K
    I --> K
    J5 --> K

    K --> L[Output Layer Representation]
```

---

## 🧩 Core Components Deep-Dive

### 1. `NeSyMacroRouter`

Extends `PredictiveDifficultyMacroRouter` by introducing an additive symbolic prior channel:

```python
z_effective = z_neural + b_symbolic
```

- **Mandate Path**: $+10^5$ (Forces selection regardless of neural confidence).
- **Forbid Path**: $-10^5$ (Blocks path entirely).
- **Neutral**: $0.0$ (Lets predictive difficulty and neural logits govern).

### 2. `TokenizerBoundRulesEngine`

Maps discrete token IDs to macro-routing priors. It auto-resolves character strings against tokenizer vocabularies so rules persist across tokenizer swaps.

- **Math Operators** (`+`, `-`, `*`, `/`, `%`, `=`): Routes to high-precision Attention/Symbolic path.
- **Control/Padding Tokens** (`[PAD]`, `<s>`, `</s>`): Forces fast Cheap path.
- **Logical Symbols** (`&`, `|`, `^`, `~`): Applies soft bias to Mamba-2 SSM.
- **JSON & SQL Tokens** (`{`, `}`, `SELECT`, `FROM`): Soft bias to Transformer Attention.
- **Symbolic Trigger Tokens** (`eval`, `exec`, `solve`, `sat`, `ast`): Mandates Symbolic Expert (`SYMBOLIC_PATH = 4`).

### 3. `VectorizedSymbolicExpert` & Domain Solvers

Executes exact GPU-vectorized logic over tensor batches:

- **`digit_squaring`**: Vectorized mod-10 digit squaring over ASCII tokens.
- **`arithmetic` / `arithmetic_eval`**: Evaluates `+`, `-`, `*` over digit tokens based on preceding operators.
- **`sat` / `sat_boolean`**: Flips Boolean bit tokens (`'0'` $\leftrightarrow$ `'1'`) under `NOT` (`~`, `!`) operations.
- **`ast` / `ast_transformer`**: Canonicalizes mismatched closing delimiters (`]`, `}`) following `(`.
- **`SubwordSequenceBridge`**: Handles multi-token subword tokenizers (BPE, SentencePiece) by decoding to text, executing string-level solvers, and re-encoding back to aligned token tensors.
- **`build_subword_vocab_map(tokenizer)`**: Precomputes a GPU-native lookup table (`vocab_map`) mapping subword vocabulary token IDs to solver target IDs in $O(1)$ time.
- **`register_solver(name, fn)`**: Allows registering custom PyTorch tensor routines dynamically.

### 4. `NeSyOutputVerifier` & Structured Grammar Guardrails

Post-hoc decoding guardrails that analyze generation state and modify next-token logits:

```python
logits = verifier.verify_and_correct_logits(generated_ids, next_token_logits)
```

- **`NeSyOutputVerifier`**: Enforces open/close bracket counts across batch sequences in parallel.
- **`JSONOutputVerifier`**: Enforces `{}` brace and `[]` bracket balance while forbidding premature `EOS` tokens when structures remain open.
- **`SQLOutputVerifier`**: Enforces SQL clause sequence (`SELECT` $\rightarrow$ `FROM`) and semicolon termination.
- **`FSMGrammarVerifier`**: Evaluates active state transitions over Finite State Machine lookup tables and masks disallowed next-tokens.

### 5. `NeSyDecoderLayer`

Integrates System 1 hybrid processing with System 2 symbolic routing seamlessly without altering the core DAPH execution signature. Supports layer-selective topology (`layer_idx`, `active_symbolic_layers`) and cached symbolic output reuse (`_get_cached_symbolic_out`):

```python
output, meta = nesy_layer(
    hidden_states,
    token_ids=input_ids,
    symbolic_priors=priors  # Optional override
)
```

### 6. `NeSyModel` Multi-Layer Container

Full multi-layer stack container managing N `NeSyDecoderLayer` instances. Handles unified state management (`mamba_state`, `attn_state`, `attn_padding_state`) and cache propagation across layers during autoregressive generation:

```python
model = NeSyModel(config, num_layers=6, tokenizer=my_tokenizer, symbolic_expert=expert)
output, past_layer_states = model(hidden_states, token_ids=input_ids, use_cache=True)
```

### 7. ExFusion Model Merging (`DARE → TIES → Fisher`)

Located in [daph_hybrid_exfusion_v2_3.py](daph_hybrid_exfusion_v2_3.py):

1. **DARE Preprocessing**: Randomly drops task vector parameters at rate $p$ and scales remaining weights by $1/(1-p)$.
2. **TIES Trimming & Sign Election**: Trims bottom $k\%$ parameters by magnitude, calculates pure sign-majority consensus voting across experts, and filters conflicting signs.
3. **Fisher Information Diagonal Weighting**: Scales expert contributions by parameter-level empirical Fisher information.

---

## 📦 Installation & Prerequisites

### Prerequisites

- Python 3.10+
- PyTorch 2.1+ (2.8+ recommended)
- `transformers` (for pretrained tokenizer/model integration)

### Setup

Clone the repository and install required dependencies:

```bash
git clone https://github.com/dawsonblock/Daph_fusion.git
cd Daph_fusion
pip install torch transformers
```

---

## 🚀 Quick Start & Code Examples

### Basic Setup

```python
import torch
from daph_hybrid_exfusion_v2_3 import DAPHConfig
from daph_nesy_v1_0 import (
    NeSyDecoderLayer,
    TokenizerBoundRulesEngine,
    VectorizedSymbolicExpert,
)

# 1. Initialize 5-path configuration
config = DAPHConfig(
    hidden_size=768,
    num_attention_heads=12,
    num_paths=5,  # 0: Mamba, 1: Attn, 2: Hybrid, 3: Cheap, 4: Symbolic
    routing_granularity="token",
)

# 2. Setup Rules Engine & Symbolic Expert
vocab_size = 32000
lm_head_weight = torch.randn(vocab_size, config.hidden_size)

rules_engine = TokenizerBoundRulesEngine(num_paths=5)
expert = VectorizedSymbolicExpert(
    hidden_size=config.hidden_size,
    vocab_size=vocab_size,
    lm_head_weight=lm_head_weight,
    domain="arithmetic_eval",
)

# 3. Instantiate NeSy Decoder Layer
layer = NeSyDecoderLayer(
    config=config,
    rules_engine=rules_engine,
    symbolic_expert=expert,
)

# 4. Forward Pass
batch_size, seq_len = 2, 16
hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

output, meta = layer(hidden_states, token_ids=input_ids)
print("Layer Output Shape:", output.shape)
print("Router Selected Paths Shape:", meta["selected_paths"].shape)
```

---

### Registering Custom Solvers

```python
import torch
from daph_nesy_v1_0 import register_solver, VectorizedSymbolicExpert

# Define custom vectorized PyTorch function over token ID tensors
def custom_rot13_solver(token_ids: torch.Tensor) -> torch.Tensor:
    # Rotate lowercase ASCII characters ('a'-'z', 97-122) by 13
    is_lower = (token_ids >= 97) & (token_ids <= 122)
    rotated = ((token_ids - 97 + 13) % 26) + 97
    return torch.where(is_lower, rotated, token_ids)

# Register with domain name
register_solver("rot13", custom_rot13_solver)

# Instantiate expert using custom solver
expert = VectorizedSymbolicExpert(
    hidden_size=768,
    vocab_size=32000,
    lm_head_weight=torch.randn(32000, 768),
    domain="rot13",
)
```

---

### Grammar Logit Masking

```python
import torch
from daph_nesy_v1_0 import NeSyOutputVerifier

# Initialize verifier for '(' (40) and ')' (41)
verifier = NeSyOutputVerifier(open_token=40, close_token=41)

# Sequence state: [ "(" ] (1 open bracket)
generated_ids = torch.tensor([[40]])
next_logits = torch.randn(1, 32000)

# Mask logits to encourage closing brackets and forbid premature closing
corrected_logits = verifier.verify_and_correct_logits(generated_ids, next_logits)
```

---

## 🤖 Pretrained Model & Merging Runners

### 1. Neurosymbolic Integration Runner (`run_model.py`)

A full end-to-end integration script [run_model.py](run_model.py) is included in the repository. It:

1. Downloads `distilbert/distilgpt2` from Hugging Face Hub.
2. Runs standard neural generation on text prompts.
3. Connects the Hugging Face tokenizer to `TokenizerBoundRulesEngine`.
4. Executes the full **DAPH NeSy-MoE v1.1 Dual-System Engine**.

Run it via:

```bash
python3 run_model.py
```

### 2. Multi-Model ExFusion Merging Runner (`run_model_merge.py`)

An automated multi-expert model merging pipeline runner [run_model_merge.py](run_model_merge.py). It:

1. Downloads 3 distinct fine-tuned expert models + 1 base model from Hugging Face Hub:
   - Base: `distilbert/distilgpt2`
   - Expert 1: `postbot/distilgpt2-emailgen` (Email / Business Writing)
   - Expert 2: `FredZhang7/distilgpt2-stable-diffusion` (Art / Prompt Generation)
   - Expert 3: `misterkilgore/distilgpt2-psy-ita` (Psychology & Dialogue)
2. Executes the DAPH ExFusion Pipeline (`DARE Preprocessing` $\rightarrow$ `TIES v2 Sign Election` $\rightarrow$ `Fisher Diagonal Weighting`).
3. Applies merged parameter deltas directly to a target model container.
4. Generates text across all individual experts, the base model, and the unified merged model.

Run it via:

```bash
python3 run_model_merge.py
```

### 3. Quantitative Evaluation & Experiment Suite (`run_experiments.py`)

An automated quantitative evaluation and scale sweep suite [run_experiments.py](run_experiments.py). It:

1. Builds calibration datasets (150 samples/domain) and held-out evaluation sets (150 samples/domain).
2. Measures baseline shift cross-entropy NLL and perplexity for the base model and individual experts.
3. Computes explicit task vector interference metrics (cosine similarity, 72.01% sign conflict ratio, task vector norms).
4. Executes multi-scale sweeps ($\lambda \in [0.0, 1.0]$) across merge baselines (Simple Task Arithmetic, Plain Averaging, TIES-only, DARE+TIES, Fisher-only, and Full ExFusion).
5. Computes Domain Retention Scores $R_d(\lambda)$ and saves machine-readable JSON artifacts (`artifacts/experiment_results.json`).

Run it via:

```bash
python3 run_experiments.py
```

---

## 🧪 Verification & Testing

The workspace features comprehensive self-testing scripts verifying all 5 router paths, gradient propagation through STE, rules engines, and model merging routines.

### Run Unit Tests

```bash
python3 test_nesy_v1_0.py
```

### Run Model Merging & ExFusion Engine Self-Tests

```bash
python3 daph_hybrid_exfusion_v2_3.py
```

### Test Results

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

## 📁 Repository Directory Structure

```
.
├── daph_nesy_v1_0.py          # Primary NeSy-MoE v1.1 Extended module
├── daph_hybrid_exfusion_v2_3.py # Base DAPH Mamba/Attention Hybrid & Model Merging
├── test_nesy_v1_0.py          # Live test suite (11 comprehensive test groups)
├── benchmark_nesy.py          # Automated performance & VRAM profiling suite
├── run_experiments.py         # Quantitative experiment runner & RESULTS.md generator
├── run_model.py               # Pretrained Hugging Face model runner & NeSy pipeline
├── run_model_merge.py         # Multi-expert Hugging Face model merging runner
├── artifacts/                 # Saved experiment evaluation artifacts & JSON sweep data
├── RESULTS.md                 # Official live test, benchmark & merging results
├── ISSUES.md                  # Issues tracking & architectural roadmap
├── README.md                  # Detailed architecture & API documentation
├── LICENSE                    # Apache License 2.0
└── .gitignore                 # Python & PyTorch cache ignores
```

---

## 𝛴 Mathematical Foundations

### 1. Symbolic Prior Logit Additivity

Given neural router logits $\mathbf{z}_{\text{neural}} \in \mathbb{R}^P$ and symbolic prior bias $\mathbf{b}_{\text{symbolic}} \in \mathbb{R}^P$:
$$\mathbf{p}_{\text{route}} = \text{Softmax}(\mathbf{z}_{\text{neural}} + \mathbf{b}_{\text{symbolic}})$$

### 2. Straight-Through Estimator (STE) Re-Embedding

Let $\mathbf{h} \in \mathbb{R}^H$ be the input hidden state, $W_{\text{de}} \in \mathbb{R}^{V \times H}$ the de-embedding weight, and $E \in \mathbb{R}^{V \times H}$ the token embedding matrix.
$$\mathbf{p}_{\text{vocab}} = \text{Softmax}(W_{\text{de}} \mathbf{h})$$
$$\hat{y} = \text{Argmax}(\mathbf{p}_{\text{vocab}}) \xrightarrow{\text{Solver}} y^*$$
$$\mathbf{e}_{\text{STE}} = \left( \text{OneHot}(y^*) - \mathbf{p}_{\text{vocab}} \right).\text{detach}() + \mathbf{p}_{\text{vocab}}$$
$$\mathbf{h}_{\text{symbolic}} = \mathbf{e}_{\text{STE}} E$$

### 3. Context Preservation Gating

$$\mathbf{h}_{\text{out}} = \text{LayerNorm}\left( (1 - \sigma(\alpha)) \cdot \mathbf{h}_{\text{symbolic}} + \sigma(\alpha) \cdot \mathbf{h} \right)$$
where $\alpha \in \mathbb{R}$ is a learnable gate initialized to $0.1$.

### 4. ExFusion Model Construction Optimization Objective

For base model parameters $\theta_B$ and expert task vectors $\Delta_i = \theta_i - \theta_B$:
$$\max_{\theta^*} \left[ \sum_{d=1}^D R_d(\theta^*) - \alpha \cdot I(\theta^*) - \beta \cdot D_B(\theta^*) \right]$$
where Domain Retention Score $R_d(\lambda)$ is defined over shift cross-entropy NLL:
$$R_d(\lambda) = \frac{\text{NLL}_{\text{base}, d} - \text{NLL}_{\text{merged}, d}(\lambda)}{\text{NLL}_{\text{base}, d} - \text{NLL}_{\text{expert}, d}} \times 100\%$$

### 5. TIES v2 Pure Sign Consensus & Fisher Weighting

Given trimmed task vectors $\tilde{\Delta}_i$ and expert weights $w_i$:
$$\text{Vote} = \sum_{i=1}^E w_i \cdot \text{Sign}(\tilde{\Delta}_i), \quad \text{ElectedSign} = \text{Sign}(\text{Vote})$$
$$\Delta_{\text{TIES}} = \frac{\sum_{i=1}^E w_i \cdot \tilde{\Delta}_i \cdot \mathbf{1}_{\text{Sign}(\tilde{\Delta}_i) = \text{ElectedSign}}}{\max\left(\sum_{i=1}^E w_i \cdot \mathbf{1}_{\text{Sign}(\tilde{\Delta}_i) = \text{ElectedSign}}, \epsilon\right)}$$
Empirical Fisher diagonal matrix elements $F_{kk} = \frac{1}{N} \sum_{n=1}^N \left( \frac{\partial \mathcal{L}_n}{\partial \theta_k} \right)^2$ modulate the Fisher merged delta:
$$\Delta_{\text{Fisher}} = \frac{\sum_{i=1}^E w_i \cdot F_{i, kk}^\gamma \cdot \tilde{\Delta}_{i, k}}{\max\left(\sum_{i=1}^E w_i \cdot F_{i, kk}^\gamma \cdot M_{\text{DARE}, i, k}, \epsilon\right)}$$

---

## 📄 License & Citation

Distributed under the **Apache License 2.0**. See [LICENSE](LICENSE) for more information.

```bibtex
@software{daph_nesy_moe_2026,
  author = {Dawson Block},
  title = {DAPH NeSy-MoE: Differentiable Adaptive Predictive Hybrid Neurosymbolic Mixture-of-Experts},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dawsonblock/Daph_fusion}}
}
```
