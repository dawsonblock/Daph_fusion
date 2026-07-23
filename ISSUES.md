# DAPH NeSy-MoE & ExFusion Open Issues & Architectural Roadmap

This document tracks identified open issues, strategic enhancement requests, and architectural roadmap items for **DAPH NeSy-MoE v1.1 Extended** and **DAPH ExFusion Hybrid v2.3**.

---

## 📌 Open Issues & Feature Requests

### Issue 1: Asymmetric Hard-Routing Compute Optimization (Sparse Mamba/Attention Dispatch)
* **Status**: Open
* **Priority**: High | **Effort**: High
* **Category**: Performance / Sparse Dispatch
* **Description**:
  For pointwise paths (`Trans-ExFusion`, `CheapPath`), token-level hard routing uses sparse gather/scatter. However, sequence-dependent paths (`MultiheadAttention`, `MemoryBankExFusionMamba`) currently execute dense linear projections across the entire sequence. Setting $\text{dt}=0$ freezes SSM state updates, but linear projections and memory bandwidth are still consumed for non-routed tokens.
* **Proposed Solution**:
  Implement sparse token dispatching or fused masked projection wrappers for non-routed sequence tokens during hard-routing passes.

---

### Issue 2: Subword BPE/SentencePiece Token Bridge for Domain Solvers
* **Status**: Open
* **Priority**: High | **Effort**: Medium
* **Category**: System 2 Reasoning / Tokenization
* **Description**:
  Built-in domain solvers operate on character byte/ASCII representations (`'0'`–`'9'`). Modern subword tokenizers (BPE, SentencePiece) chunk numbers or code tokens into multi-character subwords. While single-token subword vocabulary mapping is precomputed in `VectorizedSymbolicExpert.build_subword_vocab_map()`, multi-token subword expressions require a subword sequence bridge.
* **Proposed Solution**:
  Extend `VectorizedSymbolicExpert` with a `SubwordSequenceBridge` that decodes multi-token subword spans into string representations, runs domain solvers, and re-encodes back into discrete token IDs before STE re-embedding.

---

### Issue 3: Native Fused Triton Kernel Integration for Mamba Selective Scan
* **Status**: Open
* **Priority**: Medium | **Effort**: Medium
* **Category**: GPU Kernel Acceleration
* **Description**:
  The default PyTorch scan fallback executes sequential loops in Python when native Triton/CUDA scan binaries are unavailable.
* **Proposed Solution**:
  Register native C++/Triton `mamba_ssm.ops.selective_scan_fn` bindings via `register_scan_backend("triton", ...)` for high-throughput production GPU environments.

---

### Issue 4: Multi-Layer Macro-Routing Topology & Expert Layer Parallelism
* **Status**: Open
* **Priority**: Medium | **Effort**: High
* **Category**: Architecture / Scalability
* **Description**:
  Evaluating symbolic experts across all layers in deep stacks (e.g. 24 or 32 layers) accumulates compute overhead.
* **Proposed Solution**:
  Assign symbolic experts to strategic intermediate or deep reasoning layers while shallow layers handle fast Mamba-2 SSM context pre-processing. Add PyTorch Tensor Parallelism (`tp`) or Expert Parallelism (`ep`) for multi-GPU scaling.

---

### Issue 5: Memory-Mapped & Offloaded Fisher Information Matrix Diagonal Computation
* **Status**: Open
* **Priority**: Medium | **Effort**: High
* **Category**: Infrastructure / Model Merging
* **Description**:
  Calculating and accumulating empirical Fisher diagonals for 70B+ parameter models on GPU/RAM can cause memory pressure.
* **Proposed Solution**:
  Implement disk-backed memory mapping (`torch.mmap`) and chunked offloaded parameter buffers during empirical Fisher diagonal computation.

---

### Issue 6: Multi-Token Output Verifier & Grammar Logit Masking Expansion
* **Status**: Open
* **Priority**: Medium | **Effort**: Medium
* **Category**: System 2 Guardrails
* **Description**:
  `NeSyOutputVerifier` currently enforces balanced brackets. Extending guardrails to complex structured formats (JSON schema, SQL syntax, Python AST) will prevent syntax errors at generation time.
* **Proposed Solution**:
  Extend `NeSyOutputVerifier` with stateful FSM (Finite State Machine) logit masks for JSON schemas, SQL query syntax, and Python AST grammars.

---

### Issue 7: Automated Performance Benchmarking & VRAM Profiling Suite
* **Status**: Open
* **Priority**: Low | **Effort**: Medium
* **Category**: Testing / Benchmarking
* **Description**:
  Comprehensive throughput and memory benchmarks across batch sizes $B \in \{1, 8, 32\}$ and sequence lengths $L \in \{512, 2048, 8192\}$ are needed.
* **Proposed Solution**:
  Create `benchmark_nesy.py` to measure tokens/sec throughput, latency, and peak VRAM allocation across 5-path hard vs. soft macro-routing configurations.

---

## 🛠 Completed Issues & Fixes (v1.1 Extended)

- [x] **[Fix] DARE Pre-scaling Double-Compensation Bug**: Added `rescale_deltas=False` option to `apply_dare_preprocessing` to prevent double-scaling expert deltas during normalized weighted merges.
- [x] **[Fix] Hardened SSM Parameter Identification**: Updated `is_ssm_core_param` fallback policies and guarded `D` suffix matches to Mamba/SSM module names.
- [x] **[Fix] Attention Sink Positional Index Tracking**: Added `meta["attn_position_ids"]` in `DAPHHybridDecoderLayer.forward` to preserve spatial position IDs during KV cache trimming.
- [x] **[Perf] Duplicate Symbolic Expert Forward Pass Elimination**: Added `_get_cached_symbolic_out()` in `NeSyDecoderLayer` to prevent redundant expert execution.
- [x] **[Perf] Pre-buffered Token Rule Tensors**: Added `_update_tensor_buffers()` in `TokenizerBoundRulesEngine` for zero-allocation rule evaluation.
- [x] **[Feature] GPU-Native Subword Vocabulary Lookup**: Implemented `build_subword_vocab_map()` in `VectorizedSymbolicExpert`.
- [x] **[Refactor] Pytest Auto-Discovery**: Refactored `test_nesy_v1_0.py` into 8 auto-discovered Pytest functions.
