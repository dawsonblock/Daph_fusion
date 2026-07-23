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

### Issue 2: Native Fused Triton Kernel Integration for Mamba Selective Scan
* **Status**: Open
* **Priority**: Medium | **Effort**: Medium
* **Category**: GPU Kernel Acceleration
* **Description**:
  The default PyTorch scan fallback executes sequential loops in Python when native Triton/CUDA scan binaries are unavailable.
* **Proposed Solution**:
  Register native C++/Triton `mamba_ssm.ops.selective_scan_fn` bindings via `register_scan_backend("triton", ...)` for high-throughput production GPU environments.

---

### Issue 3: Memory-Mapped & Offloaded Fisher Information Matrix Diagonal Computation
* **Status**: Open
* **Priority**: Medium | **Effort**: High
* **Category**: Infrastructure / Model Merging
* **Description**:
  Calculating and accumulating empirical Fisher diagonals for 70B+ parameter models on GPU/RAM can cause memory pressure.
* **Proposed Solution**:
  Implement disk-backed memory mapping (`torch.mmap`) and chunked offloaded parameter buffers during empirical Fisher diagonal computation.

---

## 🛠 Completed Issues & Features (v1.1 Extended)

- [x] **[Feature] Subword Sequence Bridge for Multi-Token Solvers**: Implemented `SubwordSequenceBridge` in `daph_nesy_v1_0.py` to handle multi-token subword string decoding, string-level domain solvers, and re-encoding.
- [x] **[Feature] Expanded Grammar Output Verifiers**: Added `JSONOutputVerifier`, `SQLOutputVerifier`, and `FSMGrammarVerifier` in `daph_nesy_v1_0.py` for structured logit masking during decoding.
- [x] **[Feature] Multi-Layer Selective Routing Topology**: Added `layer_idx` and `active_symbolic_layers` support in `NeSyDecoderLayer` to bypass symbolic overhead on inactive layers.
- [x] **[Feature] Automated Performance Benchmarking & VRAM Profiling Suite**: Created `benchmark_nesy.py` to profile latency, throughput (tokens/sec), and peak VRAM allocation.
- [x] **[Fix] DARE Pre-scaling Double-Compensation Bug**: Added `rescale_deltas=False` option to `apply_dare_preprocessing` to prevent double-scaling expert deltas during normalized weighted merges.
- [x] **[Fix] Hardened SSM Parameter Identification**: Updated `is_ssm_core_param` fallback policies and guarded `D` suffix matches to Mamba/SSM module names.
- [x] **[Fix] Attention Sink Positional Index Tracking**: Added `meta["attn_position_ids"]` in `DAPHHybridDecoderLayer.forward` to preserve spatial position IDs during KV cache trimming.
- [x] **[Perf] Duplicate Symbolic Expert Forward Pass Elimination**: Added `_get_cached_symbolic_out()` in `NeSyDecoderLayer` to prevent redundant expert execution.
- [x] **[Perf] Pre-buffered Token Rule Tensors**: Added `_update_tensor_buffers()` in `TokenizerBoundRulesEngine` for zero-allocation rule evaluation.
- [x] **[Feature] GPU-Native Subword Vocabulary Lookup**: Implemented `build_subword_vocab_map()` in `VectorizedSymbolicExpert`.
- [x] **[Refactor] Pytest Auto-Discovery**: Refactored `test_nesy_v1_0.py` into 11 auto-discovered Pytest functions.
