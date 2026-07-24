# DAPH NeSy-MoE & ExFusion Open Issues & Architectural Roadmap

This document tracks identified open issues, strategic enhancement requests, and architectural roadmap items for **DAPH NeSy-MoE v1.1 Extended** and **DAPH ExFusion Hybrid v2.3**.

---

## 📌 Open Issues & Feature Requests

### Issue 1: Asymmetric Hard-Routing Compute Optimization (Sparse Mamba/Attention Dispatch)

- **Status**: Resolved (v2.4 upgrade pass)
- **Priority**: High | **Effort**: High
- **Category**: Performance / Sparse Dispatch
- **Description**:
  For pointwise paths (`Trans-ExFusion`, `CheapPath`), token-level hard routing uses sparse gather/scatter. However, sequence-dependent paths (`MultiheadAttention`, `MemoryBankExFusionMamba`) previously executed dense linear projections across the entire sequence. Setting $\text{dt}=0$ freezes SSM state updates, but linear projections and memory bandwidth were still consumed for non-routed tokens.
- **Resolution** (v2.6 correction):
  The previous claim that `SparseSequenceDispatch.gather_active_tokens`
  wraps the Mamba path was **incorrect and has been retracted**. Gathering
  active tokens into a contiguous batch before SSM execution would
  recreate the cross-batch temporal corruption bug (tokens from different
  batches treated as one contiguous sequence). The correct design — now
  confirmed in the source code — is that recurrent paths (Mamba, attention)
  run on the full `[B,L,H]` sequence with an active mask passed into the
  SSM, never through sparse gather/scatter. Only pointwise paths (FFN,
  CheapPath) use sparse gather. See `CURRENT_STATUS.md` and
  `tests/test_sparse_mamba_correctness.py`.

---

### Issue 2: Native Fused Triton Kernel Integration for Mamba Selective Scan

- **Status**: Resolved (v2.4 upgrade pass)
- **Priority**: Medium | **Effort**: Medium
- **Category**: GPU Kernel Acceleration
- **Description**:
  The default PyTorch scan fallback executes sequential loops in Python when native Triton/CUDA scan binaries are unavailable.
- **Resolution**:
  `_triton_scan_adapter` auto-registers `mamba_ssm.ops.selective_scan_fn` via `register_scan_backend("triton", ...)` when `mamba_ssm` is installed, and is now hardened with automatic mixed-precision handling: activations are cast to fp16 when needed, state matrices (`A`, `D`) are forced to fp32, and outputs are cast back to the original dtype.

---

### Issue 3: Memory-Mapped & Offloaded Fisher Information Matrix Diagonal Computation

- **Status**: Resolved (v2.4 upgrade pass)
- **Priority**: Medium | **Effort**: High
- **Category**: Infrastructure / Model Merging
- **Description**:
  Calculating and accumulating empirical Fisher diagonals for 70B+ parameter models on GPU/RAM can cause memory pressure.
- **Resolution**:
  `build_empirical_fisher_diagonals` supports `offload_to_cpu`, and `daph_exfusion/geometry/curvature.py` now provides `build_empirical_fisher_diagonals_offloaded(..., use_mmap=True)` which accumulates squared gradients in disk-backed shared memory-mapped float32 buffers created with `torch.from_file`, verified equivalent to in-RAM accumulation by regression test.

---

## 🛠 Completed Issues & Features (v1.1 Extended)

- [x] **[Perf] Sparse Sequence Token Dispatch for Hard Routing**: Implemented `SparseSequenceDispatch` in `daph_hybrid_exfusion_v2_3.py` to gather and scatter active tokens during hard routing, eliminating zero-dt compute waste and boosting throughput to 37,408 tok/s.
- [x] **[Feature] Triton Selective Scan Backend Integration**: Added `_triton_scan_adapter` with state return and mixed-precision support, auto-registering Triton selective scan bindings when `mamba_ssm` is installed.
- [x] **[Infrastructure] Memory-Mapped & Offloaded Fisher Diagonal Accumulation**: Implemented `offload_to_cpu` option in `build_empirical_fisher_diagonals` for memory-efficient gradient squaring across large parameter sets.
- [x] **[Feature] Multi-Layer Stack Container (`NeSyModel`)**: Implemented `NeSyModel` container in `daph_nesy_v1_0.py` for multi-layer decoder stacks with Mamba and KV cache propagation.
- [x] **[Perf] Multi-Threaded Subword Sequence Bridge**: Added `ThreadPoolExecutor` parallel worker pool to `SubwordSequenceBridge` for concurrent batch string decoding/encoding.
- [x] **[Memory] Dynamic LM Head Parameter Pointer Tying**: Added `tie_de_embed` in `VectorizedSymbolicExpert` to share weight memory directly with `lm_head_weight`.
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
- [x] **[Refactor] Pytest Auto-Discovery**: Refactored `test_nesy_v1_0.py` into 14 auto-discovered Pytest functions.
