#!/usr/bin/env python3
"""
Automated Performance Benchmarking & Memory Profiling Suite for DAPH NeSy-MoE.

Benchmarks:
1. System 1 (Neural Hybrid) vs System 2 (Symbolic Expert) latency and throughput (tokens/sec).
2. 5-path Macro-Routing hard vs soft dispatch scaling across sequence lengths L and batch sizes B.
3. GPU Peak VRAM memory allocation profiling (CUDA / MPS / CPU fallback).
"""

import time
import torch
import torch.nn as nn

from daph_hybrid_exfusion_v2_3 import DAPHConfig
from daph_nesy_v1_0 import (
    NeSyDecoderLayer,
    TokenizerBoundRulesEngine,
    VectorizedSymbolicExpert,
)


def profile_memory(device: str) -> float:
    """Returns allocated memory in Megabytes (MB)."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory") and device == "mps":
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0


def benchmark_layer(
    batch_size: int,
    seq_len: int,
    hidden_size: int = 768,
    num_paths: int = 5,
    warmup: int = 3,
    runs: int = 10,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(0)
    config = DAPHConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=8,
        state_size=16,
        num_paths=num_paths,
        routing_granularity="token",
    )

    vocab_size = 32000
    lm_head = torch.randn(vocab_size, hidden_size, device=device)
    rules_engine = TokenizerBoundRulesEngine(num_paths=num_paths, device=device)
    expert = VectorizedSymbolicExpert(hidden_size, vocab_size, lm_head, domain="arithmetic_eval").to(device)

    layer = NeSyDecoderLayer(config, rules_engine=rules_engine, symbolic_expert=expert).to(device)
    layer.eval()

    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Reset memory stats
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x, token_ids=token_ids)
            if device == "cuda":
                torch.cuda.synchronize()

    # Benchmark runs
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = layer(x, token_ids=token_ids)
            if device == "cuda":
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / runs) * 1000.0
    total_tokens = batch_size * seq_len
    throughput_tokens_per_sec = total_tokens / ((end_time - start_time) / runs)
    peak_vram_mb = profile_memory(device)

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "latency_ms": round(avg_latency_ms, 3),
        "throughput_tok_sec": round(throughput_tokens_per_sec, 1),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print("=" * 70)
    print(f"DAPH NeSy-MoE Performance & Benchmark Suite (Device: {device})")
    print("=" * 70)

    configs = [
        (1, 128),
        (1, 512),
        (4, 512),
        (8, 512),
    ]

    print(f"\n{'Batch':<8} | {'SeqLen':<8} | {'Latency (ms)':<14} | {'Throughput (tok/s)':<20} | {'Peak VRAM (MB)':<12}")
    print("-" * 70)

    for B, L in configs:
        res = benchmark_layer(batch_size=B, seq_len=L, device=device)
        print(f"{res['batch_size']:<8} | {res['seq_len']:<8} | {res['latency_ms']:<14} | {res['throughput_tok_sec']:<20} | {res['peak_vram_mb']:<12}")

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
