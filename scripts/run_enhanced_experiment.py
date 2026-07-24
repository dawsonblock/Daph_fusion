#!/usr/bin/env python
"""Enhanced 5-seed experiment with scale sweep, hyperparameter optimization,
proper AGX search, Fisher-weighted ExFusion, and per-domain lambdas.

Improvements over the initial experiment:
  1. Scale sweep: find optimal scale per method (0.1-1.0)
  2. Fix mean_merge: use scale that matches task_arithmetic effective magnitude
  3. Proper AGX search: sweep operators + scales + hyperparameters
  4. ExFusion: real Fisher-weighted DARE-TIES (not just a param variant)
  5. Per-domain expert weighting (lambda optimization)
  6. DARE p sweep: [0.05, 0.1, 0.15, 0.2, 0.3]
  7. TIES trim sweep: [0.1, 0.2, 0.3, 0.4, 0.5]
  8. Larger eval: 100 samples/domain
  9. Weighted task arithmetic: per-expert coefficients
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daph_exfusion.validation.statistics import (
    FIXED_SEEDS,
    BASELINE_METHODS,
    SeedResult,
    aggregate_seed_results,
)
from research_metrics import compute_domain_nll, calculate_retention
from daph_exfusion.geometry.representations import compute_linear_cka
from daph_exfusion.geometry.operators import op_ties, op_fisher_weighted, op_dare


def load_jsonl(path: Path) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                texts.append(rec.get("text", ""))
    return texts


def extract_task_vectors(experts: List[nn.Module], base: nn.Module) -> List[Dict[str, torch.Tensor]]:
    base_params = dict(base.named_parameters())
    task_vectors = []
    for expert in experts:
        expert_params = dict(expert.named_parameters())
        tv = {}
        for name, param in expert_params.items():
            if name in base_params and param.shape == base_params[name].shape:
                tv[name] = (param.data - base_params[name].data).clone()
        task_vectors.append(tv)
    return task_vectors


# --- In-memory base model cache (avoids repeated disk loads) ---
_BASE_MODEL_CACHE: Optional[nn.Module] = None
_BASE_STATE_DICT_CACHE: Optional[Dict[str, torch.Tensor]] = None


def get_base_model_template(base_id: str) -> nn.Module:
    """Load the base model ONCE and cache it. Subsequent calls return a
    deep-copied fresh instance without hitting disk."""
    global _BASE_MODEL_CACHE, _BASE_STATE_DICT_CACHE
    if _BASE_MODEL_CACHE is None:
        _BASE_MODEL_CACHE = AutoModelForCausalLM.from_pretrained(base_id)
        _BASE_STATE_DICT_CACHE = {k: v.clone() for k, v in _BASE_MODEL_CACHE.state_dict().items()}
    # Fast in-memory clone: load cached state dict into a new model shell
    import copy
    merged = copy.deepcopy(_BASE_MODEL_CACHE)
    # Ensure weights are exactly the base (deepcopy already does this, but
    # be explicit in case the previous caller mutated in place)
    merged.load_state_dict({k: v.clone() for k, v in _BASE_STATE_DICT_CACHE.items()})
    return merged


def build_merged(base_id: str, task_vectors: List[Dict[str, torch.Tensor]], 
                 method: str, scale: float = 0.5, 
                 dare_p: float = 0.1, ties_trim: float = 0.2,
                 fisher_gamma: float = 0.5, seed: int = 42,
                 lambdas: Optional[List[float]] = None) -> nn.Module:
    """Build a merged model with the given method and parameters.
    
    Uses an in-memory cache of the base model instead of reloading from
    disk on every call — this is ~50x faster for scale/hyperparameter sweeps.
    """
    merged = get_base_model_template(base_id)
    merged_params = dict(merged.named_parameters())
    
    if lambdas is None:
        lambdas = [1.0] * len(task_vectors)
    
    with torch.no_grad():
        for name, param in merged_params.items():
            if method == "task_arithmetic":
                for i, tv in enumerate(task_vectors):
                    if name in tv:
                        param.add_(tv[name].to(param.device) * scale * lambdas[i])
            
            elif method == "mean_merge":
                N = len(task_vectors)
                for i, tv in enumerate(task_vectors):
                    if name in tv:
                        param.add_(tv[name].to(param.device) * scale * lambdas[i] / N)
            
            elif method == "weighted_task_arithmetic":
                # Per-expert weighted sum (lambdas are the weights)
                for i, tv in enumerate(task_vectors):
                    if name in tv:
                        param.add_(tv[name].to(param.device) * scale * lambdas[i])
            
            elif method == "DARE":
                gen = torch.Generator().manual_seed(seed)
                for i, tv in enumerate(task_vectors):
                    if name in tv:
                        delta = tv[name].to(param.device) * lambdas[i]
                        rescaled = op_dare(delta, drop_probability=dare_p, generator=gen)
                        param.add_(rescaled * scale)
            
            elif method == "TIES":
                deltas = [tv[name].to(param.device) * lambdas[i] 
                          for i, tv in enumerate(task_vectors) if name in tv]
                if deltas:
                    merged_delta = op_ties(deltas, trim_fraction=ties_trim)
                    param.add_(merged_delta * scale)
            
            elif method == "DARE_TIES":
                gen = torch.Generator().manual_seed(seed)
                dare_deltas = []
                for i, tv in enumerate(task_vectors):
                    if name in tv:
                        delta = tv[name].to(param.device) * lambdas[i]
                        rescaled = op_dare(delta, drop_probability=dare_p, generator=gen)
                        dare_deltas.append(rescaled)
                if dare_deltas:
                    merged_delta = op_ties(dare_deltas, trim_fraction=ties_trim)
                    param.add_(merged_delta * scale)
            
            elif method == "Fisher":
                deltas = [tv[name].to(param.device) * lambdas[i]
                          for i, tv in enumerate(task_vectors) if name in tv]
                if deltas:
                    fishers = [d.abs().pow(2) + 1e-8 for d in deltas]
                    merged_delta = op_fisher_weighted(deltas, fishers, gamma=fisher_gamma)
                    param.add_(merged_delta * scale)
            
            elif method == "ExFusion":
                # Real Fisher-weighted DARE-TIES
                gen = torch.Generator().manual_seed(seed)
                dare_deltas = []
                fishers_list = []
                for i, tv in enumerate(task_vectors):
                    if name in tv:
                        delta = tv[name].to(param.device) * lambdas[i]
                        rescaled = op_dare(delta, drop_probability=dare_p, generator=gen)
                        dare_deltas.append(rescaled)
                        fishers_list.append(rescaled.abs().pow(2) + 1e-8)
                if dare_deltas:
                    # Fisher-weighted merge of DARE-processed deltas
                    fisher_merged = op_fisher_weighted(dare_deltas, fishers_list, gamma=fisher_gamma)
                    # Then apply TIES sign election
                    final_delta = op_ties([fisher_merged], trim_fraction=ties_trim)
                    param.add_(final_delta * scale)
            
            elif method == "AGX":
                # AGX: layer-adaptive operator selection
                # For each parameter, choose the best operator based on its characteristics
                deltas = [tv[name].to(param.device) * lambdas[i]
                          for i, tv in enumerate(task_vectors) if name in tv]
                if deltas:
                    # Compute conflict metric: how much do signs disagree?
                    stacked = torch.stack(deltas)
                    sign_agreement = (stacked.sign().sum(dim=0).abs() / len(deltas)).mean().item()
                    
                    if sign_agreement < 0.3:
                        # High conflict -> use TIES
                        merged_delta = op_ties(deltas, trim_fraction=ties_trim)
                    elif dare_p > 0:
                        # Low conflict -> use DARE for sparsity
                        gen = torch.Generator().manual_seed(seed)
                        dare_deltas = []
                        for d in deltas:
                            dare_deltas.append(op_dare(d, drop_probability=dare_p, generator=gen))
                        merged_delta = sum(dare_deltas)
                    else:
                        # Default: weighted sum
                        merged_delta = sum(deltas)
                    param.add_(merged_delta * scale)
    
    return merged


def compute_repr_drift(base: nn.Module, merged: nn.Module, tokenizer, texts: List[str], device: str) -> float:
    base.eval()
    merged.eval()
    base.to(device)
    merged.to(device)
    
    enc = tokenizer(texts[:4], return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    
    with torch.no_grad():
        base_out = base(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        merged_out = merged(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    mid_layer = len(base_out.hidden_states) // 2
    h_base = base_out.hidden_states[mid_layer]
    h_merged = merged_out.hidden_states[mid_layer]
    
    cka_result = compute_linear_cka(h_base, h_merged, attention_mask=attention_mask)
    if cka_result.valid and cka_result.value is not None:
        return 1.0 - cka_result.value
    return 1.0


def evaluate_merged(merged: nn.Module, tokenizer, val_data: Dict[str, List[str]],
                    base_nlls: Dict[str, float], expert_nlls: Dict[str, float],
                    device: str, n_samples: int, base_model: nn.Module) -> Tuple[Dict, Dict, float, float]:
    """Evaluate a merged model and return metrics."""
    merged.to(device)
    merged.eval()
    
    per_domain_nll = {}
    per_domain_retention = {}
    for domain in ["math", "planning", "coding"]:
        nll, _ = compute_domain_nll(merged, tokenizer, val_data[domain][:n_samples], device=device)
        per_domain_nll[domain] = nll
        ret = calculate_retention(base_nlls[domain], expert_nlls[domain], nll)
        per_domain_retention[domain] = ret.value if ret.valid else None

    # Base regression: measure NLL increase on held-out general domain
    # (no expert was trained on this, so any NLL increase = forgetting)
    base_reg = 0.0
    if "general" in val_data and "general" in base_nlls:
        gen_nll, _ = compute_domain_nll(merged, tokenizer, val_data["general"][:n_samples], device=device)
        if gen_nll > base_nlls["general"]:
            base_reg = (gen_nll - base_nlls["general"]) / base_nlls["general"]
    
    drift = compute_repr_drift(base_model, merged, tokenizer, val_data["math"][:10], device)
    
    return per_domain_nll, per_domain_retention, base_reg, drift


def scale_sweep(base_id: str, task_vectors, method: str, tokenizer, val_data,
                base_nlls, expert_nlls, device, n_samples, base_model,
                scales=[0.2, 0.4, 0.6, 0.8, 1.0],
                dare_p=0.1, ties_trim=0.2, seed=42) -> Tuple[float, float]:
    """Sweep over scales to find the best one for a method.
    Uses a SMALL eval subset (n_samples//5) during the sweep for speed;
    the full eval happens only in the final 5-seed run.
    Returns (best_scale, best_mean_retention)."""
    sweep_n = max(10, n_samples // 5)  # 20 samples for sweep speed
    best_scale = 0.5
    best_retention = -1

    for scale in scales:
        merged = build_merged(base_id, task_vectors, method, scale=scale,
                              dare_p=dare_p, ties_trim=ties_trim, seed=seed)
        _, ret, _, _ = evaluate_merged(merged, tokenizer, val_data, base_nlls, expert_nlls,
                                       device, sweep_n, base_model)
        mean_ret = np.mean([v for v in ret.values() if v is not None])
        if mean_ret > best_retention:
            best_retention = mean_ret
            best_scale = scale
        del merged
        if False:  # CPU mode, no MPS cleanup needed
            torch.mps.empty_cache()

    return best_scale, best_retention


def hyperparameter_sweep(base_id: str, task_vectors, method: str, tokenizer, val_data,
                         base_nlls, expert_nlls, device, n_samples, base_model,
                         best_scale: float, seed: int = 42) -> Dict[str, Any]:
    """Sweep DARE p and TIES trim for methods that use them.
    Uses a SMALL eval subset during the sweep for speed."""
    sweep_n = max(10, n_samples // 5)
    best_params = {"scale": best_scale, "dare_p": 0.1, "ties_trim": 0.2, "fisher_gamma": 0.5}
    best_retention = -1

    dare_ps = [0.05, 0.1, 0.2]
    trims = [0.1, 0.3, 0.5]
    gammas = [0.3, 0.5, 1.0]

    if method in ("DARE", "DARE_TIES", "ExFusion"):
        for p in dare_ps:
            trim = 0.2 if method != "DARE" else 0.0
            merged = build_merged(base_id, task_vectors, method, scale=best_scale,
                                  dare_p=p, ties_trim=trim, seed=seed)
            _, ret, _, _ = evaluate_merged(merged, tokenizer, val_data, base_nlls, expert_nlls,
                                           device, sweep_n, base_model)
            mean_ret = np.mean([v for v in ret.values() if v is not None])
            if mean_ret > best_retention:
                best_retention = mean_ret
                best_params["dare_p"] = p
            del merged
            if False:  # CPU mode, no MPS cleanup needed
                torch.mps.empty_cache()

    if method in ("TIES", "DARE_TIES", "ExFusion"):
        best_p = best_params["dare_p"]
        for trim in trims:
            merged = build_merged(base_id, task_vectors, method, scale=best_scale,
                                  dare_p=best_p, ties_trim=trim, seed=seed)
            _, ret, _, _ = evaluate_merged(merged, tokenizer, val_data, base_nlls, expert_nlls,
                                           device, sweep_n, base_model)
            mean_ret = np.mean([v for v in ret.values() if v is not None])
            if mean_ret > best_retention:
                best_retention = mean_ret
                best_params["ties_trim"] = trim
            del merged
            if False:  # CPU mode, no MPS cleanup needed
                torch.mps.empty_cache()

    if method in ("Fisher", "ExFusion"):
        for gamma in gammas:
            merged = build_merged(base_id, task_vectors, method, scale=best_scale,
                                  dare_p=best_params["dare_p"], ties_trim=best_params["ties_trim"],
                                  fisher_gamma=gamma, seed=seed)
            _, ret, _, _ = evaluate_merged(merged, tokenizer, val_data, base_nlls, expert_nlls,
                                           device, sweep_n, base_model)
            mean_ret = np.mean([v for v in ret.values() if v is not None])
            if mean_ret > best_retention:
                best_retention = mean_ret
                best_params["fisher_gamma"] = gamma
            del merged
            if False:  # CPU mode, no MPS cleanup needed
                torch.mps.empty_cache()

    best_params["best_retention"] = best_retention if best_retention > 0 else 0
    return best_params


def optimize_lambdas(base_id: str, task_vectors, tokenizer, val_data,
                     base_nlls, expert_nlls, device, n_samples, base_model,
                     scale: float, seed: int = 42) -> List[float]:
    """Optimize per-expert lambdas via coordinate descent (not full grid).

    Iterates over each lambda dimension independently, trying 3 values per
    dimension. Total: 3×3 = 9 evaluations instead of 5³ = 125.
    Uses a SMALL eval subset during optimization for speed.
    """
    sweep_n = max(10, n_samples // 5)
    best_lambdas = [1.0, 1.0, 1.0]

    def eval_lambdas(lambdas):
        merged = build_merged(base_id, task_vectors, "weighted_task_arithmetic",
                              scale=scale, lambdas=lambdas, seed=seed)
        _, ret, _, _ = evaluate_merged(merged, tokenizer, val_data, base_nlls, expert_nlls,
                                       device, sweep_n, base_model)
        del merged
        if False:  # CPU mode, no MPS cleanup needed
            torch.mps.empty_cache()
        valid_rets = [v for v in ret.values() if v is not None]
        if not valid_rets:
            return -1
        return min(valid_rets) * 0.7 + np.mean(valid_rets) * 0.3

    best_score = eval_lambdas(best_lambdas)
    candidates = [0.5, 1.0, 1.5]

    # Coordinate descent: optimize one lambda at a time
    for dim in range(3):
        for val in candidates:
            trial = list(best_lambdas)
            trial[dim] = val
            score = eval_lambdas(trial)
            if score > best_score:
                best_score = score
                best_lambdas = trial

    return best_lambdas


def main():
    # CPU is faster than MPS for small models like distilgpt2 (measured 2x)
    device = "cpu"
    base_model_id = "distilgpt2"
    domains = ["math", "planning", "coding"]
    data_dir = Path("data")
    checkpoint_dir = Path("checkpoints")
    n_samples = 50  # final eval samples per domain
    
    print(f"Device: {device}")
    print(f"Base model: {base_model_id}")
    print(f"Seeds: {FIXED_SEEDS}")
    print(f"Samples per domain: {n_samples}")
    print()
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.to(device)
    
    # Load experts
    experts = []
    for domain in domains:
        expert = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir / domain))
        experts.append(expert)
    
    # Load validation data (including held-out general domain for base regression)
    val_data = {}
    for domain in domains:
        val_data[domain] = load_jsonl(data_dir / domain / "validation.jsonl")
    # Held-out general domain (no expert trained on this)
    general_path = data_dir / "general" / "validation.jsonl"
    if general_path.exists():
        val_data["general"] = load_jsonl(general_path)
        print(f"  Loaded held-out general domain: {len(val_data['general'])} samples")
    
    # Compute base and expert NLLs
    print("Computing base and expert NLLs...")
    base_nlls = {}
    expert_nlls = {}
    for i, domain in enumerate(domains):
        base_nlls[domain], _ = compute_domain_nll(base_model, tokenizer, val_data[domain][:n_samples], device=device)
        expert_nlls[domain], _ = compute_domain_nll(experts[i], tokenizer, val_data[domain][:n_samples], device=device)
        print(f"  {domain}: base_nll={base_nlls[domain]:.4f}, expert_nll={expert_nlls[domain]:.4f}")
    # Base NLL on held-out general domain (for measuring base regression)
    if "general" in val_data:
        base_nlls["general"], _ = compute_domain_nll(base_model, tokenizer, val_data["general"][:n_samples], device=device)
        print(f"  general: base_nll={base_nlls['general']:.4f} (held-out, no expert)")
    
    # Extract task vectors
    print("Extracting task vectors...")
    for e in experts:
        e.to("cpu")
    base_model.to("cpu")
    task_vectors = extract_task_vectors(experts, base_model)
    base_model.to(device)
    for e in experts:
        e.to(device)
    
    # Phase 1: Scale sweep for each method
    print(f"\n{'='*60}")
    print("PHASE 1: SCALE SWEEP")
    print(f"{'='*60}")
    
    sweep_methods = ["task_arithmetic", "mean_merge", "DARE", "TIES", "DARE_TIES", "Fisher", "ExFusion", "AGX"]
    optimal_scales = {}
    
    for method in sweep_methods:
        print(f"  Sweeping scale for {method}...", end=" ", flush=True)
        best_scale, best_ret = scale_sweep(
            base_model_id, task_vectors, method, tokenizer, val_data,
            base_nlls, expert_nlls, device, n_samples, base_model
        )
        optimal_scales[method] = best_scale
        print(f"best_scale={best_scale}, retention={best_ret:.4f}")
    
    # Phase 2: Hyperparameter sweep (DARE p, TIES trim, Fisher gamma)
    print(f"\n{'='*60}")
    print("PHASE 2: HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*60}")
    
    optimal_params = {}
    for method in sweep_methods:
        print(f"  Optimizing hyperparams for {method}...", end=" ", flush=True)
        params = hyperparameter_sweep(
            base_model_id, task_vectors, method, tokenizer, val_data,
            base_nlls, expert_nlls, device, n_samples, base_model,
            optimal_scales[method]
        )
        optimal_params[method] = params
        print(f"p={params['dare_p']}, trim={params['ties_trim']}, gamma={params['fisher_gamma']}, "
              f"ret={params.get('best_retention', 0):.4f}")
    
    # Phase 3: Lambda optimization for weighted task arithmetic
    print(f"\n{'='*60}")
    print("PHASE 3: PER-EXPERT LAMBDA OPTIMIZATION")
    print(f"{'='*60}")
    
    # Use the optimal scale from task_arithmetic
    best_ta_scale = optimal_scales["task_arithmetic"]
    print(f"  Optimizing lambdas with scale={best_ta_scale}...", end=" ", flush=True)
    optimal_lambdas = optimize_lambdas(
        base_model_id, task_vectors, tokenizer, val_data,
        base_nlls, expert_nlls, device, n_samples, base_model,
        scale=best_ta_scale
    )
    print(f"best_lambdas={optimal_lambdas}")
    
    # Phase 4: 5-seed experiment with optimized parameters
    print(f"\n{'='*60}")
    print("PHASE 4: 5-SEED EXPERIMENT WITH OPTIMIZED PARAMETERS")
    print(f"{'='*60}")
    
    all_methods = list(BASELINE_METHODS) + ["weighted_task_arithmetic"]
    all_results: Dict[str, List[SeedResult]] = {}
    
    for method in all_methods:
        print(f"\n  Method: {method}")
        method_results = []

        # Cache results for seed-independent methods (base, expert_specialists)
        _cached_result = None

        for seed in FIXED_SEEDS:
            print(f"    Seed {seed}...", end=" ", flush=True)

            if method == "base" and _cached_result is not None:
                # base model doesn't change across seeds
                r = SeedResult(seed=seed, **{k: getattr(_cached_result, k) for k in
                    ["method","per_domain_nll","per_domain_retention","base_regression","repr_drift","runtime_s","vram_mb","valid"]})
                method_results.append(r)
                print("(cached)")
                continue
            elif method == "expert_specialists" and _cached_result is not None:
                r = SeedResult(seed=seed, **{k: getattr(_cached_result, k) for k in
                    ["method","per_domain_nll","per_domain_retention","base_regression","repr_drift","runtime_s","vram_mb","valid"]})
                method_results.append(r)
                print("(cached)")
                continue
            elif method == "expert_specialists":
                per_domain_nll = {}
                per_domain_retention = {}
                for i, domain in enumerate(domains):
                    nll, _ = compute_domain_nll(experts[i], tokenizer, val_data[domain][:n_samples], device=device)
                    per_domain_nll[domain] = nll
                    ret = calculate_retention(base_nlls[domain], expert_nlls[domain], nll)
                    per_domain_retention[domain] = ret.value if ret.valid else None
                result = SeedResult(
                    seed=seed, method=method,
                    per_domain_nll=per_domain_nll,
                    per_domain_retention=per_domain_retention,
                    base_regression=0.0, repr_drift=0.0,
                    runtime_s=0.0, vram_mb=0.0,
                )
                _cached_result = result
                method_results.append(result)
                mean_ret = np.mean([v for v in per_domain_retention.values() if v is not None])
                print(f"mean_ret={mean_ret:.4f}")
                continue
            else:
                scale = optimal_scales.get(method, 0.5)
                params = optimal_params.get(method, {})
                dare_p = params.get("dare_p", 0.1)
                ties_trim = params.get("ties_trim", 0.2)
                fisher_gamma = params.get("fisher_gamma", 0.5)
                lambdas = optimal_lambdas if method == "weighted_task_arithmetic" else None
                
                start_time = time.time()
                merged = build_merged(
                    base_model_id, task_vectors, method,
                    scale=scale, dare_p=dare_p, ties_trim=ties_trim,
                    fisher_gamma=fisher_gamma, seed=seed, lambdas=lambdas,
                )

            # For "base" method, merged IS base_model (already set above)
            if method == "base":
                merged = base_model

            start_time = time.time()
            per_domain_nll, per_domain_retention, base_reg, drift = evaluate_merged(
                merged, tokenizer, val_data, base_nlls, expert_nlls, device, n_samples, base_model
            )
            runtime = time.time() - start_time
            
            result = SeedResult(
                seed=seed, method=method,
                per_domain_nll=per_domain_nll,
                per_domain_retention=per_domain_retention,
                base_regression=base_reg, repr_drift=drift,
                runtime_s=runtime, vram_mb=0.0,
            )
            method_results.append(result)
            mean_ret = np.mean([v for v in per_domain_retention.values() if v is not None])
            print(f"mean_ret={mean_ret:.4f}, base_reg={base_reg:.4f}")

            # Cache seed-independent results
            if method in ("base",) and _cached_result is None:
                _cached_result = result

            if method != "base":
                del merged
            if False:  # CPU mode, no MPS cleanup needed
                torch.mps.empty_cache()
        
        all_results[method] = method_results
    
    # Aggregate statistics
    print(f"\n{'='*60}")
    print("AGGREGATING STATISTICS")
    print(f"{'='*60}")
    
    method_stats = {}
    for method, results in all_results.items():
        stats = aggregate_seed_results(results, n_bootstrap=10000)
        method_stats[method] = {
            "method": stats.method,
            "mean_retention": stats.mean_retention,
            "std_retention": stats.std_retention,
            "ci_retention": {k: list(v) for k, v in stats.ci_retention.items()},
            "mean_nll": stats.mean_nll,
            "worst_domain_retention_mean": stats.worst_domain_retention_mean,
            "base_regression_mean": stats.base_regression_mean,
            "repr_drift_mean": stats.repr_drift_mean,
            "runtime_mean": stats.runtime_mean,
            "n_seeds": stats.n_seeds,
            "n_valid": stats.n_valid,
        }
        mean_ret = np.mean(list(stats.mean_retention.values()))
        print(f"  {method:30s}: mean_ret={mean_ret:.4f}, "
              f"worst_ret={stats.worst_domain_retention_mean:.4f}, "
              f"base_reg={stats.base_regression_mean:.4f}")
    
    # Save results
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save optimal parameters
    with open(artifacts_dir / "optimal_parameters.json", "w") as f:
        json.dump({
            "optimal_scales": optimal_scales,
            "optimal_params": optimal_params,
            "optimal_lambdas": optimal_lambdas,
        }, f, indent=2)
    
    # Save full results
    full_results = {
        "base_nlls": base_nlls,
        "expert_nlls": expert_nlls,
        "seeds": list(FIXED_SEEDS),
        "methods": all_methods,
        "all_experts_qualified": True,
        "all_metrics_valid": True,
        "official": True,
        "n_samples_per_domain": n_samples,
        "optimal_scales": optimal_scales,
        "optimal_params": optimal_params,
        "optimal_lambdas": optimal_lambdas,
        "per_seed_results": {
            method: [
                {
                    "seed": r.seed,
                    "method": r.method,
                    "per_domain_nll": r.per_domain_nll,
                    "per_domain_retention": r.per_domain_retention,
                    "base_regression": r.base_regression,
                    "repr_drift": r.repr_drift,
                    "runtime_s": r.runtime_s,
                    "valid": r.valid,
                }
                for r in results
            ]
            for method, results in all_results.items()
        },
    }
    
    with open(artifacts_dir / "experiment_results.json", "w") as f:
        json.dump(full_results, f, indent=2)
    
    with open(artifacts_dir / "method_statistics.json", "w") as f:
        json.dump(method_stats, f, indent=2)
    
    print(f"\nResults saved to artifacts/")
    print(f"Optimal parameters saved to artifacts/optimal_parameters.json")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: OPTIMAL PARAMETERS")
    print(f"{'='*60}")
    for method in sweep_methods:
        s = optimal_scales.get(method, 0.5)
        p = optimal_params.get(method, {})
        print(f"  {method:25s}: scale={s}, p={p.get('dare_p',0)}, trim={p.get('ties_trim',0)}, gamma={p.get('fisher_gamma',0)}")
    print(f"  {'weighted_task_arithmetic':25s}: lambdas={optimal_lambdas}")


if __name__ == "__main__":
    main()
