#!/usr/bin/env python
"""Freeze config and evaluate the best method on the held-out test split.

Phase 20: The test split is evaluated ONCE after config freeze.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daph_exfusion.validation.holdout import freeze_config, verify_config_frozen
from research_metrics import compute_domain_nll, calculate_retention
from scripts.run_final_experiment import (
    load_jsonl,
    extract_task_vectors,
)
from scripts.run_enhanced_experiment import build_merged


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_model_id = "distilgpt2"
    domains = ["math", "planning", "coding"]
    data_dir = Path("data")
    checkpoint_dir = Path("checkpoints")

    # Load statistics to find the best method
    with open("artifacts/method_statistics.json") as f:
        stats = json.load(f)

    # Find best merge method by mean retention (excluding base and expert_specialists)
    merge_methods = {k: v for k, v in stats.items() if k not in ("base", "expert_specialists")}
    best_method = max(merge_methods, key=lambda k: np.mean(list(merge_methods[k]["mean_retention"].values())))
    print(f"Best merge method: {best_method}")
    print(f"  Mean retention: {np.mean(list(merge_methods[best_method]['mean_retention'].values())):.4f}")
    print(f"  Worst retention: {merge_methods[best_method]['worst_domain_retention_mean']:.4f}")

    # Load optimized parameters found during the validation-phase search so the
    # held-out test evaluation uses the exact frozen configuration (scale +
    # hyperparameters + lambdas) rather than hardcoded defaults.
    with open("artifacts/optimal_parameters.json") as f:
        opt = json.load(f)
    opt_scales = opt.get("optimal_scales", {})
    opt_params = opt.get("optimal_params", {})
    opt_lambdas = opt.get("optimal_lambdas", [1.0] * len(domains))
    best_scale = opt_scales.get(best_method, 0.5)
    best_hp = opt_params.get(best_method, {})
    dare_p = best_hp.get("dare_p", 0.1)
    ties_trim = best_hp.get("ties_trim", 0.2)
    fisher_gamma = best_hp.get("fisher_gamma", 0.5)
    print(f"  Using optimized params: scale={best_scale}, "
          f"dare_p={dare_p}, ties_trim={ties_trim}, fisher_gamma={fisher_gamma}, "
          f"lambdas={opt_lambdas}")

    # Freeze the config
    config = {
        "base_model": base_model_id,
        "best_method": best_method,
        "seeds": [11, 23, 37, 51, 73],
        "domains": domains,
        "merge_scale": best_scale,
        "merge_hyperparameters": {
            "dare_p": dare_p,
            "ties_trim": ties_trim,
            "fisher_gamma": fisher_gamma,
        },
        "optimal_lambdas": opt_lambdas,
        "n_samples_validation": 50,
    }
    frozen = freeze_config(config, release="v2.4.0-correctness")
    print(f"\nConfig frozen: hash={frozen.config_hash}")
    verify_config_frozen(frozen)
    print("Config verified: no tampering, test split not used during search")

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.to(device)

    experts = []
    for domain in domains:
        expert = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir / domain))
        experts.append(expert)

    # Extract task vectors
    for e in experts:
        e.to("cpu")
    base_model.to("cpu")
    task_vectors = extract_task_vectors(experts, base_model)
    base_model.to(device)

    # Load test data
    test_data = {}
    for domain in domains:
        test_data[domain] = load_jsonl(data_dir / domain / "test.jsonl")

    # Compute base and expert NLLs on test
    print("\n=== HELD-OUT TEST EVALUATION ===")
    base_nlls = {}
    expert_nlls = {}
    for i, domain in enumerate(domains):
        base_nlls[domain], _ = compute_domain_nll(base_model, tokenizer, test_data[domain][:50], device=device)
        expert_nlls[domain], _ = compute_domain_nll(experts[i], tokenizer, test_data[domain][:50], device=device)
        print(f"  {domain}: base_nll={base_nlls[domain]:.4f}, expert_nll={expert_nlls[domain]:.4f}")

    # Evaluate best method on test using the SAME merge construction and
    # optimized parameters as the validation-phase experiment.
    print(f"\nEvaluating {best_method} on test split...")
    merged = build_merged(
        base_model_id, task_vectors, best_method,
        scale=best_scale, dare_p=dare_p, ties_trim=ties_trim,
        fisher_gamma=fisher_gamma, seed=42, lambdas=opt_lambdas,
    )

    merged.to(device)
    merged.eval()

    test_nlls = {}
    test_retention = {}
    for domain in domains:
        nll, _ = compute_domain_nll(merged, tokenizer, test_data[domain][:50], device=device)
        test_nlls[domain] = nll
        ret = calculate_retention(base_nlls[domain], expert_nlls[domain], nll)
        test_retention[domain] = ret.value if ret.valid else None
        print(f"  {domain}: merged_nll={nll:.4f}, retention={test_retention[domain]:.4f}")

    mean_ret = np.mean([v for v in test_retention.values() if v is not None])
    worst_ret = min(v for v in test_retention.values() if v is not None)
    print(f"\nTest mean retention: {mean_ret:.4f}")
    print(f"Test worst retention: {worst_ret:.4f}")

    # Save test results
    test_results = {
        "frozen_config": frozen.to_dict(),
        "best_method": best_method,
        "test_base_nlls": base_nlls,
        "test_expert_nlls": expert_nlls,
        "test_merged_nlls": test_nlls,
        "test_retention": test_retention,
        "test_mean_retention": float(mean_ret),
        "test_worst_retention": float(worst_ret),
    }

    with open("artifacts/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to artifacts/test_results.json")


if __name__ == "__main__":
    main()
