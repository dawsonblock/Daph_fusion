#!/usr/bin/env python
"""Freeze config and evaluate the best method on the held-out test split (v2.6).

Uses the canonical merge pipeline with real algorithm implementations.
The test split is evaluated ONCE after config freeze.
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
from daph_exfusion.merge.pipeline import MergeConfig, merge_experts
from daph_exfusion.curvature.bank import CurvatureBank
from research_metrics import compute_domain_nll, calculate_retention


def load_jsonl(path: Path) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                texts.append(rec.get("text", ""))
    return texts


def main():
    device = "cpu"
    base_model_id = "distilgpt2"
    domains = ["math", "planning", "coding"]
    data_dir = Path("data")
    checkpoint_dir = Path("checkpoints")

    # Load statistics to find the best method
    with open("artifacts/method_statistics.json") as f:
        stats = json.load(f)

    # Find best merge method by mean retention (excluding base/expert/AGX_H)
    merge_methods = {
        k: v for k, v in stats.items()
        if k not in ("base", "expert_specialists", "AGX_H")
    }
    best_method = max(
        merge_methods,
        key=lambda k: np.mean(list(merge_methods[k]["mean_retention"].values())),
    )
    print(f"Best merge method: {best_method}")
    print(f"  Mean retention: {np.mean(list(merge_methods[best_method]['mean_retention'].values())):.4f}")
    print(f"  Worst retention: {merge_methods[best_method]['worst_domain_retention_mean']:.4f}")
    print(f"  Operator trace: {merge_methods[best_method].get('operator_trace', [])}")

    # Build config from the method statistics
    method_stats = merge_methods[best_method]
    config = MergeConfig(
        algorithm=best_method,
        scale=0.5,  # will be overridden by optimal if available
        dare_drop_rate=0.1,
        ties_trim_fraction=0.2,
        fisher_gamma=0.5,
        lambdas=tuple([1.0] * len(domains)),
        seed=42,
    )

    # Freeze the config
    freeze_config_dict = {
        "base_model": base_model_id,
        "best_method": best_method,
        "seeds": [11, 23, 37, 51, 73],
        "domains": domains,
        "merge_scale": config.scale,
        "merge_hyperparameters": {
            "dare_drop_rate": config.dare_drop_rate,
            "ties_trim_fraction": config.ties_trim_fraction,
            "fisher_gamma": config.fisher_gamma,
            "ties_sign_mode": config.ties_sign_mode,
        },
        "operator_trace": method_stats.get("operator_trace", []),
        "n_samples_validation": 50,
    }
    frozen = freeze_config(freeze_config_dict, release="v2.6.0-experimental-truth")
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
    expert_names = []
    for domain in domains:
        expert = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir / domain))
        experts.append(expert)
        expert_names.append(f"expert_{domain}")

    # Build curvature bank if needed
    curvature_bank = None
    if best_method in ("FISHER", "TIES_FISHER", "DARE_TIES_FISHER", "EXFUSION"):
        print("\nBuilding CurvatureBank for test evaluation...")
        cal_texts = []
        for domain in domains:
            cal_texts.extend(load_jsonl(data_dir / domain / "calibration.jsonl")[:30])
        bank = CurvatureBank.build(
            base_model, experts, cal_texts, tokenizer,
            device=device, max_length=128, max_samples=90,
            expert_names=expert_names,
        )
        curvature_bank = bank.fisher

    # Merge using canonical pipeline
    print(f"\nMerging with {best_method} (trace: {config.algorithm})...")
    result = merge_experts(
        base_model, experts, config,
        curvature_bank=curvature_bank,
        device=device,
    )
    print(f"  operator_trace: {result.operator_trace}")

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

    # Evaluate merged model on test
    merged_nlls = {}
    test_retention = {}
    for domain in domains:
        nll, _ = compute_domain_nll(result.merged_model, tokenizer, test_data[domain][:50], device=device)
        merged_nlls[domain] = nll
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
        "operator_trace": result.operator_trace,
        "test_base_nlls": base_nlls,
        "test_expert_nlls": expert_nlls,
        "test_merged_nlls": merged_nlls,
        "test_retention": test_retention,
        "test_mean_retention": float(mean_ret),
        "test_worst_retention": float(worst_ret),
    }

    with open("artifacts/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to artifacts/test_results.json")


if __name__ == "__main__":
    main()
