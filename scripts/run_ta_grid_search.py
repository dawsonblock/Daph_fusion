#!/usr/bin/env python3
"""Run the TA-1 grid search on real distilgpt2 experts.

Executes:
  1. Load distilgpt2 base + 3 specialist checkpoints (math, planning, coding)
  2. Compute base/expert NLLs on validation data
  3. Run TA-0 (uniform, scale search only)
  4. Run TA-1 (weighted, simplex grid search over (λ₁,λ₂,λ₃,α))
  5. Evaluate best config on held-out test
  6. Save results to artifacts/

Constrained objective:
  max R_mean s.t. R_min >= 0.70 and G_regression <= 0.25
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_metrics import compute_domain_nll, calculate_retention
from daph_exfusion.merge.task_search import (
    search_ta0,
    search_ta1,
    SearchResult,
)
from daph_exfusion.merge.types import MergeConfig, MergeMethod
from daph_exfusion.merge.task_arithmetic import merge_task_arithmetic


def load_jsonl(path: Path) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                texts.append(rec.get("text", ""))
    return texts


def make_evaluator(
    tokenizer,
    val_data: Dict[str, List[str]],
    base_nlls: Dict[str, float],
    expert_nlls: Dict[str, float],
    device: str,
    n_samples: int = 50,
):
    """Create an evaluator callable for the grid search.

    Returns a function(merged_model) -> dict with:
        mean_retention, min_retention, general_regression, per_domain_retention
    """
    def evaluator(merged_model: nn.Module) -> dict:
        merged_model.to(device)
        merged_model.eval()

        per_domain_retention = {}
        per_domain_nll = {}
        general_regression = 0.0

        for domain in ["math", "planning", "coding"]:
            nll, _ = compute_domain_nll(
                merged_model, tokenizer,
                val_data[domain][:n_samples],
                device=device,
            )
            per_domain_nll[domain] = nll

            ret = calculate_retention(
                base_nlls[domain], expert_nlls[domain], nll
            )
            per_domain_retention[domain] = ret.value if ret.valid else 0.0

            # General regression: how much worse than base
            if nll > base_nlls[domain]:
                general_regression += (nll - base_nlls[domain]) / base_nlls[domain]

        general_regression /= 3

        mean_ret = np.mean(list(per_domain_retention.values()))
        min_ret = np.min(list(per_domain_retention.values()))

        merged_model.to("cpu")

        return {
            "mean_retention": float(mean_ret),
            "min_retention": float(min_ret),
            "general_regression": float(general_regression),
            "per_domain_retention": {k: float(v) for k, v in per_domain_retention.items()},
            "per_domain_nll": per_domain_nll,
        }

    return evaluator


def evaluate_on_test(
    merged_model: nn.Module,
    tokenizer,
    test_data: Dict[str, List[str]],
    base_nlls: Dict[str, float],
    expert_nlls: Dict[str, float],
    device: str,
    n_samples: int = 50,
) -> dict:
    """Evaluate a merged model on held-out test data."""
    merged_model.to(device)
    merged_model.eval()

    per_domain_retention = {}
    per_domain_nll = {}
    general_regression = 0.0

    for domain in ["math", "planning", "coding"]:
        nll, _ = compute_domain_nll(
            merged_model, tokenizer,
            test_data[domain][:n_samples],
            device=device,
        )
        per_domain_nll[domain] = nll

        ret = calculate_retention(
            base_nlls[domain], expert_nlls[domain], nll
        )
        per_domain_retention[domain] = ret.value if ret.valid else 0.0

        if nll > base_nlls[domain]:
            general_regression += (nll - base_nlls[domain]) / base_nlls[domain]

    general_regression /= 3

    merged_model.to("cpu")

    mean_ret = np.mean(list(per_domain_retention.values()))
    min_ret = np.min(list(per_domain_retention.values()))

    return {
        "per_domain_nll": per_domain_nll,
        "per_domain_retention": {k: float(v) for k, v in per_domain_retention.items()},
        "mean_retention": float(mean_ret),
        "min_retention": float(min_ret),
        "general_regression": float(general_regression),
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_model_id = "distilgpt2"
    domains = ["math", "planning", "coding"]
    data_dir = Path("data")
    checkpoint_dir = Path("checkpoints")
    n_samples = 50

    print(f"Device: {device}")
    print(f"Base model: {base_model_id}")
    print(f"Domains: {domains}")
    print(f"Samples per domain: {n_samples}")
    print()

    # Load tokenizer and base model
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)

    # Load experts
    print("Loading expert checkpoints...")
    experts = []
    for domain in domains:
        expert = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir / domain))
        experts.append(expert)
        print(f"  {domain}: loaded")

    # Load validation and test data
    print("Loading data...")
    val_data = {}
    test_data = {}
    for domain in domains:
        val_data[domain] = load_jsonl(data_dir / domain / "validation.jsonl")
        test_data[domain] = load_jsonl(data_dir / domain / "test.jsonl")
        print(f"  {domain}: val={len(val_data[domain])}, test={len(test_data[domain])}")

    # Compute base and expert NLLs
    print("\nComputing base and expert NLLs on validation...")
    base_model.to(device)
    base_nlls = {}
    expert_nlls = {}
    for i, domain in enumerate(domains):
        base_nlls[domain], _ = compute_domain_nll(
            base_model, tokenizer, val_data[domain][:n_samples], device=device
        )
        experts[i].to(device)
        expert_nlls[domain], _ = compute_domain_nll(
            experts[i], tokenizer, val_data[domain][:n_samples], device=device
        )
        experts[i].to("cpu")
        print(f"  {domain}: base_nll={base_nlls[domain]:.4f}, expert_nll={expert_nlls[domain]:.4f}")

    # Also compute test NLLs for reference
    print("\nComputing base and expert NLLs on test...")
    test_base_nlls = {}
    test_expert_nlls = {}
    for i, domain in enumerate(domains):
        test_base_nlls[domain], _ = compute_domain_nll(
            base_model, tokenizer, test_data[domain][:n_samples], device=device
        )
        experts[i].to(device)
        test_expert_nlls[domain], _ = compute_domain_nll(
            experts[i], tokenizer, test_data[domain][:n_samples], device=device
        )
        experts[i].to("cpu")
        print(f"  {domain}: base_nll={test_base_nlls[domain]:.4f}, expert_nll={test_expert_nlls[domain]:.4f}")

    base_model.to("cpu")

    # Create evaluator
    evaluator = make_evaluator(
        tokenizer, val_data, base_nlls, expert_nlls, device, n_samples
    )

    # =========================================================================
    # TA-0: Uniform baseline (scale search only)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TA-0: Uniform Task Arithmetic (scale search)")
    print("=" * 60)

    t0 = time.time()
    ta0_result = search_ta0(
        base_model, experts, evaluator,
        scales=[0.25, 0.5, 0.75, 1.0, 1.25],
        tau=0.70, delta=0.25,
        device="cpu",
    )
    t0_elapsed = time.time() - t0

    print(f"  Configurations: {ta0_result.n_configurations}")
    print(f"  Feasible: {ta0_result.n_feasible}")
    if ta0_result.best:
        print(f"  Best: α={ta0_result.best.scale}, "
              f"R_mean={ta0_result.best.mean_retention:.4f}, "
              f"R_min={ta0_result.best.min_retention:.4f}, "
              f"G_reg={ta0_result.best.general_regression:.4f}")
        print(f"  Per-domain: {ta0_result.best.per_domain_retention}")
    print(f"  Time: {t0_elapsed:.1f}s")

    # =========================================================================
    # TA-1: Weighted Task Arithmetic (simplex grid search)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TA-1: Weighted Task Arithmetic (simplex grid search)")
    print("=" * 60)

    t1 = time.time()
    ta1_result = search_ta1(
        base_model, experts, evaluator,
        resolution=0.1,
        scales=[0.25, 0.5, 0.75, 1.0, 1.25],
        tau=0.70, delta=0.25,
        device="cpu",
        refine_around_best=True,
        refine_resolution=0.05,
        refine_radius=0.15,
    )
    t1_elapsed = time.time() - t1

    print(f"  Configurations: {ta1_result.n_configurations}")
    print(f"  Feasible: {ta1_result.n_feasible}")
    if ta1_result.best:
        print(f"  Best: λ={ta1_result.best.lambdas}, α={ta1_result.best.scale}")
        print(f"  R_mean={ta1_result.best.mean_retention:.4f}")
        print(f"  R_min={ta1_result.best.min_retention:.4f}")
        print(f"  G_reg={ta1_result.best.general_regression:.4f}")
        print(f"  Per-domain: {ta1_result.best.per_domain_retention}")
    print(f"  Time: {t1_elapsed:.1f}s")

    # =========================================================================
    # Evaluate best configs on held-out test
    # =========================================================================
    print("\n" + "=" * 60)
    print("Held-out test evaluation")
    print("=" * 60)

    test_results = {}

    # TA-0 best on test
    if ta0_result.best:
        config = MergeConfig(
            method=MergeMethod.TASK_ARITHMETIC,
            task_scale=ta0_result.best.scale,
        )
        merged = merge_task_arithmetic(base_model, experts, config)
        ta0_test = evaluate_on_test(
            merged, tokenizer, test_data,
            test_base_nlls, test_expert_nlls, device, n_samples
        )
        test_results["TA-0"] = ta0_test
        print(f"\nTA-0 (α={ta0_result.best.scale}):")
        print(f"  Test R_mean={ta0_test['mean_retention']:.4f}, "
              f"R_min={ta0_test['min_retention']:.4f}, "
              f"G_reg={ta0_test['general_regression']:.4f}")
        print(f"  Per-domain: {ta0_test['per_domain_retention']}")
        del merged

    # TA-1 best on test
    if ta1_result.best:
        config = MergeConfig(
            method=MergeMethod.TASK_ARITHMETIC,
            task_scale=ta1_result.best.scale,
            lambdas=ta1_result.best.lambdas,
        )
        merged = merge_task_arithmetic(base_model, experts, config)
        ta1_test = evaluate_on_test(
            merged, tokenizer, test_data,
            test_base_nlls, test_expert_nlls, device, n_samples
        )
        test_results["TA-1"] = ta1_test
        print(f"\nTA-1 (λ={ta1_result.best.lambdas}, α={ta1_result.best.scale}):")
        print(f"  Test R_mean={ta1_test['mean_retention']:.4f}, "
              f"R_min={ta1_test['min_retention']:.4f}, "
              f"G_reg={ta1_test['general_regression']:.4f}")
        print(f"  Per-domain: {ta1_test['per_domain_retention']}")
        del merged

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    output = {
        "experiment": "ta_grid_search",
        "base_model": base_model_id,
        "domains": domains,
        "n_samples": n_samples,
        "constraints": {"tau": 0.70, "delta": 0.25},
        "validation_base_nlls": base_nlls,
        "validation_expert_nlls": expert_nlls,
        "test_base_nlls": test_base_nlls,
        "test_expert_nlls": test_expert_nlls,
        "ta0": {
            "best": {
                "lambdas": list(ta0_result.best.lambdas) if ta0_result.best else None,
                "scale": ta0_result.best.scale if ta0_result.best else None,
                "mean_retention": ta0_result.best.mean_retention if ta0_result.best else None,
                "min_retention": ta0_result.best.min_retention if ta0_result.best else None,
                "general_regression": ta0_result.best.general_regression if ta0_result.best else None,
                "per_domain_retention": ta0_result.best.per_domain_retention if ta0_result.best else None,
            },
            "n_configurations": ta0_result.n_configurations,
            "n_feasible": ta0_result.n_feasible,
            "time_s": t0_elapsed,
        },
        "ta1": {
            "best": {
                "lambdas": list(ta1_result.best.lambdas) if ta1_result.best else None,
                "scale": ta1_result.best.scale if ta1_result.best else None,
                "mean_retention": ta1_result.best.mean_retention if ta1_result.best else None,
                "min_retention": ta1_result.best.min_retention if ta1_result.best else None,
                "general_regression": ta1_result.best.general_regression if ta1_result.best else None,
                "per_domain_retention": ta1_result.best.per_domain_retention if ta1_result.best else None,
            },
            "n_configurations": ta1_result.n_configurations,
            "n_feasible": ta1_result.n_feasible,
            "time_s": t1_elapsed,
        },
        "test_results": test_results,
    }

    output_path = artifacts_dir / "ta_grid_search_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Mode':10s} {'Val R_mean':>12s} {'Val R_min':>12s} {'Val G_reg':>12s} {'Test R_mean':>12s} {'Test R_min':>12s} {'Test G_reg':>12s}")
    print("-" * 82)

    if ta0_result.best:
        ta0_test = test_results.get("TA-0", {})
        print(f"{'TA-0':10s} {ta0_result.best.mean_retention:12.4f} {ta0_result.best.min_retention:12.4f} {ta0_result.best.general_regression:12.4f} "
              f"{ta0_test.get('mean_retention', 0):12.4f} {ta0_test.get('min_retention', 0):12.4f} {ta0_test.get('general_regression', 0):12.4f}")

    if ta1_result.best:
        ta1_test = test_results.get("TA-1", {})
        print(f"{'TA-1':10s} {ta1_result.best.mean_retention:12.4f} {ta1_result.best.min_retention:12.4f} {ta1_result.best.general_regression:12.4f} "
              f"{ta1_test.get('mean_retention', 0):12.4f} {ta1_test.get('min_retention', 0):12.4f} {ta1_test.get('general_regression', 0):12.4f}")


if __name__ == "__main__":
    main()
