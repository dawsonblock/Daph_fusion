#!/usr/bin/env python3
"""CLI entry point for AGX-S (Layerwise Geometry Search).

Executes the real LayerwiseGeometrySearchEngine — not a hardcoded result.
Requires trained experts and a validation batch.

Usage:
    python search_geometry.py --base-model distilgpt2 \
        --checkpoint-dir checkpoints --data-dir data \
        --num-candidates 16 --output artifacts/agx_search_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daph_exfusion.search.optimization import LayerwiseGeometrySearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="AGX-S Layerwise Geometry Search")
    parser.add_argument("--base-model", default="distilgpt2")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--domains", nargs="+", default=["math", "planning", "coding"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-cka-drift", type=float, default=0.15)
    parser.add_argument("--output", default="artifacts/agx_search_results.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"[+] Loading base model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)

    print(f"[+] Loading {len(args.domains)} experts...")
    experts = []
    for domain in args.domains:
        expert = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir / domain)).to(device)
        experts.append(expert)

    # Build validation batch from the specified split
    print(f"[+] Building validation batch from {args.split} split...")
    texts = []
    for domain in args.domains:
        split_path = data_dir / domain / f"{args.split}.jsonl"
        if split_path.exists():
            with open(split_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        texts.append(rec.get("text", ""))
    # Use a small batch for CKA measurement
    batch_texts = texts[:4]
    enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    val_batch = {k: v.to(device) for k, v in enc.items()}

    print(f"[+] Running AGX-S search ({args.num_candidates} candidates)...")
    engine = LayerwiseGeometrySearchEngine(
        base_model=base_model,
        experts=experts,
        validation_batch=val_batch,
        max_cka_drift=args.max_cka_drift,
    )

    best_candidate, history = engine.search(
        num_candidates=args.num_candidates,
        seed=args.seed,
    )

    # Build results
    results = {
        "algorithm": "AGX-S",
        "mode": "layerwise",
        "split": args.split,
        "num_candidates": args.num_candidates,
        "seed": args.seed,
        "max_cka_drift": args.max_cka_drift,
        "selected_candidate": None,
        "pareto_front_size": sum(1 for h in history if h.feasible),
        "search_trace": [
            {
                "candidate_hash": h.candidate_hash,
                "validation_nll": h.validation_nll,
                "feasible": h.feasible,
                "max_layer_drift": h.max_layer_drift,
            }
            for h in history
        ],
    }
    if best_candidate is not None:
        results["selected_candidate"] = {
            "layer_configs": {
                str(k): {
                    "operator": v.operator,
                    "lambdas": list(v.lambdas),
                    "ties_trim": v.ties_trim,
                    "dare_drop": v.dare_drop,
                    "fisher_gamma": v.fisher_gamma,
                }
                for k, v in sorted(best_candidate.layer_configs.items())
            },
            "hash": best_candidate.compute_hash(),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] AGX-S search complete. {results['pareto_front_size']}/{args.num_candidates} feasible. "
          f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
