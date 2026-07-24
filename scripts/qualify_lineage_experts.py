#!/usr/bin/env python
"""Qualify the trained lineage experts against the base model.

Loads distilgpt2 (base) and the trained specialists, runs the qualification
gate on the qualification split, and reports results. Fail-closed: raises
QualificationError if any expert fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.qualification import (
    ExpertQualificationPipeline,
    QualificationError,
)
from research_metrics import compute_domain_nll


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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_model_id = "distilgpt2"
    domains = ["math", "planning", "coding"]
    data_dir = Path("data")
    checkpoint_dir = Path("checkpoints")

    print(f"Device: {device}")
    print(f"Base model: {base_model_id}")
    print(f"Loading base model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.to(device)

    # Load qualification data
    qual_data = {}
    for domain in domains:
        qual_data[domain] = load_jsonl(data_dir / domain / "qualification.jsonl")
        print(f"  {domain} qualification: {len(qual_data[domain])} texts")

    # Load experts
    experts = []
    expert_meta = []
    for domain in domains:
        expert_path = checkpoint_dir / domain
        if not expert_path.exists():
            print(f"ERROR: expert checkpoint not found at {expert_path}")
            sys.exit(1)
        expert = AutoModelForCausalLM.from_pretrained(str(expert_path))
        expert.to(device)
        experts.append(expert)
        expert_meta.append({"name": f"lineage-{domain}", "revision": "local", "domain": domain})

    # Run qualification
    print("\n=== QUALIFICATION GATE (I_i >= 0.05, FAIL-CLOSED) ===")
    pipeline = ExpertQualificationPipeline(
        base_model, tokenizer, device=device, min_expert_improvement=0.05
    )

    qualifications = []
    all_passed = True
    for meta, expert in zip(expert_meta, experts):
        q = pipeline.qualify_expert(
            expert_name=meta["name"],
            expert_revision=meta["revision"],
            expert_model=expert,
            domain=meta["domain"],
            qualification_texts=qual_data[meta["domain"]],
        )
        qualifications.append(q)
        status = "PASS" if q.passed else "FAIL"
        print(f"  [{status}] {q.expert_name} ({q.domain}): "
              f"base_nll={q.base_nll:.4f}, expert_nll={q.expert_nll:.4f}, "
              f"rel_improvement={q.relative_improvement:.4f}")
        if q.rejection_reason:
            print(f"         Reason: {q.rejection_reason}")
        if not q.passed:
            all_passed = False

    # Also check cross-domain degradation
    print("\n=== CROSS-DOMAIN DEGRADATION CHECK ===")
    for i, (meta, expert) in enumerate(zip(expert_meta, experts)):
        target_domain = meta["domain"]
        for j, other_domain in enumerate(domains):
            if other_domain == target_domain:
                continue
            eval_texts = load_jsonl(data_dir / other_domain / "qualification.jsonl")
            base_nll, _ = compute_domain_nll(base_model, tokenizer, eval_texts[:50], device=device)
            expert_nll, _ = compute_domain_nll(expert, tokenizer, eval_texts[:50], device=device)
            degradation = (expert_nll - base_nll) / base_nll if base_nll > 0 else 0
            status = "OK" if degradation <= 0.20 else "WARN"
            print(f"  [{status}] {meta['name']} on {other_domain}: "
                  f"degradation={degradation:.4f} (threshold: 0.20)")

    # Save qualification report
    report = {
        "status": "PASS" if all_passed else "FAIL",
        "min_expert_improvement": 0.05,
        "base_model": base_model_id,
        "experts": [
            {
                "expert_name": q.expert_name,
                "domain": q.domain,
                "base_nll": q.base_nll,
                "expert_nll": q.expert_nll,
                "relative_improvement": q.relative_improvement,
                "passed": q.passed,
                "rejection_reason": q.rejection_reason,
            }
            for q in qualifications
        ],
    }
    report_path = Path("artifacts/qualification_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nQualification report saved to {report_path}")

    if not all_passed:
        print("\nFAILED: One or more experts did not pass qualification.")
        sys.exit(1)
    else:
        print("\nALL EXPERTS QUALIFIED.")


if __name__ == "__main__":
    main()
