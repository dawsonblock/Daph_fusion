#!/usr/bin/env python3
"""
CLI tool for validating expert qualifications before running ExFusion experiments.

Loads the base model and every candidate expert checkpoint, runs the Phase 1
qualification pipeline (relative improvement I_i >= 0.05 on the isolated
qualification split), and writes a machine-readable report.

Exit behavior:
    default        report-only: writes report, exits 0 with per-expert statuses
    --fail-closed  raises/exits non-zero if any expert fails qualification
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.qualification import ExpertQualificationPipeline, InvalidExperiment

DEFAULT_BASE_MODEL = "distilbert/distilgpt2"
DEFAULT_EXPERTS = [
    {"name": "postbot/distilgpt2-emailgen", "revision": "main", "domain": "math"},
    {
        "name": "FredZhang7/distilgpt2-stable-diffusion",
        "revision": "main",
        "domain": "planning",
    },
    {
        "name": "misterkilgore/distilgpt2-psy-ita",
        "revision": "main",
        "domain": "coding",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate source expert qualification."
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/runs/qualification_report.json"
    )
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="Relative improvement threshold I_i.",
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help="Exit non-zero when any expert fails qualification.",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from run_experiments import build_datasets

    print("[+] Expert qualification preflight check running...")
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)

    qualification_data, _, _ = build_datasets()
    pipeline = ExpertQualificationPipeline(
        base_model,
        tokenizer,
        device=device,
        min_expert_improvement=args.min_improvement,
    )

    qualifications = []
    for meta in DEFAULT_EXPERTS:
        expert_model = AutoModelForCausalLM.from_pretrained(meta["name"]).to(device)
        q = pipeline.qualify_expert(
            expert_name=meta["name"],
            expert_revision=meta["revision"],
            expert_model=expert_model,
            domain=meta["domain"],
            qualification_texts=qualification_data[meta["domain"]],
        )
        qualifications.append(q)
        print(
            f"[Qualification] {q.expert_name} ({q.domain}): "
            f"Rel Gain = {q.relative_improvement:.4f} | Passed = {q.passed}"
        )
        del expert_model

    all_passed = all(q.passed for q in qualifications)
    report = {
        "status": "PASS" if all_passed else "FAIL",
        "min_expert_improvement": args.min_improvement,
        "base_model": args.base_model,
        "message": (
            "All source experts validated."
            if all_passed
            else "One or more source experts failed the qualification gate."
        ),
        "experts": [dataclasses.asdict(q) for q in qualifications],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[✓] Expert qualification report written to {args.output}")

    if args.fail_closed:
        pipeline.validate_preflight(qualifications)
    elif not all_passed:
        print(
            "[!] WARNING: unqualified experts detected (report-only mode); "
            "rerun with --fail-closed to enforce the preflight gate."
        )


if __name__ == "__main__":
    main()
