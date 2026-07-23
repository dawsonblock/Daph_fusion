#!/usr/bin/env python3
"""
Lineage-Matched Expert Fine-Tuning Utility (Phase 1 / Phase 3 of ROADMAP_PLAN.md).

Fine-tunes domain-specialist checkpoints directly from the exact base model
checkpoint (default: distilbert/distilgpt2) so every expert shares identical
lineage, parameter topology, and tokenizer with the merge target theta_0.

Training data is read from the isolated 4-layer dataset splits:
    data/{domain}/qualification.jsonl   (never used for training)
    data/{domain}/calibration.jsonl     (default training split)

Usage:
    python3 scripts/train_lineage_experts.py \
        --base-model distilbert/distilgpt2 \
        --domain math \
        --train-jsonl data/math/calibration.jsonl \
        --output-dir artifacts/experts/math

    # Train all domains in one pass:
    python3 scripts/train_lineage_experts.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_BASE_MODEL = "distilbert/distilgpt2"
DEFAULT_DOMAINS = ("math", "planning", "coding")


def load_jsonl_texts(path: str) -> List[str]:
    """Loads newline-delimited JSON records with a 'text' field."""
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text") if isinstance(record, dict) else None
            if text:
                texts.append(str(text))
    if not texts:
        raise ValueError(f"No usable 'text' records found in {path}")
    return texts


def build_synthetic_training_texts(domain: str) -> List[str]:
    """Falls back to the synthetic calibration corpus generators when a JSONL
    split is unavailable (keeps qualification/evaluation splits isolated)."""
    from run_experiments import build_datasets

    _, calibration_data, _ = build_datasets()
    if domain not in calibration_data:
        raise ValueError(
            f"Unknown domain '{domain}'; expected {sorted(calibration_data)}"
        )
    return calibration_data[domain]


def train_lineage_expert(
    base_model_id: str,
    domain: str,
    train_texts: List[str],
    output_dir: str,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 8,
    num_train_epochs: float = 3.0,
    weight_decay: float = 0.01,
    max_length: int = 128,
    seed: int = 17,
) -> Dict[str, Any]:
    """Fine-tunes one lineage-matched specialist expert from the exact base checkpoint."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_id)

    class _TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts: List[str]) -> None:
            self.encodings = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        def __len__(self) -> int:
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            return {
                key: torch.tensor(values[idx]) for key, values in self.encodings.items()
            }

    train_dataset = _TextDataset(train_texts)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Fine-tune with low learning rate to guarantee controlled parameter delta
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    train_result = trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "base_model": base_model_id,
        "domain": domain,
        "num_train_samples": len(train_texts),
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "final_train_loss": float(train_result.training_loss),
        "seed": seed,
    }
    with open(os.path.join(output_dir, "lineage_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[✓] Lineage expert '{domain}' trained from '{base_model_id}' "
        f"(final loss {train_result.training_loss:.4f}) -> {output_dir}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune lineage-matched specialist experts from the exact base checkpoint."
    )
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--domain", type=str, choices=DEFAULT_DOMAINS)
    parser.add_argument(
        "--train-jsonl",
        type=str,
        default=None,
        help="Optional JSONL training split (falls back to synthetic calibration corpus).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--all", action="store_true", help="Train experts for all domains sequentially."
    )
    args = parser.parse_args()

    domains = list(DEFAULT_DOMAINS) if args.all else [args.domain]
    if domains == [None]:
        parser.error("Provide --domain <name> or --all.")

    for domain in domains:
        if args.train_jsonl and not args.all:
            train_texts = load_jsonl_texts(args.train_jsonl)
        else:
            jsonl_path = os.path.join("data", domain, "calibration.jsonl")
            if os.path.exists(jsonl_path):
                train_texts = load_jsonl_texts(jsonl_path)
            else:
                train_texts = build_synthetic_training_texts(domain)
        output_dir = args.output_dir or os.path.join("artifacts", "experts", domain)
        train_lineage_expert(
            base_model_id=args.base_model,
            domain=domain,
            train_texts=train_texts,
            output_dir=output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
        )


if __name__ == "__main__":
    main()
