#!/usr/bin/env python
"""Train same-lineage specialist experts from a single base checkpoint.

Phase 4.2 of the DAPH ExFusion repair plan. All specialists MUST originate
from the exact same base checkpoint and tokenizer so that task-vector merges
are on-lineage. The previous run used off-the-shelf distilgpt2 fine-tunes
(postbot/distilgpt2-emailgen, FredZhang7/distilgpt2-stable-diffusion,
misterkilgore/distilgpt2-psy-ita) which failed qualification.

Usage:
    python scripts/train_lineage_experts.py \
        --base-model distilgpt2 \
        --output-dir checkpoints \
        --domains math planning coding \
        --train-data data/train/ \
        --steps 500 --lr 5e-5 --seed 23

Each produced checkpoint directory contains:
    pytorch_model.bin   (or safetensors)
    config.json
    tokenizer files
    lineage_manifest.json   (base_model, base_revision, tokenizer_hash,
                             training_data_hash, seed, steps, lr)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonicalize_text(text: str) -> str:
    """Unicode + whitespace + newline normalization before hashing."""
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of whitespace except newlines
    lines = []
    for line in text.split("\n"):
        lines.append(" ".join(line.split()))
    return "\n".join(lines).strip()


def _hash_training_data(data_dir: Path, domain: str) -> str:
    """SHA-256 of canonicalized training text for a domain."""
    domain_dir = data_dir / domain
    if not domain_dir.exists():
        raise FileNotFoundError(f"Training data dir not found: {domain_dir}")
    h = hashlib.sha256()
    for f in sorted(domain_dir.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        canonical = _canonicalize_text(text)
        h.update(canonical.encode("utf-8"))
        h.update(b"\n---\n")
    return h.hexdigest()


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_domain_texts(data_dir: Path, domain: str) -> List[str]:
    domain_dir = data_dir / domain
    if not domain_dir.exists():
        raise FileNotFoundError(f"Training data dir not found: {domain_dir}")
    texts = []
    for f in sorted(domain_dir.glob("*.txt")):
        texts.append(f.read_text(encoding="utf-8"))
    if not texts:
        raise ValueError(f"No .txt training files found in {domain_dir}")
    return texts


def train_specialist(
    base_model_id: str,
    domain: str,
    train_texts: List[str],
    output_dir: Path,
    tokenizer,
    steps: int,
    lr: float,
    seed: int,
    max_length: int = 128,
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict:
    """Fine-tune a specialist from the base checkpoint."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    model.to(device)
    model.train()

    dataset = TextDataset(train_texts, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    step = 0
    total_loss = 0.0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            step += 1
            if step % 50 == 0:
                avg = total_loss / step
                print(f"  [{domain}] step {step}/{steps} | avg_loss={avg:.4f}")

    # Save
    out = output_dir / domain
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Manifest
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_model_id)
    base_revision = getattr(config, "_commit_hash", "unknown")

    manifest = {
        "base_model": base_model_id,
        "base_revision": base_revision,
        "domain": domain,
        "seed": seed,
        "steps": steps,
        "learning_rate": lr,
        "max_length": max_length,
        "batch_size": batch_size,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "final_avg_loss": total_loss / max(steps, 1),
    }
    with open(out / "lineage_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train same-lineage specialist experts")
    parser.add_argument("--base-model", default="distilgpt2", help="HuggingFace base model ID")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory for experts")
    parser.add_argument("--domains", nargs="+", default=["math", "planning", "coding"])
    parser.add_argument("--train-data", default="data/train", help="Root dir with per-domain subdirs")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.train_data)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_manifests = []
    for domain in args.domains:
        print(f"\n{'='*60}\nTraining specialist for domain: {domain}\n{'='*60}")
        texts = load_domain_texts(data_dir, domain)
        data_hash = _hash_training_data(data_dir, domain)
        manifest = train_specialist(
            base_model_id=args.base_model,
            domain=domain,
            train_texts=texts,
            output_dir=output_dir,
            tokenizer=tokenizer,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
        )
        manifest["training_data_hash"] = data_hash
        manifest["tokenizer_hash"] = _hash_str(str(tokenizer))
        # Re-write manifest with hashes
        with open(output_dir / domain / "lineage_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        all_manifests.append(manifest)
        print(f"  Saved to {output_dir / domain}")

    # Write global lineage index
    with open(output_dir / "lineage_index.json", "w") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "seed": args.seed,
                "experts": all_manifests,
            },
            f,
            indent=2,
        )
    print(f"\nLineage index written to {output_dir / 'lineage_index.json'}")


if __name__ == "__main__":
    main()
