"""
4-Layer Isolated Dataset Split Management (Phase 4).
Structures corpora across qualification, calibration, validation, and test splits.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from diagnostics.dataset_audit import audit_dataset_splits, compute_text_hash


def build_4layer_dataset_splits() -> Dict[str, Dict[str, List[str]]]:
    """Generates 4-layer isolated corpora per domain."""
    domains = ["email", "art", "psychology", "generic"]
    layers = ["qualification", "calibration", "validation", "test"]

    dataset: Dict[str, Dict[str, List[str]]] = {d: {} for d in domains}

    sizes = {
        "qualification": 250,
        "calibration": 500,
        "validation": 500,
        "test": 1000,
    }

    # Template generators
    for domain in domains:
        offset = 0
        for layer in layers:
            count = sizes[layer]
            samples = []
            for i in range(offset + 1, offset + count + 1):
                if domain == "email":
                    samples.append(
                        f"Dear Team, this is official email communication sample {i} regarding quarterly goals."
                    )
                elif domain == "art":
                    samples.append(
                        f"A highly detailed digital painting of a fantasy landscape, 8k render, masterpiece quality sample {i}."
                    )
                elif domain == "psychology":
                    samples.append(
                        f"In psychological terms, cognitive behavior and emotional resilience sample {i} defines mental health."
                    )
                else:
                    samples.append(
                        f"General knowledge and standard Wikipedia encyclopedia text statement sample {i} for generic domain."
                    )
            dataset[domain][layer] = samples
            offset += count

    return dataset


def save_dataset_splits(base_dir: str = "data") -> None:
    splits = build_4layer_dataset_splits()
    for domain, layers in splits.items():
        domain_dir = os.path.join(base_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        for layer, samples in layers.items():
            path = os.path.join(domain_dir, f"{layer}.jsonl")
            with open(path, "w") as f:
                for s in samples:
                    f.write(
                        json.dumps({"text": s, "hash": compute_text_hash(s)}) + "\n"
                    )
    print(f"[✓] 4-layer dataset splits saved under {base_dir}/")


if __name__ == "__main__":
    save_dataset_splits()
