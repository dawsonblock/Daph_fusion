"""
Dataset & Hash Disjointness Diagnostics (Phase 2).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Set, Tuple


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def audit_dataset_splits(
    splits: Dict[str, List[str]],
) -> Dict[str, Any]:
    seen_hashes: Dict[str, str] = {}
    overlaps: List[Dict[str, Any]] = []

    split_counts = {name: len(texts) for name, texts in splits.items()}

    for split_name, texts in splits.items():
        for idx, text in enumerate(texts):
            h = compute_text_hash(text)
            if h in seen_hashes:
                overlaps.append(
                    {
                        "text": text[:50],
                        "hash": h,
                        "first_seen_in": seen_hashes[h],
                        "duplicate_in": split_name,
                        "index": idx,
                    }
                )
            else:
                seen_hashes[h] = split_name

    return {
        "valid_disjoint": len(overlaps) == 0,
        "split_counts": split_counts,
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }
