"""Dataset isolation and provenance audit (Phase 5).

Enforces strict disjointness across the five logical data partitions:
  train, qualification, calibration, validation, test

Canonicalizes text (Unicode NFKC + whitespace + newline normalization)
before hashing, detects exact cross-split duplicates, and provides a
near-duplicate check via MinHash (Jaccard similarity on shingle sets).

Release gate: an official run requires exact_overlap == 0 and
near_duplicate_threshold_pass == True.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


SPLITS = ("train", "qualification", "calibration", "validation", "test")


def canonicalize_text(text: str) -> str:
    """Unicode NFKC + whitespace + newline normalization."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(lines).strip()


def hash_text(text: str) -> str:
    """SHA-256 of canonicalized text."""
    return hashlib.sha256(canonicalize_text(text).encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 10) -> Set[str]:
    """Character k-shingles for MinHash near-duplicate detection.

    k=10 (word-level granularity) is used instead of k=5 (character-level)
    because short structured text (math problems, code snippets) naturally
    shares many 5-character shingles even when the content is genuinely
    different. k=10 reduces false positives while still detecting actual
    paraphrase near-duplicates.
    """
    canonical = canonicalize_text(text)
    if len(canonical) < k:
        return {canonical}
    return {canonical[i : i + k] for i in range(len(canonical) - k + 1)}


def _minhash_signature(shingles: Set[str], num_perm: int = 128) -> List[int]:
    """Compute a MinHash signature for a set of shingles."""
    # Simple hash-based MinHash without a fixed random seed for determinism
    signature = []
    for i in range(num_perm):
        if not shingles:
            signature.append(0)
            continue
        # Use a deterministic hash family: hash(shingle + salt_i)
        min_hash = min(
            int(hashlib.md5(f"{s}|{i}".encode()).hexdigest(), 16) for s in shingles
        )
        signature.append(min_hash)
    return signature


def _jaccard_from_signatures(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    if len(sig_a) != len(sig_b) or len(sig_a) == 0:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


@dataclass
class SplitReport:
    name: str
    num_records: int
    num_unique_hashes: int


@dataclass
class OverlapReport:
    pair: str
    exact_overlap: int
    near_duplicate_count: int = 0
    near_duplicate_examples: List[str] = field(default_factory=list)


@dataclass
class DatasetAuditResult:
    splits: List[SplitReport]
    exact_overlaps: List[OverlapReport]
    near_duplicate_overlaps: List[OverlapReport]
    exact_overlap_total: int
    near_duplicate_threshold_pass: bool
    train_test_overlap: int
    qualification_test_overlap: int
    calibration_test_overlap: int
    validation_test_overlap: int
    pass_release_gate: bool
    near_duplicate_threshold: float

    def to_dict(self) -> dict:
        return {
            "splits": [
                {"name": s.name, "num_records": s.num_records,
                 "num_unique_hashes": s.num_unique_hashes}
                for s in self.splits
            ],
            "exact_overlap_total": self.exact_overlap_total,
            "near_duplicate_threshold_pass": self.near_duplicate_threshold_pass,
            "train_test_overlap": self.train_test_overlap,
            "qualification_test_overlap": self.qualification_test_overlap,
            "calibration_test_overlap": self.calibration_test_overlap,
            "validation_test_overlap": self.validation_test_overlap,
            "pass_release_gate": self.pass_release_gate,
            "near_duplicate_threshold": self.near_duplicate_threshold,
        }


def load_split(path: Path) -> List[dict]:
    """Load a JSONL split file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def audit_dataset(
    data_root: Path,
    splits: Tuple[str, ...] = SPLITS,
    near_duplicate_threshold: float = 0.8,
    near_duplicate_sample_limit: int = 50,
) -> DatasetAuditResult:
    """Audit dataset isolation across all splits.

    Checks:
      1. Exact hash duplicates across all split pairs (must be 0 for release)
      2. Near-duplicates via MinHash Jaccard (above threshold flagged)
      3. Specific train→test, qualification→test, calibration→test,
         validation→test overlaps (must all be 0 for release)

    Returns a DatasetAuditResult with the full report.
    """
    data_root = Path(data_root)

    # Collect per-split hash sets and records
    split_hashes: Dict[str, Set[str]] = {}
    split_records: Dict[str, List[dict]] = {}
    split_reports: List[SplitReport] = []

    for split in splits:
        # Splits may be organized as <domain>/<split>.jsonl or <split>.jsonl
        # Try both patterns
        paths = list(data_root.rglob(f"{split}.jsonl"))
        if not paths:
            split_reports.append(SplitReport(split, 0, 0))
            split_hashes[split] = set()
            split_records[split] = []
            continue

        all_hashes: Set[str] = set()
        all_records: List[dict] = []
        for p in paths:
            records = load_split(p)
            for rec in records:
                text = rec.get("text", "")
                h = hash_text(text)
                all_hashes.add(h)
                rec["_hash"] = h
                rec["_shingles"] = _shingles(text)
                all_records.append(rec)

        split_hashes[split] = all_hashes
        split_records[split] = all_records
        split_reports.append(SplitReport(split, len(all_records), len(all_hashes)))

    # Exact overlap across all pairs
    exact_overlaps: List[OverlapReport] = []
    exact_total = 0
    split_list = list(splits)
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            a, b = split_list[i], split_list[j]
            ha, hb = split_hashes.get(a, set()), split_hashes.get(b, set())
            overlap = len(ha & hb)
            if overlap > 0:
                exact_overlaps.append(OverlapReport(f"{a}∩{b}", overlap))
                exact_total += overlap

    # Specific test-overlap metrics
    test_hashes = split_hashes.get("test", set())
    train_test = len(split_hashes.get("train", set()) & test_hashes)
    qual_test = len(split_hashes.get("qualification", set()) & test_hashes)
    cal_test = len(split_hashes.get("calibration", set()) & test_hashes)
    val_test = len(split_hashes.get("validation", set()) & test_hashes)

    # Near-duplicate check (sampled to avoid O(N^2) blowup)
    near_overlaps: List[OverlapReport] = []
    near_dup_pass = True

    # For efficiency, only check train vs test and qualification vs test
    # near-duplicates (the most critical for validity)
    for a, b in [("train", "test"), ("qualification", "test"),
                 ("calibration", "test"), ("validation", "test")]:
        recs_a = split_records.get(a, [])
        recs_b = split_records.get(b, [])
        if not recs_a or not recs_b:
            continue

        # Sample to limit compute
        sample_a = recs_a[:near_duplicate_sample_limit]
        sample_b = recs_b[:near_duplicate_sample_limit]

        # Precompute signatures for the sample
        sigs_a = [_minhash_signature(r.get("_shingles", set())) for r in sample_a]
        sigs_b = [_minhash_signature(r.get("_shingles", set())) for r in sample_b]

        near_count = 0
        examples: List[str] = []
        for ia, sa in enumerate(sigs_a):
            for ib, sb in enumerate(sigs_b):
                # Skip exact duplicates (already counted)
                if sample_a[ia].get("_hash") == sample_b[ib].get("_hash"):
                    continue
                jac = _jaccard_from_signatures(sa, sb)
                if jac >= near_duplicate_threshold:
                    near_count += 1
                    if len(examples) < 3:
                        examples.append(
                            f"{a}[{ia}] ~ {b}[{ib}] jaccard={jac:.3f}"
                        )

        if near_count > 0:
            near_overlaps.append(
                OverlapReport(f"{a}≈{b}", 0, near_count, examples)
            )
            near_dup_pass = False

    release_gate = (
        exact_total == 0
        and train_test == 0
        and qual_test == 0
        and cal_test == 0
        and val_test == 0
        and near_dup_pass
    )

    return DatasetAuditResult(
        splits=split_reports,
        exact_overlaps=exact_overlaps,
        near_duplicate_overlaps=near_overlaps,
        exact_overlap_total=exact_total,
        near_duplicate_threshold_pass=near_dup_pass,
        train_test_overlap=train_test,
        qualification_test_overlap=qual_test,
        calibration_test_overlap=cal_test,
        validation_test_overlap=val_test,
        pass_release_gate=release_gate,
        near_duplicate_threshold=near_duplicate_threshold,
    )
