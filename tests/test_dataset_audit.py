"""Phase 5: Dataset isolation audit tests."""
import json
import tempfile
from pathlib import Path

import pytest

from daph_exfusion.data.dataset_audit import (
    audit_dataset,
    canonicalize_text,
    hash_text,
)


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_canonicalize_normalizes_whitespace():
    assert canonicalize_text("  hello   world  ") == "hello world"
    assert canonicalize_text("a\r\nb") == "a\nb"


def test_hash_is_deterministic():
    assert hash_text("hello") == hash_text("hello")
    assert hash_text("hello") != hash_text("world")


def test_hash_canonicalizes_before_hashing():
    """Whitespace differences should not produce different hashes."""
    assert hash_text("hello  world") == hash_text("hello world")


def test_no_overlap_passes():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_jsonl(root / "math/train.jsonl", [{"text": f"train sample {i}"} for i in range(10)])
        _write_jsonl(root / "math/qualification.jsonl", [{"text": f"qual sample {i}"} for i in range(10)])
        _write_jsonl(root / "math/calibration.jsonl", [{"text": f"cal sample {i}"} for i in range(10)])
        _write_jsonl(root / "math/validation.jsonl", [{"text": f"val sample {i}"} for i in range(10)])
        _write_jsonl(root / "math/test.jsonl", [{"text": f"test sample {i}"} for i in range(10)])
        result = audit_dataset(root)
        assert result.exact_overlap_total == 0
        assert result.pass_release_gate


def test_exact_overlap_detected():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_jsonl(root / "math/train.jsonl", [{"text": "shared text"}, {"text": "unique train"}])
        _write_jsonl(root / "math/test.jsonl", [{"text": "shared text"}, {"text": "unique test"}])
        _write_jsonl(root / "math/qualification.jsonl", [{"text": "qual only"}])
        _write_jsonl(root / "math/calibration.jsonl", [{"text": "cal only"}])
        _write_jsonl(root / "math/validation.jsonl", [{"text": "val only"}])
        result = audit_dataset(root)
        assert result.exact_overlap_total > 0
        assert result.train_test_overlap == 1
        assert not result.pass_release_gate


def test_near_duplicate_detected():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Two texts that are near-duplicates (only the number differs)
        _write_jsonl(root / "math/train.jsonl", [{"text": "The quick brown fox jumps over the lazy dog number 1"}])
        _write_jsonl(root / "math/test.jsonl", [{"text": "The quick brown fox jumps over the lazy dog number 2"}])
        _write_jsonl(root / "math/qualification.jsonl", [{"text": "completely different qualification text here"}])
        _write_jsonl(root / "math/calibration.jsonl", [{"text": "completely different calibration text here"}])
        _write_jsonl(root / "math/validation.jsonl", [{"text": "completely different validation text here"}])
        result = audit_dataset(root, near_duplicate_threshold=0.7)
        assert not result.near_duplicate_threshold_pass
        assert not result.pass_release_gate
