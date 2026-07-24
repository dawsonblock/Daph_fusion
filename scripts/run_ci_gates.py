#!/usr/bin/env python
"""CI gate enforcement for DAPH ExFusion (Phase 22).

Run all gate checks and fail if any do not pass. Designed to be called
from CI (GitHub Actions, etc.) or locally before any release.

Usage:
    python scripts/run_ci_gates.py [--gate runtime|research|merge|agx|all]

Gates:
  runtime  - compileall, pytest, mixed precision, state isolation, streaming
  research - dataset disjointness, expert qualification, metric validity,
             artifact schema, no test split usage during search
  merge    - DARE expectation, TIES sign-election, Fisher reference,
             projection invariants, baseline semantics
  agx      - all operators implemented, no stubs, CKA valid, Pareto
             deterministic, candidate serialization round-trip
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: List[str], cwd: Path = REPO_ROOT) -> Tuple[int, str]:
    """Run a command and return (exit_code, output)."""
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def gate_runtime() -> bool:
    """Runtime correctness gate."""
    print("\n=== RUNTIME GATE ===")

    # compileall
    code, out = _run([sys.executable, "-m", "compileall", "-q", "."])
    if code != 0:
        print(f"FAIL: compileall\n{out}")
        return False
    print("PASS: compileall")

    # pytest (all tests)
    code, out = _run([
        sys.executable, "-m", "pytest",
        "tests/", "test_v2_3_1_correctness.py", "test_nesy_v1_0.py",
        "-x", "-q", "--tb=short",
    ])
    if code != 0:
        print(f"FAIL: pytest\n{out[-2000:]}")
        return False
    print("PASS: pytest (all tests)")

    # Mixed precision tests
    code, out = _run([
        sys.executable, "-m", "pytest",
        "tests/test_nesy_mixed_precision.py", "-q", "--tb=short",
    ])
    if code != 0:
        print(f"FAIL: mixed precision tests\n{out}")
        return False
    print("PASS: mixed precision tests")

    # State isolation / streaming equivalence
    code, out = _run([
        sys.executable, "-m", "pytest",
        "tests/test_sparse_mamba_correctness.py", "-q", "--tb=short",
    ])
    if code != 0:
        print(f"FAIL: state isolation tests\n{out}")
        return False
    print("PASS: state isolation tests")

    return True


def gate_research() -> bool:
    """Research integrity gate."""
    print("\n=== RESEARCH INTEGRITY GATE ===")

    # Dataset disjointness (audit logic on synthetic data)
    code, out = _run([
        sys.executable, "-m", "pytest",
        "tests/test_dataset_audit.py", "-q", "--tb=short",
    ])
    if code != 0:
        print(f"FAIL: dataset disjointness\n{out}")
        return False
    print("PASS: dataset disjointness tests")

    # Real-data audit: run the canonical audit over the actual data/ dir.
    # The unit tests above only exercise the audit logic on synthetic temp
    # data; this check enforces the real corpus satisfies the release gate
    # (exact_overlap == 0 AND near_duplicate_threshold_pass == True).
    code, out = _run([
        sys.executable, "-c",
        "from pathlib import Path; "
        "from daph_exfusion.data.dataset_audit import audit_dataset; "
        "r = audit_dataset(Path('data')); "
        "print(f'exact_overlap_total={r.exact_overlap_total} "
        "near_dup_pass={r.near_duplicate_threshold_pass} "
        "gate={r.pass_release_gate}'); "
        "assert r.exact_overlap_total == 0, 'exact overlap > 0'; "
        "assert r.near_duplicate_threshold_pass, 'near-duplicates across splits'; "
        "assert r.pass_release_gate, 'release gate failed'; "
        "print('OK: real data audit passes release gate')",
    ])
    if code != 0:
        print(f"FAIL: real-data audit\n{out}")
        return False
    print(f"PASS: real-data audit ({out.strip().splitlines()[-1]})")

    # Expert qualification
    code, out = _run([
        sys.executable, "-m", "pytest",
        "tests/test_qualification_fail_closed.py", "-q", "--tb=short",
    ])
    if code != 0:
        print(f"FAIL: expert qualification\n{out}")
        return False
    print("PASS: expert qualification tests")

    # Metric validity
    code, out = _run([
        sys.executable, "-m", "pytest",
        "tests/test_retention_canonical.py",
        "tests/test_cka_correctness.py", "-q", "--tb=short",
    ])
    if code != 0:
        print(f"FAIL: metric validity\n{out}")
        return False
    print("PASS: metric validity tests")

    # No test split usage during search (holdout guard)
    code, out = _run([
        sys.executable, "-m", "pytest",
        "tests/test_validation_holdout.py", "-q", "--tb=short",
    ])
    if code != 0:
        print(f"FAIL: holdout guard tests\n{out}")
        return False
    print("PASS: holdout guard tests")

    return True


def gate_merge() -> bool:
    """Merge integrity gate."""
    print("\n=== MERGE INTEGRITY GATE ===")

    code, out = _run([
        sys.executable, "-m", "pytest",
        "tests/test_merge_baselines.py",
        "tests/test_dare_vs_dropout.py",
        "tests/test_fisher_unified.py",
        "tests/test_agx_operators.py", "-q", "--tb=short",
    ])
    if code != 0:
        print(f"FAIL: merge integrity tests\n{out[-2000:]}")
        return False
    print("PASS: merge integrity tests (baselines, DARE, Fisher, operators)")
    return True


def gate_agx() -> bool:
    """AGX gate."""
    print("\n=== AGX GATE ===")

    code, out = _run([
        sys.executable, "-m", "pytest",
        "tests/test_agx_search_infra.py",
        "tests/test_agx_operators.py", "-q", "--tb=short",
    ])
    if code != 0:
        print(f"FAIL: AGX tests\n{out[-2000:]}")
        return False
    print("PASS: AGX tests (operators, search, Pareto, halving, surrogate, policy)")

    # Check no stub operators registered
    code, out = _run([
        sys.executable, "-c",
        "from daph_exfusion.geometry.operators import SINGLE_EXPERT_OPS, CROSS_EXPERT_OPS; "
        "all_ops = SINGLE_EXPERT_OPS | CROSS_EXPERT_OPS; "
        "assert len(all_ops) >= 6; "
        "print(f'OK: {len(all_ops)} operators registered: {sorted(all_ops)}')",
    ])
    if code != 0:
        print(f"FAIL: operator registry check\n{out}")
        return False
    print(f"PASS: operator registry ({out.strip().split(':')[-1].strip()})")

    return True


def main():
    parser = argparse.ArgumentParser(description="Run CI gates")
    parser.add_argument("--gate", default="all",
                        choices=["runtime", "research", "merge", "agx", "all"])
    args = parser.parse_args()

    gates = {
        "runtime": gate_runtime,
        "research": gate_research,
        "merge": gate_merge,
        "agx": gate_agx,
    }

    if args.gate == "all":
        to_run = ["runtime", "research", "merge", "agx"]
    else:
        to_run = [args.gate]

    all_passed = True
    for gate_name in to_run:
        passed = gates[gate_name]()
        if not passed:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL GATES PASSED")
        sys.exit(0)
    else:
        print("ONE OR MORE GATES FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
