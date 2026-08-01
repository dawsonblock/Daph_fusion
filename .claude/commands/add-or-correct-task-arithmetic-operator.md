---
name: add-or-correct-task-arithmetic-operator
description: Workflow command scaffold for add-or-correct-task-arithmetic-operator in Daph_fusion.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-or-correct-task-arithmetic-operator

Use this workflow when working on **add-or-correct-task-arithmetic-operator** in `Daph_fusion`.

## Goal

Implements or fixes a task arithmetic operator, routes it through the pipeline, exports it, and adds corresponding tests and CI validation.

## Common Files

- `daph_exfusion/merge/task_arithmetic.py`
- `daph_exfusion/merge/fisher_task_arithmetic.py`
- `daph_exfusion/merge/pipeline_v3.py`
- `daph_exfusion/merge/__init__.py`
- `tests/test_v3_corrected_dense_merge.py`
- `.github/workflows/exfusion-v3-corrected.yml`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Implement or fix the operator logic in a new or existing file under daph_exfusion/merge/
- Update the pipeline (e.g., pipeline_v3.py) to use the new or corrected operator
- Export the operator in daph_exfusion/merge/__init__.py
- Add or update tests in tests/ (e.g., test_v3_corrected_dense_merge.py)
- Update or add CI workflow for validation and packaging

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.