"""Command-line interface for the ExFusion v3 dense production path.

The CLI intentionally keeps the executable surface small. ``run`` performs a
real TA-0/TA-1/TA-2 merge from local or Hugging Face checkpoints and writes the
merged checkpoint plus an immutable provenance JSON. Search/evaluation remain
Python-API workflows because they require project-specific evaluators.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


def load_config(config_path: str) -> dict:
    """Load a YAML or JSON config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML configs") from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def _model_ref(value: Any, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        ref = value.get("model") or value.get("path") or value.get("checkpoint")
        if isinstance(ref, str) and ref:
            return ref
    raise ValueError(f"{field} must be a model/checkpoint string or mapping")


def _expert_refs(config: dict) -> List[str]:
    values = config.get("experts", [])
    if not isinstance(values, list) or not values:
        raise ValueError("config.experts must contain at least one expert")
    return [_model_ref(value, f"experts[{index}]") for index, value in enumerate(values)]


def _merge_config_from_dict(config: dict):
    from daph_exfusion.merge.types import FisherStabilization, MergeConfig, MergeMethod

    method_text = str(config.get("method", "task_arithmetic")).lower()
    aliases = {
        "ta-0": MergeMethod.TASK_ARITHMETIC,
        "ta0": MergeMethod.TASK_ARITHMETIC,
        "ta-1": MergeMethod.TASK_ARITHMETIC,
        "ta1": MergeMethod.TASK_ARITHMETIC,
        "task_arithmetic": MergeMethod.TASK_ARITHMETIC,
        "weighted_ta": MergeMethod.TASK_ARITHMETIC,
        "ta-2": MergeMethod.FISHER_DENSE,
        "ta2": MergeMethod.FISHER_DENSE,
        "fisher_dense": MergeMethod.FISHER_DENSE,
        "fisher_ta": MergeMethod.FISHER_DENSE,
    }
    if method_text not in aliases:
        raise ValueError(
            "CLI production run supports task_arithmetic/TA-0/TA-1 and "
            "fisher_dense/TA-2 only"
        )

    lambdas_raw = config.get("lambdas", ())
    if lambdas_raw is None:
        lambdas_raw = ()
    if not isinstance(lambdas_raw, (list, tuple)):
        raise ValueError("lambdas must be a list/tuple")

    return MergeConfig(
        method=aliases[method_text],
        task_scale=float(config.get("task_scale", config.get("alpha", 1.0))),
        lambdas=tuple(float(value) for value in lambdas_raw),
        fisher_gamma=float(config.get("fisher_gamma", config.get("gamma", 1.0))),
        fisher_stabilization=FisherStabilization(
            str(config.get("fisher_stabilization", "floor"))
        ),
        fisher_floor_eps=float(config.get("fisher_floor_eps", 1e-8)),
        allow_missing_fisher=bool(config.get("allow_missing_fisher", False)),
        seed=int(config.get("seed", 42)),
    )


def _load_curvature_bank(spec: Any) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load a serialized curvature bank from one .pt/.pth file.

    Expected structure: ``{expert_0: {parameter_name: Tensor}, ...}``.
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError("TA-2 requires curvature_bank: /path/to/curvature.pt")
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"curvature bank not found: {path}")
    bank = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bank, dict):
        raise ValueError("curvature bank file must contain a mapping")
    return bank


def _print_plan(config: dict) -> None:
    base_ref = _model_ref(config.get("base", {}), "base")
    experts = _expert_refs(config)
    print("DAPH ExFusion v3 — Dense Merge")
    print(f"  Method: {config.get('method', 'task_arithmetic')}")
    print(f"  Base: {base_ref}")
    print(f"  Experts: {len(experts)}")
    print(f"  Alpha: {config.get('task_scale', config.get('alpha', 1.0))}")
    print(f"  Lambdas: {config.get('lambdas', 'uniform')}")
    print(f"  Output: {config.get('output_dir', 'merged_model')}")


def cmd_run(args: argparse.Namespace) -> int:
    """Load checkpoints, perform a real dense merge, and save it."""
    config_dict = load_config(args.config)
    _print_plan(config_dict)
    merge_config = _merge_config_from_dict(config_dict)
    base_ref = _model_ref(config_dict.get("base", {}), "base")
    expert_refs = _expert_refs(config_dict)

    if args.dry_run:
        print(f"  Config hash: {merge_config.config_hash()}")
        print("  [DRY RUN] No models loaded or written.")
        return 0

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for CLI model loading; install the project dependencies"
        ) from exc

    device = str(config_dict.get("device", "cpu"))
    output_dir = Path(str(config_dict.get("output_dir", "merged_model")))
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs: Dict[str, Any] = {}
    dtype_text = str(config_dict.get("dtype", "auto")).lower()
    if dtype_text == "float32":
        load_kwargs["torch_dtype"] = torch.float32
    elif dtype_text in ("float16", "fp16"):
        load_kwargs["torch_dtype"] = torch.float16
    elif dtype_text in ("bfloat16", "bf16"):
        load_kwargs["torch_dtype"] = torch.bfloat16

    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_ref, **load_kwargs)
    experts = []
    for index, ref in enumerate(expert_refs):
        print(f"  Loading expert {index + 1}/{len(expert_refs)}: {ref}")
        experts.append(AutoModelForCausalLM.from_pretrained(ref, **load_kwargs))

    curvature_bank = None
    if merge_config.method.value == "fisher_dense":
        curvature_bank = _load_curvature_bank(config_dict.get("curvature_bank"))

    from daph_exfusion.merge.pipeline_v3 import merge_experts

    result = merge_experts(
        base_model,
        experts,
        merge_config,
        curvature_bank=curvature_bank,
        device=device,
    )
    print("  Saving merged checkpoint...")
    result.merged_model.save_pretrained(output_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_ref)
        tokenizer.save_pretrained(output_dir)
    except Exception as exc:  # tokenizer is convenient, not merge-critical
        print(f"  Warning: tokenizer was not saved: {exc}", file=sys.stderr)

    provenance = result.to_provenance_dict()
    provenance.update({
        "base": base_ref,
        "experts": expert_refs,
        "output_dir": str(output_dir),
    })
    with (output_dir / "merge_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)

    print(f"  Done: {output_dir}")
    print(f"  Config hash: {merge_config.config_hash()}")
    return 0


def cmd_fisher(args: argparse.Namespace) -> int:
    """Point to the rigorous Fisher builder without pretending a dataset contract."""
    print("DAPH ExFusion v3 — Fisher")
    print("Use build_exact_fisher(model, calibration_data, ...) from")
    print("daph_exfusion.merge.fisher_dense. The CLI intentionally does not guess")
    print("how your calibration dataset should be tokenized or labelled.")
    return 0 if args.dry_run else 2


def cmd_search(args: argparse.Namespace) -> int:
    """Expose the production search API rather than the retired AGX default."""
    config = load_config(args.config)
    mode = str(config.get("mode", "TA-1")).upper()
    print(f"DAPH ExFusion v3 — Task Arithmetic Search ({mode})")
    print("Search requires an evaluator that returns mean_retention, min_retention,")
    print("and general_regression. Use daph_exfusion.merge.search_task_arithmetic")
    print("from Python so evaluator semantics remain explicit and auditable.")
    return 0 if args.dry_run else 2


def cmd_verify(args: argparse.Namespace) -> int:
    """Print release-gate state."""
    from daph_exfusion.validation.release_gates import ReleaseGates

    gates = ReleaseGates()
    print(json.dumps(gates.to_dict(), indent=2, sort_keys=True))
    return 0 if gates.paper_ready else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daph-merge",
        description="DAPH ExFusion v3 — optimized dense Task Arithmetic",
    )
    subparsers = parser.add_subparsers(dest="command", help="commands")

    run = subparsers.add_parser("run", help="execute TA-0/TA-1/TA-2 from config")
    run.add_argument("config", help="YAML/JSON merge config")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    fisher = subparsers.add_parser("fisher", help="show rigorous Fisher workflow")
    fisher.add_argument("--dry-run", action="store_true")
    fisher.set_defaults(func=cmd_fisher)

    search = subparsers.add_parser("search", help="show/run production TA search contract")
    search.add_argument("config", help="YAML/JSON search config")
    search.add_argument("--dry-run", action="store_true")
    search.set_defaults(func=cmd_search)

    verify = subparsers.add_parser("verify", help="verify release gates")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
