"""CLI for DAPH ExFusion v3 (Phase 28).

Commands:
    daph-merge run config.yaml          — run a merge experiment
    daph-merge fisher ...               — compute Fisher diagonals
    daph-merge activations ...          — collect activation covariance
    daph-merge evaluate ...             — evaluate a merged model
    daph-merge search ...               — run AGX search
    daph-merge verify ...               — verify release gates

No interactive prompts. Support resume and dry-run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_config(config_path: str) -> dict:
    """Load a YAML or JSON config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML configs. Install with: pip install pyyaml")
    elif path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def cmd_run(args: argparse.Namespace) -> int:
    """Run a merge experiment from a config file."""
    config = load_config(args.config)

    method = config.get("method", "task_arithmetic")
    base_model_id = config.get("base", {}).get("model", "")
    experts = config.get("experts", [])

    print(f"DAPH ExFusion v3 — Merge Run")
    print(f"  Method: {method}")
    print(f"  Base: {base_model_id}")
    print(f"  Experts: {len(experts)}")

    if args.dry_run:
        print("  [DRY RUN] No computation performed.")
        print(f"  Config hash: {json.dumps(config, sort_keys=True)[:16]}...")
        return 0

    # TODO: Load models and execute merge
    print("  Model loading and merge execution not yet implemented in CLI.")
    print("  Use the Python API directly:")
    print("    from daph_exfusion.merge import merge_experts, MergeConfig")
    return 1


def cmd_fisher(args: argparse.Namespace) -> int:
    """Compute Fisher diagonals for experts."""
    print(f"DAPH ExFusion v3 — Fisher Computation")
    print(f"  Estimator: {args.estimator}")
    print(f"  Samples: {args.samples}")
    if args.dry_run:
        print("  [DRY RUN]")
        return 0
    print("  Use the Python API:")
    print("    from daph_exfusion.merge.fisher_dense import build_exact_fisher")
    return 1


def cmd_activations(args: argparse.Namespace) -> int:
    """Collect activation covariance."""
    print(f"DAPH ExFusion v3 — Activation Covariance Collection")
    print(f"  Mode: {args.mode}")
    if args.dry_run:
        print("  [DRY RUN]")
        return 0
    print("  Use the Python API:")
    print("    from daph_exfusion.geometry.activations import ActivationBank")
    return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a merged model."""
    print(f"DAPH ExFusion v3 — Evaluation")
    print(f"  Model: {args.model}")
    print(f"  Bootstrap samples: {args.bootstrap}")
    if args.dry_run:
        print("  [DRY RUN]")
        return 0
    return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Run AGX search."""
    print(f"DAPH ExFusion v3 — AGX Search")
    print(f"  Config: {args.config}")
    if args.dry_run:
        print("  [DRY RUN]")
        return 0
    print("  Use the Python API:")
    print("    from daph_exfusion.search.agx import merge_agx")
    return 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify release gates."""
    from daph_exfusion.validation.release_gates import ReleaseGates
    gates = ReleaseGates()
    result = gates.to_dict()
    print(json.dumps(result, indent=2))
    return 0 if gates.paper_ready else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="daph-merge",
        description="DAPH ExFusion v3 — Dense Merge Research Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    p_run = subparsers.add_parser("run", help="Run a merge experiment")
    p_run.add_argument("config", help="Path to config file (YAML or JSON)")
    p_run.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p_run.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    p_run.set_defaults(func=cmd_run)

    # fisher
    p_fisher = subparsers.add_parser("fisher", help="Compute Fisher diagonals")
    p_fisher.add_argument("--estimator", default="exact_per_sample",
                          choices=["exact_per_sample", "microbatch_gradient_square"])
    p_fisher.add_argument("--samples", type=int, default=512)
    p_fisher.add_argument("--dry-run", action="store_true")
    p_fisher.set_defaults(func=cmd_fisher)

    # activations
    p_act = subparsers.add_parser("activations", help="Collect activation covariance")
    p_act.add_argument("--mode", default="diagonal",
                       choices=["full", "diagonal", "low_rank"])
    p_act.add_argument("--dry-run", action="store_true")
    p_act.set_defaults(func=cmd_activations)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a merged model")
    p_eval.add_argument("model", help="Path to merged model")
    p_eval.add_argument("--bootstrap", type=int, default=10000)
    p_eval.add_argument("--dry-run", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    # search
    p_search = subparsers.add_parser("search", help="Run AGX search")
    p_search.add_argument("config", help="Path to search config")
    p_search.add_argument("--dry-run", action="store_true")
    p_search.set_defaults(func=cmd_search)

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify release gates")
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
