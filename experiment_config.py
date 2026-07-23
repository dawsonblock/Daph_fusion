from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from research_metrics import seed_everything


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    base_model: str
    base_revision: str
    experts: Tuple[str, ...]
    expert_revisions: Tuple[str, ...]
    merge_method: str
    lambda_scale: float
    dare_drop_probability: float
    ties_trim_fraction: float
    fisher_blend: float
    qualification_dataset: str
    calibration_dataset: str
    validation_dataset: str
    test_dataset: str

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = json.load(f)
        data["experts"] = tuple(data["experts"])
        data["expert_revisions"] = tuple(data["expert_revisions"])
        return cls(**data)


class MultiSeedExperimentRunner:
    """
    Phase 5 Multi-Seed Experiment Runner.
    Runs merge experiments across a defined list of stochastic seeds
    (e.g., [11, 23, 37, 51, 73]) and aggregates results.
    """

    def __init__(self, seeds: Tuple[int, ...] = (11, 23, 37, 51, 73)) -> None:
        self.seeds = seeds

    def run_multi_seed(
        self,
        config: ExperimentConfig,
        run_fn: Any,
        output_dir: str = "results/merge_experiments",
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        config.to_json(os.path.join(output_dir, "config.json"))

        seed_results = []
        for seed in self.seeds:
            # Explicitly seed all RNG sources and get generator
            generator = seed_everything(seed)

            # Execute run function for this seed
            result = run_fn(config, seed, generator)
            seed_results.append(result)

            # Save individual seed result
            seed_file = os.path.join(output_dir, f"seed_{seed}.json")
            with open(seed_file, "w") as f:
                json.dump(result, f, indent=2)

        # Aggregate across seeds
        aggregate_summary = {
            "num_seeds": len(self.seeds),
            "seeds": list(self.seeds),
            "seed_results": seed_results,
        }
        with open(os.path.join(output_dir, "aggregate.json"), "w") as f:
            json.dump(aggregate_summary, f, indent=2)

        return aggregate_summary
