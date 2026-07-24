"""Experiment provenance manifest (Phase 13).

Records all information needed to regenerate an experiment result:
base model, experts, dataset hashes, merge config, software versions.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelProvenance:
    repo: str = ""
    revision: str = ""
    sha256: str = ""
    path: str = ""


@dataclass
class DatasetProvenance:
    train_hash: str = ""
    calibration_hash: str = ""
    validation_hash: str = ""
    test_hash: str = ""
    general_hash: str = ""


@dataclass
class MergeProvenance:
    algorithm: str = ""
    operator_trace: List[str] = field(default_factory=list)
    scale: float = 0.0
    dare_drop_rate: float = 0.0
    ties_trim_fraction: float = 0.0
    ties_sign_mode: str = "magnitude"
    fisher_gamma: float = 0.0
    lambdas: List[float] = field(default_factory=list)
    seed: int = 42
    is_stochastic: bool = False
    # v3 fields
    method: str = ""
    config_hash: str = ""
    fisher_used: bool = False
    fisher_estimator: str = ""
    activation_covariance_used: bool = False
    trust_region_enforced: bool = False
    kfac_used: bool = False
    base_precision_weight: float = 0.0
    checkpoint_hashes: List[str] = field(default_factory=list)
    dataset_hashes: Dict[str, str] = field(default_factory=dict)


@dataclass
class SoftwareProvenance:
    git_commit: str = ""
    python_version: str = ""
    torch_version: str = ""
    platform: str = ""
    timestamp: str = ""


@dataclass
class ExperimentManifest:
    base_model: ModelProvenance = field(default_factory=ModelProvenance)
    experts: List[ModelProvenance] = field(default_factory=list)
    dataset: DatasetProvenance = field(default_factory=DatasetProvenance)
    merge: MergeProvenance = field(default_factory=MergeProvenance)
    software: SoftwareProvenance = field(default_factory=SoftwareProvenance)
    experiment_id: str = ""

    def compute_id(self) -> str:
        d = {
            "base_model": asdict(self.base_model),
            "experts": [asdict(e) for e in self.experts],
            "dataset": asdict(self.dataset),
            "merge": asdict(self.merge),
            "software": asdict(self.software),
        }
        serialized = json.dumps(d, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        d = {
            "base_model": asdict(self.base_model),
            "experts": [asdict(e) for e in self.experts],
            "dataset": asdict(self.dataset),
            "merge": asdict(self.merge),
            "software": asdict(self.software),
        }
        d["experiment_id"] = self.compute_id()
        return d

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        return d["experiment_id"]


def hash_directory(path: Path) -> str:
    """Compute SHA-256 hash of all files in a directory."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode("utf-8"))
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a single file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_manifest(
    base_model_id: str,
    expert_paths: List[str],
    data_dir: str,
    merge_config: MergeProvenance,
    git_commit: str = "",
) -> ExperimentManifest:
    """Build a complete experiment manifest."""
    import torch

    data_path = Path(data_dir)
    dataset = DatasetProvenance()
    for split in ["train", "calibration", "validation", "test"]:
        split_hashes = []
        for domain_dir in sorted(data_path.iterdir()):
            if domain_dir.is_dir():
                split_file = domain_dir / f"{split}.jsonl"
                if split_file.exists():
                    split_hashes.append(hash_file(split_file))
        if split_hashes:
            combined = hashlib.sha256()
            for h in split_hashes:
                combined.update(h.encode())
            setattr(dataset, f"{split}_hash", combined.hexdigest()[:16])

    # General domain
    general_path = data_path / "general"
    if general_path.exists():
        general_hashes = []
        for split_file in sorted(general_path.glob("*.jsonl")):
            general_hashes.append(hash_file(split_file))
        if general_hashes:
            combined = hashlib.sha256()
            for h in general_hashes:
                combined.update(h.encode())
            dataset.general_hash = combined.hexdigest()[:16]

    experts = []
    for ep in expert_paths:
        e_path = Path(ep)
        sha = ""
        if e_path.exists():
            # Hash safetensors files
            h = hashlib.sha256()
            for f in sorted(e_path.glob("*.safetensors")):
                h.update(f.read_bytes())
            sha = h.hexdigest()[:16] if h.digest() != b'\x00' * 32 else ""
        experts.append(ModelProvenance(
            repo="local",
            path=str(ep),
            sha256=sha,
        ))

    manifest = ExperimentManifest(
        base_model=ModelProvenance(repo="huggingface", revision=base_model_id),
        experts=experts,
        dataset=dataset,
        merge=merge_config,
        software=SoftwareProvenance(
            git_commit=git_commit,
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
            platform=platform.platform(),
            timestamp=datetime.now().isoformat(),
        ),
    )
    manifest.experiment_id = manifest.compute_id()
    return manifest
