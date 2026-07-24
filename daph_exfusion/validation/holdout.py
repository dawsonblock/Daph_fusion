"""Final holdout test + config freeze (Phase 20).

AGX must NEVER use the test split for:
  - operator selection
  - lambda selection
  - threshold selection
  - surrogate training
  - early stopping
  - expert qualification
  - Fisher calibration

The final test runs ONCE after configuration freeze. Write
artifacts/final_config.json before test evaluation. Hash it. Then
evaluate the immutable candidate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FrozenConfig:
    """Immutable configuration snapshot before final test evaluation."""
    release: str
    config_hash: str
    frozen_at: str
    config: Dict[str, Any]
    test_split_used_during_search: bool = False  # must be False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "release": self.release,
            "config_hash": self.config_hash,
            "frozen_at": self.frozen_at,
            "config": self.config,
            "test_split_used_during_search": self.test_split_used_during_search,
        }


def compute_config_hash(config: Dict[str, Any]) -> str:
    """SHA-256 hash of the serialized configuration."""
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def freeze_config(
    config: Dict[str, Any],
    release: str = "v2.4.0-correctness",
    output_path: Optional[Path] = None,
) -> FrozenConfig:
    """Freeze the configuration before final test evaluation.

    Writes artifacts/final_config.json with the frozen config and its hash.
    After this point, the config is immutable; any change to the merged
    model invalidates the frozen hash.
    """
    config_hash = compute_config_hash(config)
    frozen = FrozenConfig(
        release=release,
        config_hash=config_hash,
        frozen_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        test_split_used_during_search=False,
    )

    if output_path is None:
        output_path = Path("artifacts/final_config.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(frozen.to_dict(), f, indent=2)

    return frozen


def verify_config_frozen(frozen: FrozenConfig) -> None:
    """Verify that the frozen config has not been tampered with."""
    actual_hash = compute_config_hash(frozen.config)
    if actual_hash != frozen.config_hash:
        raise RuntimeError(
            f"Config hash mismatch: frozen hash {frozen.config_hash} != "
            f"actual hash {actual_hash}. The configuration was modified "
            f"after freezing; the final test result is INVALID."
        )
    if frozen.test_split_used_during_search:
        raise RuntimeError(
            "Config indicates test_split_used_during_search=True. "
            "The test split was contaminated during search; "
            "the final test result is INVALID."
        )


class TestSplitGuard:
    """Runtime guard to prevent test-split usage during search.

    Wrap any function that accesses data splits with this guard to
    ensure the test split is never touched during search/calibration.
    """

    def __init__(self) -> None:
        self._test_accessed = False
        self._in_search = False

    def enter_search_mode(self) -> None:
        """Call before starting search/selection/calibration."""
        self._in_search = True
        self._test_accessed = False

    def exit_search_mode(self) -> None:
        """Call after search is complete, before final test."""
        self._in_search = False

    def check_access(self, split_name: str) -> None:
        """Call when accessing a data split. Raises if test is touched during search."""
        if self._in_search and split_name == "test":
            self._test_accessed = True
            raise RuntimeError(
                "Test split accessed during search mode! "
                "The test split must NEVER be used for operator selection, "
                "lambda selection, threshold selection, surrogate training, "
                "early stopping, expert qualification, or Fisher calibration."
            )

    @property
    def test_was_accessed_during_search(self) -> bool:
        return self._test_accessed
