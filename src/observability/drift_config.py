"""Drift monitor config loader — reads config/drift.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DriftConfig:
    reference_window: int
    test_window: int
    psi_bins: int
    psi_threshold: float
    ks_threshold: float


def load_drift_config(path: Path | str) -> DriftConfig:
    raw = yaml.safe_load(Path(path).read_text())
    default = raw.get("default", {})
    return DriftConfig(
        reference_window=int(raw["reference_window"]),
        test_window=int(raw["test_window"]),
        psi_bins=int(raw["psi_bins"]),
        psi_threshold=float(default["psi_threshold"]),
        ks_threshold=float(default["ks_threshold"]),
    )
