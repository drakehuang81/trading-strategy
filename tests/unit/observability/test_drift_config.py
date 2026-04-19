"""Drift config loader tests."""
from __future__ import annotations

from pathlib import Path

from observability.drift_config import DriftConfig, load_drift_config


def test_load_default_yaml(tmp_path: Path):
    y = tmp_path / "drift.yaml"
    y.write_text(
        "reference_window: 1000\n"
        "test_window: 100\n"
        "psi_bins: 10\n"
        "default:\n"
        "  psi_threshold: 0.25\n"
        "  ks_threshold: 0.10\n"
    )
    cfg = load_drift_config(y)
    assert isinstance(cfg, DriftConfig)
    assert cfg.reference_window == 1000
    assert cfg.test_window == 100
    assert cfg.psi_bins == 10
    assert cfg.psi_threshold == 0.25
    assert cfg.ks_threshold == 0.10


def test_load_accepts_string_path(tmp_path: Path):
    y = tmp_path / "drift.yaml"
    y.write_text(
        "reference_window: 500\n"
        "test_window: 50\n"
        "psi_bins: 8\n"
        "default:\n"
        "  psi_threshold: 0.3\n"
        "  ks_threshold: 0.15\n"
    )
    cfg = load_drift_config(str(y))
    assert cfg.reference_window == 500
