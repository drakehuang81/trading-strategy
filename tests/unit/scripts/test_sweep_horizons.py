"""Horizon sweep automation — Plan 5E Task 1."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.sweep_horizons import (
    SweepResult,
    parse_brier_from_meta,
    run_sweep,
)


def test_parse_brier_picks_chosen_calibration_method(tmp_path):
    """parse_brier_from_meta returns the brier of whichever calibrator
    train_xgb chose (not always isotonic)."""
    meta = tmp_path / "meta_v1.json"
    meta.write_text(json.dumps({
        "model_version": "v1",
        "calibration_method": "platt",
        "brier_isotonic": 0.27,
        "brier_platt": 0.23,
    }))
    chosen, version, calib = parse_brier_from_meta(meta)
    assert chosen == 0.23
    assert version == "v1"
    assert calib == "platt"


def test_parse_brier_handles_isotonic_choice(tmp_path):
    meta = tmp_path / "meta_v2.json"
    meta.write_text(json.dumps({
        "model_version": "v2",
        "calibration_method": "isotonic",
        "brier_isotonic": 0.21,
        "brier_platt": 0.25,
    }))
    chosen, version, calib = parse_brier_from_meta(meta)
    assert chosen == 0.21
    assert calib == "isotonic"


def test_run_sweep_invokes_build_labels_and_train_per_horizon(tmp_path, monkeypatch):
    """For each horizon, run_sweep calls build_labels then train_xgb,
    collects the resulting meta JSON's chosen Brier."""
    kline = tmp_path / "kline.parquet"
    features = tmp_path / "features.parquet"
    out_root = tmp_path / "sweep_models"
    sqlite = tmp_path / "state.db"
    kline.touch()
    features.touch()

    def fake_run(cmd, **kwargs):
        # Identify whether this was a build_labels or train_xgb call.
        if "build_labels.py" in " ".join(cmd):
            # find horizon from cmd
            h = int(cmd[cmd.index("--horizon") + 1])
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"")  # placeholder
        elif "train_xgb.py" in " ".join(cmd):
            out_dir = Path(cmd[cmd.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "meta_fake.json").write_text(json.dumps({
                "model_version": "fake",
                "calibration_method": "platt",
                "brier_isotonic": 0.27,
                "brier_platt": 0.24,
            }))
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.sweep_horizons.subprocess.run", fake_run)

    results = run_sweep(
        horizons=[4, 24],
        kline=kline,
        features=features,
        out_root=out_root,
        sqlite_path=sqlite,
    )
    assert len(results) == 2
    assert results[0].horizon == 4
    assert results[1].horizon == 24
    assert all(isinstance(r, SweepResult) for r in results)
    assert all(r.brier == 0.24 for r in results)
    assert all(r.calibration_method == "platt" for r in results)


def test_run_sweep_continues_on_train_failure(tmp_path, monkeypatch):
    """If train fails for one horizon, continue with the rest; mark that result with brier=None."""
    kline = tmp_path / "kline.parquet"
    features = tmp_path / "features.parquet"
    out_root = tmp_path / "sweep_models"
    sqlite = tmp_path / "state.db"
    kline.touch()
    features.touch()

    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        from types import SimpleNamespace
        if "train_xgb.py" in " ".join(cmd):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First training call fails (e.g. for horizon=4).
                return SimpleNamespace(returncode=1, stdout="", stderr="boom")
            # Second call succeeds.
            out_dir = Path(cmd[cmd.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "meta_ok.json").write_text(json.dumps({
                "model_version": "ok",
                "calibration_method": "isotonic",
                "brier_isotonic": 0.22,
                "brier_platt": 0.25,
            }))
        else:
            # build_labels: write a placeholder file.
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.sweep_horizons.subprocess.run", fake_run)

    results = run_sweep(
        horizons=[4, 24],
        kline=kline,
        features=features,
        out_root=out_root,
        sqlite_path=sqlite,
    )
    assert results[0].brier is None
    assert "train" in results[0].error.lower()
    assert results[1].brier == 0.22
    assert results[1].error is None
