"""Model bundle discovery + load helpers (Plan 5A Task 9).

Bundle layout (written by scripts/train_xgb.py):
    <model_dir>/xgb_<version>.json
    <model_dir>/calib_<version>.pkl
    <model_dir>/meta_<version>.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.xgb_predictor import XGBPredictor


@dataclass(frozen=True)
class BundleHandle:
    version: str
    booster_path: Path
    calib_path: Path
    meta_path: Path
    mtime: float


def list_bundles(model_dir: Path) -> list[BundleHandle]:
    """Returns bundle handles sorted by booster mtime (oldest first)."""
    out: list[BundleHandle] = []
    for booster in sorted(model_dir.glob("xgb_*.json")):
        version = booster.stem.removeprefix("xgb_")
        calib = model_dir / f"calib_{version}.pkl"
        meta = model_dir / f"meta_{version}.json"
        if not calib.exists() or not meta.exists():
            continue
        out.append(BundleHandle(
            version=version,
            booster_path=booster,
            calib_path=calib,
            meta_path=meta,
            mtime=booster.stat().st_mtime,
        ))
    out.sort(key=lambda b: b.mtime)
    return out


def load_latest_model(model_dir: Path) -> XGBPredictor:
    bundles = list_bundles(model_dir)
    if not bundles:
        raise FileNotFoundError(f"No model bundles in {model_dir}")
    latest = bundles[-1]
    return XGBPredictor.load(str(latest.booster_path), str(latest.calib_path))


def load_meta(bundle: BundleHandle) -> dict[str, Any]:
    return dict(json.loads(bundle.meta_path.read_text()))
