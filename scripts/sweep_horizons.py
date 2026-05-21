"""Horizon sweep — Plan 5E Task 1.

For each horizon H in --horizons:
  1. build_labels.py --horizon H --out <out>/labels_h<H>.parquet
  2. train_xgb.py --labels <out>/labels_h<H>.parquet --out <out>/h<H>/
  3. parse the trained meta_*.json for the chosen calibrator's Brier
Print a comparison table at the end.

Usage:
    python -m scripts.sweep_horizons \
        --kline data/history/ETHUSDT_1h.parquet \
        --features data/training/ETHUSDT_1h_features.parquet \
        --horizons 4,24,48,96 \
        --out-root models/sweep \
        --sqlite-path data/state.db
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SweepResult:
    horizon: int
    model_version: str | None
    calibration_method: str | None
    brier: float | None
    error: str | None = None


def parse_brier_from_meta(meta_path: Path) -> tuple[float, str, str]:
    """Returns (chosen_brier, model_version, calibration_method)."""
    meta = json.loads(meta_path.read_text())
    method = meta["calibration_method"]
    brier = float(meta[f"brier_{method}"])
    return brier, meta["model_version"], method


def run_sweep(
    *,
    horizons: list[int],
    kline: Path,
    features: Path,
    out_root: Path,
    sqlite_path: Path,
) -> list[SweepResult]:
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[SweepResult] = []
    for h in horizons:
        labels_path = out_root / f"labels_h{h}.parquet"
        model_dir = out_root / f"h{h}"
        # Step 1: build labels.
        rc = subprocess.run([
            "python", "scripts/build_labels.py",
            "--kline", str(kline),
            "--out", str(labels_path),
            "--horizon", str(h),
        ], capture_output=True, text=True)
        if rc.returncode != 0:
            results.append(SweepResult(
                horizon=h, model_version=None, calibration_method=None,
                brier=None, error=f"build_labels failed: {rc.stderr.strip()}",
            ))
            continue

        # Step 2: train.
        rc = subprocess.run([
            "python", "scripts/train_xgb.py",
            "--features", str(features),
            "--labels", str(labels_path),
            "--out", str(model_dir),
            "--sqlite-path", str(sqlite_path),
        ], capture_output=True, text=True)
        if rc.returncode != 0:
            results.append(SweepResult(
                horizon=h, model_version=None, calibration_method=None,
                brier=None, error=f"train_xgb failed: {rc.stderr.strip()}",
            ))
            continue

        # Step 3: parse latest meta JSON in model_dir.
        meta_files = sorted(model_dir.glob("meta_*.json"),
                            key=lambda p: p.stat().st_mtime)
        if not meta_files:
            results.append(SweepResult(
                horizon=h, model_version=None, calibration_method=None,
                brier=None, error="no meta JSON written by train_xgb",
            ))
            continue
        brier, version, method = parse_brier_from_meta(meta_files[-1])
        results.append(SweepResult(
            horizon=h, model_version=version,
            calibration_method=method, brier=brier, error=None,
        ))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--horizons", required=True, type=str,
                    help="comma-separated horizons, e.g. 4,24,48,96")
    ap.add_argument("--out-root", default=Path("models/sweep"), type=Path)
    ap.add_argument("--sqlite-path", default="data/state.db")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    results = run_sweep(
        horizons=horizons,
        kline=args.kline,
        features=args.features,
        out_root=args.out_root,
        sqlite_path=Path(args.sqlite_path),
    )

    print(f"\nHorizon sweep — {len(results)} runs\n")
    print(f"{'H':<6} {'BRIER':<8} {'CALIB':<10} {'MODEL':<14} ERROR")
    print("-" * 80)
    for r in results:
        b = f"{r.brier:.4f}" if r.brier is not None else "—"
        v = r.model_version or "—"
        c = r.calibration_method or "—"
        e = r.error or ""
        print(f"{r.horizon:<6} {b:<8} {c:<10} {v:<14} {e}")
    print()


if __name__ == "__main__":
    main()
