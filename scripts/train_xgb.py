"""Trains XGBoost on a precomputed features+labels parquet pair, runs
walk-forward CV, picks isotonic vs Platt by OOS Brier (Plan 5A Task 8).

Writes:
    <out_dir>/xgb_<model_version>.json   (booster)
    <out_dir>/calib_<model_version>.pkl  (calibrator + feature_order)
    <out_dir>/meta_<model_version>.json  (training window, calib comparison)
    <out_dir>/drift_reference.json       (overwritten by Task 10)

Inserts a row into model_versions.

Usage:
    python scripts/train_xgb.py \
        --features data/training/ETHUSDT_1h_features.parquet \
        --labels   data/training/ETHUSDT_1h_labels.parquet \
        --out      models
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class CalibrationChoice:
    method: str               # "isotonic" or "platt"
    brier_isotonic: float
    brier_platt: float
    calibrator: object        # the chosen, fit calibrator (last fold)


@dataclass
class BundleMeta:
    model_version: str
    calibration_method: str
    brier_isotonic: float
    brier_platt: float
    feature_order: list[str]


def _fit_booster(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    booster = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", tree_method="hist",
    )
    booster.fit(X_train, y_train)
    return booster


def _fit_isotonic(raw: np.ndarray, y_calib: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, y_calib)
    return iso


def _fit_platt(raw: np.ndarray, y_calib: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression()
    lr.fit(raw.reshape(-1, 1), y_calib)
    return lr


def walk_forward_calibration_choice(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> CalibrationChoice:
    """Average OOS Brier across folds for isotonic vs Platt; pick the
    lower one. Returns the chosen calibrator fit on the LAST fold's
    calibration set so it sees the most recent regime."""
    # TODO(plan-5b): consider gap=horizon in TimeSeriesSplit to prevent
    # label leakage between train/calib folds (label uses close[t+H]).
    tscv = TimeSeriesSplit(n_splits=n_splits)
    bs_iso: list[float] = []
    bs_platt: list[float] = []
    last_iso: IsotonicRegression | None = None
    last_platt: LogisticRegression | None = None
    splits = list(tscv.split(X))
    for fit_idx, calib_idx in splits:
        # Within a fold, split fit_idx in two: first 80% train booster,
        # last 20% fit calibrator. calib_idx is the OOS chunk we score on.
        cut = int(len(fit_idx) * 0.8)
        train_idx, cal_idx = fit_idx[:cut], fit_idx[cut:]
        booster = _fit_booster(X.iloc[train_idx], y.iloc[train_idx])
        raw_cal = booster.predict_proba(X.iloc[cal_idx])[:, 1]
        iso = _fit_isotonic(raw_cal, y.iloc[cal_idx].to_numpy())
        platt = _fit_platt(raw_cal, y.iloc[cal_idx].to_numpy())

        raw_oos = booster.predict_proba(X.iloc[calib_idx])[:, 1]
        p_iso = iso.transform(raw_oos)
        p_platt = platt.predict_proba(raw_oos.reshape(-1, 1))[:, 1]
        bs_iso.append(brier_score_loss(y.iloc[calib_idx], p_iso))
        bs_platt.append(brier_score_loss(y.iloc[calib_idx], p_platt))
        last_iso, last_platt = iso, platt

    avg_iso = float(np.mean(bs_iso))
    avg_platt = float(np.mean(bs_platt))
    if avg_iso <= avg_platt:
        return CalibrationChoice("isotonic", avg_iso, avg_platt, last_iso)
    return CalibrationChoice("platt", avg_iso, avg_platt, last_platt)


def write_drift_reference(X: pd.DataFrame, out_path: Path,
                          max_samples: int = 5000) -> None:
    """Persists per-column samples for use as FeatureDriftMonitor reference."""
    rng = np.random.default_rng(0)
    blob: dict[str, list[float]] = {}
    for col in X.columns:
        vals = X[col].dropna().to_numpy()
        if len(vals) > max_samples:
            idx = rng.choice(len(vals), size=max_samples, replace=False)
            vals = vals[idx]
        blob[col] = [float(v) for v in vals]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob))


def train_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    training_window_start: str,
    training_window_end: str,
    n_splits: int = 5,
) -> BundleMeta:
    choice = walk_forward_calibration_choice(X, y, n_splits=n_splits)

    # Final booster fit on ALL training rows.
    final_booster = _fit_booster(X, y)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_version = hashlib.sha256(
        final_booster.get_booster().save_raw()
    ).hexdigest()[:12]

    booster_path = out_dir / f"xgb_{model_version}.json"
    final_booster.save_model(str(booster_path))
    calib_path = out_dir / f"calib_{model_version}.pkl"
    with open(calib_path, "wb") as fh:
        pickle.dump({
            "calibrator": choice.calibrator,
            "feature_order": list(X.columns),
        }, fh)

    meta = BundleMeta(
        model_version=model_version,
        calibration_method=choice.method,
        brier_isotonic=choice.brier_isotonic,
        brier_platt=choice.brier_platt,
        feature_order=list(X.columns),
    )
    meta_path = out_dir / f"meta_{model_version}.json"
    meta_path.write_text(json.dumps({
        "model_version": meta.model_version,
        "calibration_method": meta.calibration_method,
        "brier_isotonic": meta.brier_isotonic,
        "brier_platt": meta.brier_platt,
        "training_window_start": training_window_start,
        "training_window_end": training_window_end,
        "feature_order": meta.feature_order,
    }, indent=2))

    write_drift_reference(X, out_dir / "drift_reference.json")
    return meta


def _register(meta: BundleMeta, out_dir: Path,
              window_start: datetime, window_end: datetime,
              sqlite_path: str) -> None:
    engine = sa.create_engine(f"sqlite:///{sqlite_path}")
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT OR REPLACE INTO model_versions "
            "(ml_model_version, path, training_window_start, training_window_end, "
            " calibration_method, deployed_at) "
            "VALUES (:mv, :path, :s, :e, :cm, :ts)"
        ), {
            "mv": meta.model_version,
            "path": str(out_dir / f"xgb_{meta.model_version}.json"),
            "s": window_start, "e": window_end, "cm": meta.calibration_method,
            "ts": datetime.now(tz=timezone.utc),
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", default=Path("models"), type=Path)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--sqlite-path", default="data/state.db")
    args = ap.parse_args()

    X = pd.read_parquet(args.features)
    y_df = pd.read_parquet(args.labels)
    if not X.index.is_monotonic_increasing:
        raise ValueError(f"features index not monotonic increasing in {args.features}")
    if X.select_dtypes(include="number").shape[1] != X.shape[1]:
        non_numeric = X.select_dtypes(exclude="number").columns.tolist()
        raise ValueError(f"features parquet has non-numeric columns: {non_numeric}")
    # Inner join on as_of -> drop rows where label is NaN.
    joined = X.join(y_df, how="inner")
    label_col = y_df.columns[0]
    X_aligned = joined.drop(columns=[label_col])
    y_aligned = joined[label_col].astype(int)
    if len(X_aligned) < 100:
        raise ValueError(
            f"after join, only {len(X_aligned)} rows — check that "
            f"features and labels parquets have overlapping as_of indices"
        )
    print(f"loaded {len(X_aligned)} rows × {X_aligned.shape[1]} features for training")

    window_start = X_aligned.index.min().to_pydatetime()
    window_end = X_aligned.index.max().to_pydatetime()

    meta = train_walk_forward(
        X=X_aligned, y=y_aligned,
        out_dir=args.out,
        training_window_start=window_start.isoformat(),
        training_window_end=window_end.isoformat(),
        n_splits=args.n_splits,
    )
    _register(meta, args.out,
              window_start=window_start,
              window_end=window_end,
              sqlite_path=args.sqlite_path)
    print(f"trained {meta.model_version}; calib={meta.calibration_method} "
          f"brier_iso={meta.brier_isotonic:.4f} brier_platt={meta.brier_platt:.4f}")


if __name__ == "__main__":
    main()
