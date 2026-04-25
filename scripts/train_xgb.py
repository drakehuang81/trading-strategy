"""Trains XGBoost on historical ETHUSDT features + isotonic calibration.

Labels: 4-bar forward return > 0. Features: build_default_registry().
Writes:
  - models/xgb_<model_version>.json (booster)
  - models/calib_<model_version>.pkl (IsotonicRegression)
  - row in SQLite model_versions

Usage:
  python scripts/train_xgb.py --data data/ETHUSDT_1h_long.csv
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

from features.registry import build_default_registry, flatten_features


def _make_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    reg = build_default_registry()
    rows: list[dict[str, float]] = []
    ys: list[int] = []
    for i, ts in enumerate(df.index):
        if i < 200 or i > len(df) - 5:
            continue
        feats = reg.compute_all(df, as_of=ts)
        flat = flatten_features(feats)
        rows.append(flat)
        y = int(df["close"].iloc[i + 4] > df["close"].iloc[i])
        ys.append(y)
    X = pd.DataFrame(rows).fillna(0.0)
    return X, pd.Series(ys, name="y")


def train(data_path: Path, out_dir: Path) -> str:
    df = pd.read_csv(data_path, parse_dates=["open_time"]).set_index("open_time")
    X, y = _make_dataset(df)

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, calib_idx = list(tscv.split(X))[-1]
    booster = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", tree_method="hist",
    )
    booster.fit(X.iloc[train_idx], y.iloc[train_idx])
    raw_prob = booster.predict_proba(X.iloc[calib_idx])[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_prob, y.iloc[calib_idx])

    out_dir.mkdir(parents=True, exist_ok=True)
    model_version = hashlib.sha256(booster.get_booster().save_raw()).hexdigest()[:12]
    booster.save_model(str(out_dir / f"xgb_{model_version}.json"))
    with open(out_dir / f"calib_{model_version}.pkl", "wb") as fh:
        pickle.dump({"isotonic": isotonic, "feature_order": list(X.columns)}, fh)

    _register(model_version, str(X.index.min()), str(X.index.max()), out_dir)
    return model_version


def _register(model_version: str, start: str, end: str, out_dir: Path) -> None:
    engine = sa.create_engine("sqlite:///data/state.db")
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT OR REPLACE INTO model_versions "
            "(ml_model_version, path, training_window_start, training_window_end, "
            " calibration_method, deployed_at) "
            "VALUES (:mv, :path, :s, :e, :cm, :ts)"
        ), {
            "mv": model_version,
            "path": str(out_dir / f"xgb_{model_version}.json"),
            "s": start, "e": end,
            "cm": "isotonic",
            "ts": datetime.now(tz=timezone.utc),
        })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=Path("models"), type=Path)
    args = ap.parse_args()
    mv = train(args.data, args.out)
    print(f"Trained model_version={mv}")
