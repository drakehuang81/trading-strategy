"""scripts/backtest.py CLI — Plan 5B-3 Task 4."""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression


def _setup_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Seed klines, features, labels, model bundle in tmp_path."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    n = 100
    rng = np.random.default_rng(0)
    closes = 3000.0 + np.cumsum(rng.normal(0, 5.0, size=n))
    idx = pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(n)], name="open_time",
    )
    klines = pd.DataFrame({
        "open": closes, "high": closes + 5, "low": closes - 5,
        "close": closes, "volume": np.full(n, 1.0),
    }, index=idx)
    kline_path = tmp_path / "history" / "ETHUSDT_1h.parquet"
    kline_path.parent.mkdir(parents=True)
    klines.to_parquet(kline_path)

    feature_idx = pd.DatetimeIndex(idx, name="as_of")
    features = pd.DataFrame({
        "a.f1": rng.normal(size=n), "a.f2": rng.normal(size=n),
    }, index=feature_idx)
    feat_path = tmp_path / "training" / "ETHUSDT_1h_features.parquet"
    feat_path.parent.mkdir(parents=True)
    features.to_parquet(feat_path)

    # Tiny model bundle.
    booster = xgb.XGBClassifier(n_estimators=10, max_depth=2, eval_metric="logloss")
    booster.fit(features.to_numpy(), (rng.uniform(size=n) > 0.5).astype(int))
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    version = "test00000001"
    booster.save_model(str(model_dir / f"xgb_{version}.json"))
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.5, 0.9], [0, 1, 1])
    with open(model_dir / f"calib_{version}.pkl", "wb") as fh:
        pickle.dump({"calibrator": iso, "feature_order": ["a.f1", "a.f2"]}, fh)
    (model_dir / f"meta_{version}.json").write_text(json.dumps({
        "model_version": version,
        "calibration_method": "isotonic",
        "feature_order": ["a.f1", "a.f2"],
    }))

    return kline_path, feat_path, model_dir, tmp_path


@pytest.mark.asyncio
async def test_backtest_cli_writes_backtest_runs_row(tmp_path):
    import argparse
    from scripts.backtest import main_async

    kline_path, feat_path, model_dir, root = _setup_artifacts(tmp_path)
    sqlite_path = root / "state.db"
    import alembic.command, alembic.config
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{sqlite_path}")
    alembic.command.upgrade(ac, "head")

    args = argparse.Namespace(
        symbol="ETHUSDT",
        kline=str(kline_path),
        features=str(feat_path),
        funding="",
        model_dir=str(model_dir),
        sqlite_path=str(sqlite_path),
        oos_fraction=0.3,
        long_threshold=0.0,    # always-fire so we get trades
        short_threshold=0.0,
        n_trials=2,
    )
    await main_async(args)

    engine = sa.create_engine(f"sqlite:///{sqlite_path}")
    with engine.begin() as conn:
        rows = conn.execute(sa.text("SELECT * FROM backtest_runs")).fetchall()
    assert len(rows) == 1
