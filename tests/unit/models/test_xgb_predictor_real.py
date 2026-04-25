"""Offline integration: trains on fixture CSV, loads, predicts."""
import asyncio
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.slow
def test_trained_predictor_roundtrip(tmp_path: Path):
    from scripts.train_xgb import train_walk_forward
    from features.registry import build_default_registry, flatten_features
    from models.xgb_predictor import XGBPredictor

    fixture = Path("tests/fixtures/ethusdt_1h_sample.csv")
    if not fixture.exists():
        pytest.skip("fixture CSV not available")

    df = pd.read_csv(fixture, parse_dates=["open_time"]).set_index("open_time")

    reg = build_default_registry()
    rows: list[dict] = []
    ys: list[int] = []
    for i, ts in enumerate(df.index):
        if i < 200 or i > len(df) - 5:
            continue
        feats = reg.compute_all(df, as_of=ts)
        flat = flatten_features(feats)
        rows.append(flat)
        ys.append(int(df["close"].iloc[i + 4] > df["close"].iloc[i]))

    if len(rows) < 30:
        pytest.skip("fixture too small for walk-forward")

    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(ys, name="y")

    out_dir = tmp_path / "models"
    bundle_meta = train_walk_forward(
        X=X, y=y,
        out_dir=out_dir,
        training_window_start=str(df.index.min()),
        training_window_end=str(df.index.max()),
        n_splits=3,
    )
    mv = bundle_meta.model_version

    pred = XGBPredictor.load(
        str(out_dir / f"xgb_{mv}.json"),
        str(out_dir / f"calib_{mv}.pkl"),
    )
    bundle = asyncio.run(pred.predict({
        "smc": {}, "fib": {}, "liquidity": {},
        "divergence": {}, "funding": {}, "confidence": {},
    }))
    assert 0 <= bundle.prob_up <= 1
    assert bundle.ml_model_version == mv
