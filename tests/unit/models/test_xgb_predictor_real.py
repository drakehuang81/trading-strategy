"""Offline integration: trains on fixture CSV, loads, predicts."""
import asyncio
from pathlib import Path

import pytest


@pytest.mark.slow
def test_trained_predictor_roundtrip(tmp_path: Path):
    from scripts.train_xgb import train
    from models.xgb_predictor import XGBPredictor

    fixture = Path("tests/fixtures/ethusdt_1h_sample.csv")
    if not fixture.exists():
        pytest.skip("fixture CSV not available")

    out_dir = tmp_path / "models"
    mv = train(fixture, out_dir)

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
