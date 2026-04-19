import pytest

from models.xgb_predictor import XGBPredictor


@pytest.mark.asyncio
async def test_stub_returns_bundle_with_fixed_prob():
    pred = XGBPredictor.stub(prob_up=0.62, ml_model_version="stub-v0")
    bundle = await pred.predict({"smc": {}, "confidence": {"score": 5}})
    assert bundle.prob_up == 0.62
    assert bundle.direction == "long"
    assert bundle.ml_model_version == "stub-v0"
    assert bundle.feature_snapshot_hash        # non-empty
