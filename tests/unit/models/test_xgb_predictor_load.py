import pickle
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from models.xgb_predictor import XGBPredictor


def _train_tiny_booster() -> xgb.XGBClassifier:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=200) > 0).astype(int)
    booster = xgb.XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss")
    booster.fit(X, y)
    return booster


@pytest.mark.asyncio
async def test_load_uses_calibrator_key(tmp_path):
    booster = _train_tiny_booster()
    booster_path = tmp_path / "xgb_test.json"
    booster.save_model(str(booster_path))
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.5, 0.9], [0, 1, 1])
    meta_path = tmp_path / "calib_test.pkl"
    with open(meta_path, "wb") as fh:
        pickle.dump({
            "calibrator": iso,
            "feature_order": ["x.a", "x.b", "x.c"],
        }, fh)

    pred = XGBPredictor.load(str(booster_path), str(meta_path))
    bundle = await pred.predict({"x": {"a": 0.5, "b": 0.1, "c": -0.2}})
    assert 0.0 <= bundle.prob_up <= 1.0
    assert pred.ml_model_version == "test"


@pytest.mark.asyncio
async def test_load_falls_back_to_isotonic_key(tmp_path):
    booster = _train_tiny_booster()
    booster_path = tmp_path / "xgb_legacy.json"
    booster.save_model(str(booster_path))
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.5, 0.9], [0, 1, 1])
    meta_path = tmp_path / "calib_legacy.pkl"
    with open(meta_path, "wb") as fh:
        pickle.dump({
            "isotonic": iso,
            "feature_order": ["x.a", "x.b", "x.c"],
        }, fh)

    pred = XGBPredictor.load(str(booster_path), str(meta_path))
    bundle = await pred.predict({"x": {"a": 0.5, "b": 0.1, "c": -0.2}})
    assert 0.0 <= bundle.prob_up <= 1.0
