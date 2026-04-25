from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from models.registry import load_latest_model, list_bundles


def _write_bundle(model_dir: Path, version: str, calibration_method: str = "isotonic") -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(int)
    booster = xgb.XGBClassifier(n_estimators=10, max_depth=2, eval_metric="logloss")
    booster.fit(X, y)
    booster.save_model(str(model_dir / f"xgb_{version}.json"))
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.5, 0.9], [0, 1, 1])
    with open(model_dir / f"calib_{version}.pkl", "wb") as fh:
        pickle.dump({"calibrator": iso, "feature_order": ["a.x", "a.y"]}, fh)
    (model_dir / f"meta_{version}.json").write_text(json.dumps({
        "model_version": version,
        "calibration_method": calibration_method,
        "feature_order": ["a.x", "a.y"],
    }))


def test_list_bundles_returns_versions_in_order(tmp_path):
    _write_bundle(tmp_path, "aaaa00000001")
    _write_bundle(tmp_path, "bbbb00000002")
    bundles = list_bundles(tmp_path)
    assert [b.version for b in bundles] == ["aaaa00000001", "bbbb00000002"]


def test_load_latest_model_returns_xgb_predictor(tmp_path):
    _write_bundle(tmp_path, "aaaa00000001")
    _write_bundle(tmp_path, "bbbb00000002")
    pred = load_latest_model(tmp_path)
    assert pred.ml_model_version == "bbbb00000002"


def test_load_latest_model_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_latest_model(tmp_path)
