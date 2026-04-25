"""XGBPredictor — spec §4.3.

Ships in two forms:
  - XGBPredictor.stub(prob_up) → fixed predictor for scaffolding tests
  - XGBPredictor.load(path, calibrator_path) → real trained + calibrated
The `predict()` signature is stable across both.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from features.registry import canonical_hash, flatten_features
from models.base import PredictionBundle


@dataclass
class XGBPredictor:
    _model: Any = None
    _calibrator: Any = None
    _feature_order: tuple[str, ...] = ()
    ml_model_version: str = "stub-v0"
    _fixed_prob: float | None = None

    @classmethod
    def stub(cls, prob_up: float, ml_model_version: str = "stub-v0") -> "XGBPredictor":
        return cls(ml_model_version=ml_model_version, _fixed_prob=prob_up)

    @classmethod
    def load(cls, model_path: str, calib_path: str) -> "XGBPredictor":
        import pickle
        import xgboost as xgb
        booster = xgb.XGBClassifier()
        booster.load_model(model_path)
        with open(calib_path, "rb") as fh:
            meta = pickle.load(fh)
        # Prefer the new "calibrator" key (Plan 5A Task 8); fall back to the
        # original "isotonic" key for bundles produced by the pre-rewrite
        # script.
        calibrator = meta.get("calibrator", meta.get("isotonic"))
        if calibrator is None:
            raise ValueError(f"meta at {calib_path} missing calibrator")
        version = Path(model_path).stem.removeprefix("xgb_")
        return cls(
            _model=booster,
            _calibrator=calibrator,
            _feature_order=tuple(meta["feature_order"]),
            ml_model_version=version,
        )

    async def predict(self, features: dict[str, Any]) -> PredictionBundle:
        prob_up = self._fixed_prob if self._fixed_prob is not None else self._run_model(features)
        direction = "long" if prob_up > 0.52 else ("short" if prob_up < 0.48 else "flat")
        return PredictionBundle(
            direction=direction,
            prob_up=float(prob_up),
            horizon_bars=4,
            size_multiplier=1.0,
            feature_snapshot_hash=canonical_hash(features),
            feature_registry_version="1.0.0",
            ml_model_version=self.ml_model_version,
            llm_prompt_version="none",
            predictions_detail={"xgb_prob_up": prob_up},
        )

    def _run_model(self, features: dict[str, Any]) -> float:
        flat = flatten_features(features)
        row = [flat.get(k, 0.0) for k in self._feature_order]
        # TODO(plan-5b): pass NaN to XGBoost native missing-value handling
        # instead of 0.0; current behavior conflates missing-data with
        # signal=0 and is a known model-quality limitation.
        raw = self._model.predict_proba([row])[0, 1]
        # Isotonic: .transform([raw]) -> array.  Platt (LogisticRegression):
        # .predict_proba([[raw]])[:, 1].
        if hasattr(self._calibrator, "transform"):
            calibrated = self._calibrator.transform([raw])[0]
        else:
            calibrated = self._calibrator.predict_proba([[raw]])[0, 1]
        return float(calibrated)
