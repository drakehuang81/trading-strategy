"""Plan 5A end-to-end smoke: trained model + real-shaped klines + one scan."""
from __future__ import annotations

import asyncio
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from orchestrator import Orchestrator, OrchestratorConfig
from pipeline import scheduled_macro_scan


def _fake_klines() -> pd.DataFrame:
    # 250 bars of plausible ETHUSDT 1h.
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    closes = 3000.0 + np.cumsum(rng.normal(0, 5.0, size=250))
    return pd.DataFrame({
        "open":   closes,
        "high":   closes + 5,
        "low":    closes - 5,
        "close":  closes,
        "volume": np.full(250, 1.0),
    }, index=pd.DatetimeIndex([base + timedelta(hours=i) for i in range(250)],
                               name="open_time"))


def _seed_model_dir(model_dir: Path, feature_order: list[str]) -> str:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(feature_order)))
    y = (X[:, 0] > 0).astype(int)
    booster = xgb.XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss")
    booster.fit(X, y)
    version = "smoke00000001"
    booster.save_model(str(model_dir / f"xgb_{version}.json"))
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.5, 0.9], [0, 1, 1])
    with open(model_dir / f"calib_{version}.pkl", "wb") as fh:
        pickle.dump({"calibrator": iso, "feature_order": feature_order}, fh)
    (model_dir / f"meta_{version}.json").write_text(json.dumps({
        "model_version": version,
        "calibration_method": "isotonic",
        "feature_order": feature_order,
    }))
    return version


@pytest.mark.e2e
async def test_real_data_smoke_inserts_proposal_row(tmp_path):
    # Discover the actual feature_order our registry would emit for these
    # synthetic klines (avoids hardcoding the column list).
    from features.registry import build_default_registry, flatten_features
    feats = build_default_registry().compute_all(_fake_klines(),
                                                  as_of=_fake_klines().index[-1])
    feature_order = sorted(flatten_features(feats).keys())
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _seed_model_dir(model_dir, feature_order)

    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        use_trained_model=True,
        model_dir=str(model_dir),
        drift_reference_path=str(tmp_path / "missing.json"),  # absent on purpose
        ollama_host="http://127.0.0.1:0",  # no Ollama; Ensemble fallback path
        long_threshold=0.0,                # accept any prob_up so we get a proposal
        short_threshold=0.0,
    )

    orch = Orchestrator(cfg)
    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=_fake_klines())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        await orch.boot()
    assert orch.ctx is not None

    # OllamaClient scheduler is never started (orch.boot() does not call
    # orch.run()), so OllamaClient._acquire() blocks forever on both
    # complete() (used by Ensemble → GemmaContextProvider) and
    # chat() (used by ChatLLM.explain for accepted-proposal rationale).
    # Patch both to raise immediately so Ensemble falls back to
    # LLM_UNAVAILABLE_MARKER and ChatLLM.explain is skipped via its
    # except-branch in _scan_symbol.
    err = ConnectionRefusedError("no ollama in test")
    with patch("models.llm.ollama_client.OllamaClient.complete",
               new=AsyncMock(side_effect=err)), \
         patch("models.llm.ollama_client.OllamaClient.chat",
               new=AsyncMock(side_effect=err)):
        await scheduled_macro_scan(orch.ctx, trace_id="smoke")

    engine = sa.create_engine(f"sqlite:///{cfg.sqlite_path}")
    with engine.begin() as conn:
        row_count = conn.execute(sa.text("SELECT COUNT(*) FROM proposals")).scalar()
    assert row_count >= 1, "no proposal rows after scheduled_macro_scan"

    await fake_kline.close()
