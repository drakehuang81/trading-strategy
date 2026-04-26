"""BacktestRunner — Plan 5B-3 Task 4."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backtest.runner import BacktestRunner, BacktestResult
from execution.cost_model import SlippageConfig
from execution.paper_broker import PaperBrokerConfig
from execution.replay_broker import ReplayBroker
from features.registry import flatten_features
from models.xgb_predictor import XGBPredictor


def _make_klines(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    closes = 3000.0 + np.cumsum(rng.normal(0, 5.0, size=n))
    return pd.DataFrame({
        "open":   closes,
        "high":   closes + 5,
        "low":    closes - 5,
        "close":  closes,
        "volume": np.full(n, 1.0),
    }, index=pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(n)], name="open_time",
    ))


def _make_features(klines: pd.DataFrame) -> pd.DataFrame:
    """Synthetic features parquet aligned to klines.index but without
    the warmup period — mimics what build_training_set produces."""
    rng = np.random.default_rng(0)
    n = len(klines)
    return pd.DataFrame({
        "a.f1": rng.normal(size=n),
        "a.f2": rng.normal(size=n),
    }, index=pd.DatetimeIndex(klines.index, name="as_of"))


@pytest.fixture
def cfg():
    return PaperBrokerConfig(
        slippage=SlippageConfig(
            slippage_bps_base=1.0, slippage_bps_per_adv_unit=0.0, adv_stub=1000.0,
        ),
    )


@pytest.mark.asyncio
async def test_runner_produces_equity_series(cfg):
    klines = _make_klines(n=50)
    features = _make_features(klines)
    broker = ReplayBroker(cfg=cfg, klines=klines, symbol="ETHUSDT")
    predictor = XGBPredictor.stub(prob_up=0.7, ml_model_version="stub")
    runner = BacktestRunner(
        symbol="ETHUSDT",
        klines=klines,
        features=features,
        funding=None,
        broker=broker,
        predictor=predictor,
        long_threshold=0.6,
        short_threshold=0.6,
        initial_equity_usdt=10_000.0,
    )
    result = await runner.run(oos_start=klines.index[10],
                              oos_end=klines.index[-1])
    assert isinstance(result, BacktestResult)
    assert len(result.equity) >= 1
    # With prob_up=0.7 stub > long_threshold=0.6, every bar after start
    # should produce a long-side fill. Final equity should differ from start.
    assert result.n_trades >= 1


@pytest.mark.asyncio
async def test_runner_records_no_trades_when_predictor_below_threshold(cfg):
    klines = _make_klines(n=20)
    features = _make_features(klines)
    broker = ReplayBroker(cfg=cfg, klines=klines, symbol="ETHUSDT")
    # prob_up=0.5 below long=0.6 AND below 1-short=0.4 → flat, no proposals.
    predictor = XGBPredictor.stub(prob_up=0.5, ml_model_version="stub")
    runner = BacktestRunner(
        symbol="ETHUSDT", klines=klines, features=features, funding=None,
        broker=broker, predictor=predictor,
        long_threshold=0.6, short_threshold=0.6,
        initial_equity_usdt=10_000.0,
    )
    result = await runner.run(oos_start=klines.index[5],
                              oos_end=klines.index[-1])
    assert result.n_trades == 0
    # All equity values should be the initial equity (no fills, no funding).
    assert (result.equity == 10_000.0).all()
