"""Wiring switches broker by cfg.broker_kind — Plan 5B-2 Task 5."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import sqlalchemy as sa

from orchestrator import OrchestratorConfig
from wiring import build_scan_context


def _seed_klines(path: Path) -> None:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "open":   [3000.0] * 5, "high": [3010.0] * 5, "low": [2990.0] * 5,
        "close":  [3005.0] * 5, "volume": [1.0] * 5,
    }, index=pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(5)], name="open_time",
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _seed_funding(path: Path) -> None:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {"funding_rate": [0.0001] * 3},
        index=pd.DatetimeIndex(
            [base + timedelta(hours=i * 8) for i in range(3)], name="ts",
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _bootstrap_engine(cfg) -> sa.Engine:
    import alembic.command, alembic.config
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{cfg.sqlite_path}")
    alembic.command.upgrade(ac, "head")
    return sa.create_engine(f"sqlite:///{cfg.sqlite_path}")


@pytest.mark.asyncio
async def test_default_broker_kind_is_paper(tmp_path):
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
    )
    engine = _bootstrap_engine(cfg)
    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=pd.DataFrame())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        ctx, _ = await build_scan_context(cfg, engine)
    from execution.paper_broker import PaperBroker
    assert isinstance(ctx.broker, PaperBroker)


@pytest.mark.asyncio
async def test_broker_kind_replay_uses_replay_broker(tmp_path):
    kline_path = tmp_path / "klines.parquet"
    funding_path = tmp_path / "funding.parquet"
    _seed_klines(kline_path)
    _seed_funding(funding_path)
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        broker_kind="replay",
        replay_kline_path=str(kline_path),
        replay_funding_path=str(funding_path),
    )
    engine = _bootstrap_engine(cfg)
    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=pd.DataFrame())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        ctx, _ = await build_scan_context(cfg, engine)
    from execution.replay_broker import ReplayBroker
    assert isinstance(ctx.broker, ReplayBroker)


@pytest.mark.asyncio
async def test_broker_kind_live_uses_live_broker(tmp_path):
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        broker_kind="live",
    )
    engine = _bootstrap_engine(cfg)
    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=pd.DataFrame())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        ctx, _ = await build_scan_context(cfg, engine)
    from execution.live_broker import LiveBroker
    assert isinstance(ctx.broker, LiveBroker)
