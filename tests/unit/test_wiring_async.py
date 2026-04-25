"""Verifies build_scan_context is async and returns a (ScanContext, lifecycle) pair
even when BinanceKline open() succeeds via an injected fake client."""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest
import sqlalchemy as sa

from orchestrator import OrchestratorConfig
from wiring import build_scan_context


@pytest.mark.asyncio
async def test_build_scan_context_is_async(tmp_path):
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        use_trained_model=False,                 # stay on stub for this unit test
    )
    # Build a real engine + run migrations so repo wiring works.
    import alembic.command, alembic.config
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{cfg.sqlite_path}")
    alembic.command.upgrade(ac, "head")
    engine = sa.create_engine(f"sqlite:///{cfg.sqlite_path}")

    # Patch BinanceKline.open so we never touch the network.
    fake_kline = AsyncMock()
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        assert inspect.iscoroutinefunction(build_scan_context)
        ctx, lifecycle = await build_scan_context(cfg, engine)

    assert ctx.symbols == ["ETHUSDT"]
    assert "binance_kline" in lifecycle
    assert lifecycle["binance_kline"] is fake_kline
