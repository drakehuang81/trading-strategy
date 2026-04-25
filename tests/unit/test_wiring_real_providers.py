# tests/unit/test_wiring_real_providers.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import sqlalchemy as sa

from orchestrator import OrchestratorConfig
from wiring import build_scan_context


def _fake_klines() -> pd.DataFrame:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    idx = [base + timedelta(hours=i) for i in range(50)]
    return pd.DataFrame({
        "open": [3000.0 + i for i in range(50)],
        "high": [3010.0 + i for i in range(50)],
        "low":  [2990.0 + i for i in range(50)],
        "close": [3005.0 + i for i in range(50)],
        "volume": [1.0] * 50,
    }, index=pd.DatetimeIndex(idx, name="open_time"))


@pytest.mark.asyncio
async def test_wiring_uses_cache_backed_mid_after_seed(tmp_path):
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        use_trained_model=False,
    )
    import alembic.command, alembic.config
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{cfg.sqlite_path}")
    alembic.command.upgrade(ac, "head")
    engine = sa.create_engine(f"sqlite:///{cfg.sqlite_path}")

    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=_fake_klines())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        ctx, lifecycle = await build_scan_context(cfg, engine)

    cache = lifecycle["kline_cache"]
    # Seed already happened during build_scan_context.
    assert cache.last_close("ETHUSDT", "1h") == 3054.0
    # The mid provider held by the policy must read the same value.
    assert ctx.policy.mid_provider("ETHUSDT") == 3054.0
