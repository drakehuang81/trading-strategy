from __future__ import annotations

import json
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
        "open": [3000.0] * 50, "high": [3010.0] * 50,
        "low": [2990.0] * 50, "close": [3005.0] * 50, "volume": [1.0] * 50,
    }, index=pd.DatetimeIndex(idx, name="open_time"))


@pytest.mark.asyncio
async def test_drift_reference_loaded_from_json(tmp_path):
    drift_path = tmp_path / "drift.json"
    drift_path.write_text(json.dumps({"a.x": [0.1, 0.2, 0.3]}))
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        drift_reference_path=str(drift_path),
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
        _, lifecycle = await build_scan_context(cfg, engine)

    monitor = lifecycle["drift_monitor"]
    assert "a.x" in monitor.reference
