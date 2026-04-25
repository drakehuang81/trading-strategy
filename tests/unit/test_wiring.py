"""Wiring factory — build_scan_context returns a fully wired ScanContext."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from orchestrator import OrchestratorConfig
from pipeline import ScanContext


def _engine(tmp_path: Path) -> sa.Engine:
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{db}")


def _make_drift_yaml(tmp_path: Path) -> Path:
    y = tmp_path / "drift.yaml"
    y.write_text(
        "reference_window: 100\ntest_window: 10\npsi_bins: 5\n"
        "default:\n  psi_threshold: 0.25\n  ks_threshold: 0.10\n"
    )
    return y


def _fake_kline() -> AsyncMock:
    kline = AsyncMock()
    kline.close = AsyncMock()
    return kline


@pytest.mark.asyncio
async def test_build_scan_context_returns_wired_context(tmp_path: Path):
    from wiring import build_scan_context

    engine = _engine(tmp_path)
    drift_yaml = _make_drift_yaml(tmp_path)
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        drift_yaml=str(drift_yaml),
    )

    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=_fake_kline())):
        ctx, lifecycle = await build_scan_context(cfg, engine)

    assert isinstance(ctx, ScanContext)
    assert ctx.symbols == ["ETHUSDT"]
    assert ctx.halt is not None
    assert ctx.data_source is not None
    assert ctx.registry is not None
    assert ctx.ensemble is not None
    assert ctx.policy is not None
    assert ctx.risk is not None
    assert ctx.sizer is not None
    assert ctx.broker is not None
    assert ctx.proposal_repo is not None
    assert ctx.event_repo is not None
    assert ctx.session_repo is not None
    assert ctx.chat_llm is not None

    assert "ollama_client" in lifecycle
    assert "drift_state" in lifecycle
    assert "drift_monitor" in lifecycle
    assert "binance_kline" in lifecycle
    # drift_state is the SAME dict the FeatureDriftTrigger reads from
    assert lifecycle["drift_state"] == {"breached": False}


@pytest.mark.asyncio
async def test_build_scan_context_wires_all_three_halt_triggers(tmp_path: Path):
    from wiring import build_scan_context

    engine = _engine(tmp_path)
    drift_yaml = _make_drift_yaml(tmp_path)
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        drift_yaml=str(drift_yaml),
    )

    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=_fake_kline())):
        ctx, _ = await build_scan_context(cfg, engine)

    trigger_names = {t.name for t in ctx.halt._triggers}
    assert trigger_names == {
        "heartbeat_stale", "daily_loss_kill_switch", "feature_drift",
    }


@pytest.mark.asyncio
async def test_drift_trigger_and_drift_state_alias_same_object(tmp_path: Path):
    """Setting lifecycle['drift_state']['breached']=True must flip the trigger."""
    from wiring import build_scan_context

    engine = _engine(tmp_path)
    drift_yaml = _make_drift_yaml(tmp_path)
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        drift_yaml=str(drift_yaml),
    )

    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=_fake_kline())):
        ctx, lifecycle = await build_scan_context(cfg, engine)

    drift_trigger = next(
        t for t in ctx.halt._triggers if t.name == "feature_drift"
    )
    assert drift_trigger.is_breached() is False
    lifecycle["drift_state"]["breached"] = True
    assert drift_trigger.is_breached() is True
