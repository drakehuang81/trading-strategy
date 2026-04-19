"""HaltTrigger concrete implementations — spec §5.5."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from decision.triggers import (
    DailyLossTrigger,
    FeatureDriftTrigger,
    HeartbeatTrigger,
)


def _engine(tmp_path: Path) -> sa.Engine:
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{db}")


def test_heartbeat_trigger_fresh_not_breached(tmp_path: Path):
    engine = _engine(tmp_path)
    now = datetime.now(tz=timezone.utc)
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO heartbeat (ts, trace_id) VALUES (:ts, :tid)"
        ), {"ts": now.isoformat(), "tid": "t"})
    trig = HeartbeatTrigger(engine, max_stale_seconds=300)
    assert trig.name == "heartbeat_stale"
    assert not trig.is_breached()


def test_heartbeat_trigger_stale_breached(tmp_path: Path):
    engine = _engine(tmp_path)
    old = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO heartbeat (ts, trace_id) VALUES (:ts, :tid)"
        ), {"ts": old.isoformat(), "tid": "t"})
    trig = HeartbeatTrigger(engine, max_stale_seconds=300)
    assert trig.is_breached()


def test_heartbeat_trigger_no_rows_breached(tmp_path: Path):
    trig = HeartbeatTrigger(_engine(tmp_path), max_stale_seconds=300)
    assert trig.is_breached()


def test_daily_loss_trigger_above_limit_not_breached(tmp_path: Path):
    engine = _engine(tmp_path)
    today = datetime.now(tz=timezone.utc).date().isoformat()
    now = datetime.now(tz=timezone.utc).isoformat()
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO session_state (date, consecutive_wins, day_pnl_r, last_update_ts) "
            "VALUES (:d, :cw, :pnl, :ts)"
        ), {"d": today, "cw": 0, "pnl": -1.5, "ts": now})
    trig = DailyLossTrigger(engine, max_loss_r=-2.0)
    assert trig.name == "daily_loss_kill_switch"
    assert not trig.is_breached()


def test_daily_loss_trigger_below_limit_breached(tmp_path: Path):
    engine = _engine(tmp_path)
    today = datetime.now(tz=timezone.utc).date().isoformat()
    now = datetime.now(tz=timezone.utc).isoformat()
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO session_state (date, consecutive_wins, day_pnl_r, last_update_ts) "
            "VALUES (:d, :cw, :pnl, :ts)"
        ), {"d": today, "cw": 0, "pnl": -2.5, "ts": now})
    trig = DailyLossTrigger(engine, max_loss_r=-2.0)
    assert trig.is_breached()


def test_daily_loss_trigger_no_row_not_breached(tmp_path: Path):
    trig = DailyLossTrigger(_engine(tmp_path), max_loss_r=-2.0)
    assert not trig.is_breached()


def test_feature_drift_trigger_clean_not_breached():
    state = {"breached": False}
    trig = FeatureDriftTrigger(state)
    assert trig.name == "feature_drift"
    assert not trig.is_breached()


def test_feature_drift_trigger_breached():
    state = {"breached": True}
    trig = FeatureDriftTrigger(state)
    assert trig.is_breached()
