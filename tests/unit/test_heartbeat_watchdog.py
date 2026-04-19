"""Heartbeat watchdog tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config


def _make_engine(tmp_path: Path) -> sa.Engine:
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{db}")


def test_recent_heartbeat_no_halt(tmp_path: Path):
    from scripts.heartbeat_watchdog import check_heartbeat_staleness
    engine = _make_engine(tmp_path)
    now = datetime.now(tz=timezone.utc)
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO heartbeat (ts, trace_id) VALUES (:ts, :tid)"
        ), {"ts": now.isoformat(), "tid": "test"})

    halt_file = tmp_path / "HALT"
    is_stale = check_heartbeat_staleness(engine, halt_file, max_stale_minutes=5)
    assert not is_stale
    assert not halt_file.exists()


def test_stale_heartbeat_creates_halt(tmp_path: Path):
    from scripts.heartbeat_watchdog import check_heartbeat_staleness
    engine = _make_engine(tmp_path)
    old_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO heartbeat (ts, trace_id) VALUES (:ts, :tid)"
        ), {"ts": old_ts.isoformat(), "tid": "test"})

    halt_file = tmp_path / "HALT"
    is_stale = check_heartbeat_staleness(engine, halt_file, max_stale_minutes=5)
    assert is_stale
    assert halt_file.exists()
    assert "heartbeat_stale" in halt_file.read_text()


def test_no_heartbeat_rows_creates_halt(tmp_path: Path):
    from scripts.heartbeat_watchdog import check_heartbeat_staleness
    engine = _make_engine(tmp_path)
    halt_file = tmp_path / "HALT"
    is_stale = check_heartbeat_staleness(engine, halt_file, max_stale_minutes=5)
    assert is_stale
    assert halt_file.exists()


def test_existing_halt_not_clobbered(tmp_path: Path):
    """If HALT is already present (user /halt), watchdog must not overwrite the reason."""
    from scripts.heartbeat_watchdog import check_heartbeat_staleness
    engine = _make_engine(tmp_path)
    halt_file = tmp_path / "HALT"
    halt_file.write_text("user: manual halt via /halt\n")

    is_stale = check_heartbeat_staleness(engine, halt_file, max_stale_minutes=5)
    assert is_stale  # stale (no rows) still reported
    assert "manual halt" in halt_file.read_text()  # original reason preserved
