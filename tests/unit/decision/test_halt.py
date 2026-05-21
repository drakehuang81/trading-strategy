"""HaltManager unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from decision.halt import HaltManager, HaltTrigger


def _make_engine(tmp_path: Path) -> sa.Engine:
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{db}")


class AlwaysBreached:
    name = "always_breached"
    def is_breached(self) -> bool:
        return True


class NeverBreached:
    name = "never_breached"
    def is_breached(self) -> bool:
        return False


def test_not_halted_by_default(tmp_path: Path):
    engine = _make_engine(tmp_path)
    halt_file = tmp_path / "HALT"
    mgr = HaltManager(halt_file=halt_file, engine=engine, triggers=[])
    assert not mgr.is_halted()


def test_activate_creates_halt_file(tmp_path: Path):
    engine = _make_engine(tmp_path)
    halt_file = tmp_path / "HALT"
    mgr = HaltManager(halt_file=halt_file, engine=engine, triggers=[])
    mgr.activate(source="daily_loss_kill_switch", reason="day_pnl_r=-2.5")
    assert halt_file.exists()
    assert mgr.is_halted()


def test_activate_persists_halt_event(tmp_path: Path):
    engine = _make_engine(tmp_path)
    halt_file = tmp_path / "HALT"
    mgr = HaltManager(halt_file=halt_file, engine=engine, triggers=[])
    mgr.activate(source="feature_drift", reason="PSI=0.45")
    with engine.connect() as conn:
        row = conn.execute(sa.text("SELECT trigger_source, reason FROM halt_events")).first()
    assert row is not None
    assert row[0] == "feature_drift"
    assert row[1] == "PSI=0.45"


def test_resume_succeeds_when_triggers_clear(tmp_path: Path):
    engine = _make_engine(tmp_path)
    halt_file = tmp_path / "HALT"
    mgr = HaltManager(halt_file=halt_file, engine=engine, triggers=[NeverBreached()])
    mgr.activate(source="test", reason="test")
    assert mgr.is_halted()
    ok, still_active = mgr.attempt_resume()
    assert ok
    assert still_active == []
    assert not halt_file.exists()


def test_resume_fails_when_trigger_still_breached(tmp_path: Path):
    engine = _make_engine(tmp_path)
    halt_file = tmp_path / "HALT"
    mgr = HaltManager(halt_file=halt_file, engine=engine, triggers=[AlwaysBreached()])
    mgr.activate(source="test", reason="test")
    ok, still_active = mgr.attempt_resume()
    assert not ok
    assert "always_breached" in still_active
    assert halt_file.exists()


def test_resume_updates_halt_event(tmp_path: Path):
    engine = _make_engine(tmp_path)
    halt_file = tmp_path / "HALT"
    mgr = HaltManager(halt_file=halt_file, engine=engine, triggers=[NeverBreached()])
    mgr.activate(source="test", reason="test")
    mgr.attempt_resume()
    with engine.connect() as conn:
        row = conn.execute(sa.text("SELECT resumed_at FROM halt_events ORDER BY id DESC LIMIT 1")).first()
    assert row is not None
    assert row[0] is not None  # resumed_at populated


def test_duplicate_activate_is_idempotent(tmp_path: Path):
    engine = _make_engine(tmp_path)
    halt_file = tmp_path / "HALT"
    mgr = HaltManager(halt_file=halt_file, engine=engine, triggers=[])
    mgr.activate(source="test", reason="first")
    mgr.activate(source="test2", reason="second")
    assert mgr.is_halted()
    with engine.connect() as conn:
        count = conn.execute(sa.text("SELECT count(*) FROM halt_events")).scalar()
    assert count == 2  # both events logged
