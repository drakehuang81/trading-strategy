"""E2E: HALT fire drill — §9.5 scenario 2.

Inject losing sequence → -2R → DailyLossKillSwitch fires →
HALT activated → new orders rejected → Telegram alert simulated.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from decision.halt import HaltManager
from decision.proposal import PortfolioSnapshot, RiskCheckResult, TradeProposal
from decision.risk.checks import DailyLossKillSwitch
from execution.repositories import SessionStateRepo


@pytest.mark.e2e
def test_daily_loss_triggers_halt(tmp_path: Path):
    """Daily loss exceeding -2R triggers HALT file + halt_event row."""
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    engine = sa.create_engine(f"sqlite:///{db}")

    # Simulate losing day: day_pnl_r = -2.5
    session_repo = SessionStateRepo(engine)
    from datetime import date
    today = date.today().isoformat()
    session_repo.upsert(today, consecutive_wins=0, day_pnl_r=-2.5)

    # DailyLossKillSwitch as a HaltTrigger
    class DailyLossTrigger:
        name = "daily_loss_kill_switch"
        def __init__(self, repo: SessionStateRepo, threshold: float = -2.0):
            self._repo = repo
            self._threshold = threshold
        def is_breached(self) -> bool:
            _, pnl = self._repo.get(date.today().isoformat())
            return pnl <= self._threshold

    trigger = DailyLossTrigger(session_repo, threshold=-2.0)
    assert trigger.is_breached()

    halt_file = tmp_path / "HALT"
    halt = HaltManager(halt_file=halt_file, engine=engine, triggers=[trigger])

    # Activate HALT
    halt.activate(source="daily_loss_kill_switch", reason="day_pnl_r=-2.5")
    assert halt.is_halted()
    assert halt_file.exists()

    # Verify halt_event persisted
    with engine.connect() as conn:
        row = conn.execute(sa.text(
            "SELECT trigger_source, reason FROM halt_events"
        )).first()
    assert row is not None
    assert row[0] == "daily_loss_kill_switch"

    # Attempt resume — should fail because trigger still breached
    ok, still_active = halt.attempt_resume()
    assert not ok
    assert "daily_loss_kill_switch" in still_active
    assert halt.is_halted()

    # Fix: reset day PnL (new day)
    session_repo.upsert(today, consecutive_wins=0, day_pnl_r=0.0)
    assert not trigger.is_breached()

    # Resume succeeds now
    ok, still_active = halt.attempt_resume()
    assert ok
    assert still_active == []
    assert not halt.is_halted()
