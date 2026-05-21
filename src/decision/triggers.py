"""Concrete HaltTrigger implementations — spec §5.5.

Each trigger is a cheap, side-effect free boolean check against SQLite
or an in-memory state dict. Constructor stores dependencies; `is_breached()`
performs one query per call and returns True when the trigger fires.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa


@dataclass
class HeartbeatTrigger:
    engine: sa.Engine
    max_stale_seconds: int = 300
    name: str = "heartbeat_stale"

    def is_breached(self) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT ts FROM heartbeat ORDER BY ts DESC LIMIT 1"
            )).first()
        if row is None:
            return True
        last_ts = datetime.fromisoformat(str(row[0]))
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(tz=timezone.utc) - last_ts).total_seconds()
        return age > self.max_stale_seconds


@dataclass
class DailyLossTrigger:
    engine: sa.Engine
    max_loss_r: float = -2.0
    name: str = "daily_loss_kill_switch"

    def is_breached(self) -> bool:
        today = datetime.now(tz=timezone.utc).date().isoformat()
        with self.engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT day_pnl_r FROM session_state WHERE date = :d"
            ), {"d": today}).first()
        if row is None:
            return False
        return float(row[0]) <= self.max_loss_r


@dataclass
class FeatureDriftTrigger:
    """Bridge between FeatureDriftMonitor and HaltManager.

    The drift-monitor loop writes `state["breached"] = True` when
    `FeatureDriftMonitor.has_breach()` returns True; this trigger reads
    that flag so `HaltManager.attempt_resume()` refuses until the loop
    clears it.
    """
    state: dict[str, Any]
    name: str = "feature_drift"

    def is_breached(self) -> bool:
        return bool(self.state.get("breached", False))
