"""HaltManager — spec §5.5 / §7.3.

Centralises HALT state across 5 trigger sources:
1. Manual `touch ./HALT`
2. Telegram `/halt`
3. DailyLossKillSwitch (daily loss exceeds -2R)
4. FeatureDriftMonitor (PSI/KS breach)
5. External heartbeat watchdog (stale > 5 min)

Recovery is via `/resume` only — re-evaluates all triggers; refuses
if any still breached. Manual file removal alone does NOT resume.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import sqlalchemy as sa
import structlog

log = structlog.get_logger()


class HaltTrigger(Protocol):
    """A check that can block resume."""
    name: str
    def is_breached(self) -> bool: ...


class HaltManager:
    def __init__(
        self,
        halt_file: Path,
        engine: sa.Engine,
        triggers: list[HaltTrigger],
    ) -> None:
        self._halt_file = halt_file
        self._engine = engine
        self._triggers = triggers

    def is_halted(self) -> bool:
        return self._halt_file.exists()

    def activate(self, source: str, reason: str) -> None:
        """Write HALT file and persist halt_event row."""
        self._halt_file.write_text(f"{source}: {reason}\n")
        with self._engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO halt_events (activated_at, trigger_source, reason) "
                "VALUES (:ts, :src, :reason)"
            ), {
                "ts": datetime.now(tz=timezone.utc),
                "src": source,
                "reason": reason,
            })
        log.warning("halt_activated", source=source, reason=reason)

    def attempt_resume(self) -> tuple[bool, list[str]]:
        """Re-evaluate all triggers. Resume only if all clear.

        Returns (success, list of still-breached trigger names).
        """
        breached = [t.name for t in self._triggers if t.is_breached()]
        if breached:
            log.warning("resume_refused", still_breached=breached)
            return False, breached
        self._halt_file.unlink(missing_ok=True)
        self._mark_resumed()
        log.info("halt_resumed")
        return True, []

    def _mark_resumed(self) -> None:
        """Set resumed_at on the most recent un-resumed halt_event."""
        with self._engine.begin() as conn:
            conn.execute(sa.text(
                "UPDATE halt_events SET resumed_at = :ts "
                "WHERE id = (SELECT id FROM halt_events "
                "WHERE resumed_at IS NULL ORDER BY id DESC LIMIT 1)"
            ), {"ts": datetime.now(tz=timezone.utc)})
