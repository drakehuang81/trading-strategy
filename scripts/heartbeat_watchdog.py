"""External heartbeat watchdog — spec §6.3.

Separate process, launched by launchd/cron every minute.
Reads latest heartbeat row from SQLite (read-only mode).
If stale > max_stale_minutes, writes ./HALT with reason="heartbeat_stale".

Independence: no shared asyncio loop, no coordination with orchestrator.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlalchemy as sa


def _write_halt_if_absent(halt_file: Path, reason: str) -> None:
    """Only write HALT if file does not exist. Preserves any prior reason."""
    if not halt_file.exists():
        halt_file.write_text(reason)


def check_heartbeat_staleness(
    engine: sa.Engine,
    halt_file: Path,
    max_stale_minutes: int = 5,
) -> bool:
    """Check if heartbeat is stale. Write HALT file if so. Returns True if stale."""
    with engine.connect() as conn:
        row = conn.execute(sa.text(
            "SELECT ts FROM heartbeat ORDER BY ts DESC LIMIT 1"
        )).first()

    if row is None:
        _write_halt_if_absent(halt_file, "heartbeat_stale: no heartbeat rows found\n")
        return True

    last_ts_str = str(row[0])
    last_ts = datetime.fromisoformat(last_ts_str)
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    now = datetime.now(tz=timezone.utc)
    age = now - last_ts

    if age > timedelta(minutes=max_stale_minutes):
        _write_halt_if_absent(
            halt_file,
            f"heartbeat_stale: last heartbeat {age.total_seconds():.0f}s ago "
            f"(threshold: {max_stale_minutes * 60}s)\n",
        )
        return True

    return False


def _record_ping(log_path: Path) -> None:
    """Append current UTC timestamp to watchdog ping log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(datetime.now(tz=timezone.utc).isoformat() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Heartbeat watchdog (external monitor)")
    ap.add_argument("--sqlite", default="data/state.db", help="Path to SQLite database")
    ap.add_argument("--halt-file", default="HALT", help="Path to HALT file")
    ap.add_argument("--max-stale-minutes", type=int, default=5)
    ap.add_argument("--watchdog-log", default="data/watchdog_pings.log",
                    help="Path to watchdog ping log")
    args = ap.parse_args()

    engine = sa.create_engine(f"sqlite:///{args.sqlite}")
    halt_file = Path(args.halt_file)
    is_stale = check_heartbeat_staleness(engine, halt_file, args.max_stale_minutes)

    _record_ping(Path(args.watchdog_log))

    if is_stale:
        print("STALE — HALT file written", file=sys.stderr)
        sys.exit(1)
    else:
        print("OK — heartbeat is fresh")
        sys.exit(0)


if __name__ == "__main__":
    main()
