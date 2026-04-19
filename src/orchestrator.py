"""Orchestrator — spec §4.8. Single asyncio TaskGroup main.

Plan-2 minimum: boot sequence + manual-trigger scan method. Hourly
scheduler + Telegram + event consumer lifecycle land in Plan 3.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import sqlalchemy as sa
import structlog
from alembic import command
from alembic.config import Config

log = structlog.get_logger()


@dataclass
class OrchestratorConfig:
    sqlite_path: str = "data/state.db"
    halt_file: str = "HALT"


class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg
        self.engine: sa.Engine | None = None

    async def boot(self) -> None:
        if Path(self.cfg.halt_file).exists():
            log.warning("halt_file_present_on_boot", path=self.cfg.halt_file)
            raise SystemExit("HALT file present; refusing to boot")

        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{self.cfg.sqlite_path}")
        command.upgrade(alembic_cfg, "head")
        self.engine = sa.create_engine(f"sqlite:///{self.cfg.sqlite_path}")
        log.info("boot_complete", sqlite=self.cfg.sqlite_path)

    async def run(self) -> None:
        await self.boot()
        async with asyncio.TaskGroup() as tg:
            _ = tg.create_task(self._heartbeat_loop(), name="heartbeat")
            # Plan-3: scheduler, telegram, event_consumer tasks attach here.

    async def _heartbeat_loop(self) -> None:
        while True:
            assert self.engine is not None
            with self.engine.begin() as conn:
                conn.execute(sa.text(
                    "INSERT INTO heartbeat (ts, trace_id) VALUES (:ts, :tid)"
                ), {"ts": datetime.now(tz=timezone.utc).isoformat(), "tid": "boot"})
            await asyncio.sleep(60)
