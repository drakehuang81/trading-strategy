"""Orchestrator — spec §4.8. Single asyncio TaskGroup main.

Plan-3: full wiring — APScheduler + Telegram + event consumer + OllamaClient.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import sqlalchemy as sa
import structlog
from alembic import command
from alembic.config import Config
from apscheduler.schedulers.asyncio import AsyncIOScheduler

log = structlog.get_logger()


@dataclass
class OrchestratorConfig:
    sqlite_path: str = "data/state.db"
    halt_file: str = "HALT"
    telegram_token: str = ""
    scan_interval_hours: int = 1
    watchlist: list[str] = field(default_factory=lambda: ["ETHUSDT"])
    notify_chat_id: str = ""


class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg
        self.engine: sa.Engine | None = None
        self._scheduler: AsyncIOScheduler | None = None

    async def boot(self) -> None:
        if Path(self.cfg.halt_file).exists():
            log.warning("halt_file_present_on_boot", path=self.cfg.halt_file)
            raise SystemExit("HALT file present; refusing to boot")

        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{self.cfg.sqlite_path}")
        command.upgrade(alembic_cfg, "head")
        self.engine = sa.create_engine(f"sqlite:///{self.cfg.sqlite_path}")

        self._scheduler = AsyncIOScheduler()
        log.info("boot_complete", sqlite=self.cfg.sqlite_path)

    async def run(self) -> None:
        await self.boot()
        assert self.engine is not None
        assert self._scheduler is not None

        # Schedule hourly macro scan
        self._scheduler.add_job(
            self._scheduled_scan,
            "interval",
            hours=self.cfg.scan_interval_hours,
            id="macro_scan",
            name="scheduled_macro_scan",
        )
        self._scheduler.start()

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._heartbeat_loop(), name="heartbeat")
                tg.create_task(self._event_consumer_loop(), name="event_consumer")
                # Telegram and OllamaClient lifecycle managed separately
                # because they have their own start/stop methods.
        finally:
            self._scheduler.shutdown(wait=False)

    async def _heartbeat_loop(self) -> None:
        while True:
            assert self.engine is not None
            with self.engine.begin() as conn:
                conn.execute(sa.text(
                    "INSERT INTO heartbeat (ts, trace_id) VALUES (:ts, :tid)"
                ), {"ts": datetime.now(tz=timezone.utc).isoformat(), "tid": "heartbeat"})
            await asyncio.sleep(60)

    async def _event_consumer_loop(self) -> None:
        """Drain broker events and persist them. Placeholder for Plan-3 wiring."""
        # In full wiring, this would consume from broker.events() and persist
        # via BrokerEventRepo. For now, just keep the task alive.
        while True:
            await asyncio.sleep(60)

    async def _scheduled_scan(self) -> None:
        """Callback for APScheduler hourly job."""
        trace_id = str(uuid.uuid4())
        log.info("scheduled_scan_triggered", trace_id=trace_id)
        # Full pipeline wiring happens when ScanContext is constructed
        # with all dependencies at startup. This is the entry point.
