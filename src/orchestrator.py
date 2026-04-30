"""Orchestrator — spec §4.8. Single asyncio TaskGroup main.

Plan-4: full wiring — build_scan_context + APScheduler + Telegram +
event consumer + drift monitor. Graceful shutdown lands in Task 6.
"""
from __future__ import annotations

import asyncio
import signal
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import sqlalchemy as sa
import structlog
from alembic import command
from alembic.config import Config
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from pipeline import ScanContext, scheduled_macro_scan

log = structlog.get_logger()


@dataclass
class OrchestratorConfig:
    sqlite_path: str = "data/state.db"
    halt_file: str = "HALT"
    telegram_token: str = ""
    scan_interval_hours: int = 1
    watchlist: list[str] = field(default_factory=lambda: ["ETHUSDT"])
    notify_chat_id: str = ""
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma2:4b"
    long_threshold: float = 0.58
    short_threshold: float = 0.58
    drift_yaml: str = "config/drift.yaml"
    daily_loss_max_r: float = -2.0
    heartbeat_max_stale_seconds: int = 300
    paper_broker_seed: int = 42
    drift_check_interval_minutes: int = 60
    use_trained_model: bool = False
    model_dir: str = "models"
    drift_reference_path: str = "models/drift_reference.json"
    broker_kind: Literal["paper", "replay", "live"] = "paper"
    replay_kline_path: str = "data/history/ETHUSDT_1h.parquet"
    replay_funding_path: str = "data/funding/ETHUSDT.parquet"


class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg
        self.engine: sa.Engine | None = None
        self._scheduler: AsyncIOScheduler | None = None
        self.ctx: ScanContext | None = None
        self._lifecycle: dict[str, Any] = {}
        self._telegram: Any = None
        self._stop_event: asyncio.Event | None = None

    async def boot(self) -> None:
        if Path(self.cfg.halt_file).exists():
            log.warning("halt_file_present_on_boot", path=self.cfg.halt_file)
            raise SystemExit("HALT file present; refusing to boot")

        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{self.cfg.sqlite_path}")
        command.upgrade(alembic_cfg, "head")
        self.engine = sa.create_engine(f"sqlite:///{self.cfg.sqlite_path}")

        # Lazy import — wiring imports OrchestratorConfig from this module.
        from wiring import build_scan_context
        self.ctx, self._lifecycle = await build_scan_context(self.cfg, self.engine)

        self._scheduler = AsyncIOScheduler()
        log.info("boot_complete", sqlite=self.cfg.sqlite_path,
                 symbols=self.ctx.symbols)

    def request_stop(self) -> None:
        """Idempotent stop request; sets internal asyncio.Event."""
        if self._stop_event is not None:
            self._stop_event.set()

    def is_stopping(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    async def run(self) -> None:
        await self.boot()
        assert self.engine is not None
        assert self._scheduler is not None
        assert self.ctx is not None

        self._stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                pass  # Windows / restricted envs

        # Start OllamaClient scheduler — chat()/complete() block on its PriorityQueue drain.
        await self._lifecycle["ollama_client"].start()

        self._scheduler.add_job(
            self._scheduled_scan,
            "interval",
            hours=self.cfg.scan_interval_hours,
            id="macro_scan",
            name="scheduled_macro_scan",
        )
        self._scheduler.start()

        if self.cfg.telegram_token:
            from interface.telegram_bot import TelegramBot
            self._telegram = TelegramBot(
                token=self.cfg.telegram_token,
                chat_llm=self.ctx.chat_llm,
                halt_manager=self.ctx.halt,
                scan_fn=self._on_demand_scan,
                broker=self.ctx.broker,
                engine=self.engine,
            )
            self.ctx = replace(self.ctx, telegram=self._telegram)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._heartbeat_loop(), name="heartbeat")
                tg.create_task(self._event_consumer_loop(), name="event_consumer")
                tg.create_task(self._drift_monitor_loop(), name="drift_monitor")
                tg.create_task(self._kline_refresh_loop(), name="kline_refresh")
                if self._telegram is not None:
                    tg.create_task(self._telegram_loop(), name="telegram")
                await self._stop_event.wait()  # main blocks here; signal sets it
                raise asyncio.CancelledError()  # unwind TaskGroup
        except* asyncio.CancelledError:
            pass
        finally:
            self._scheduler.shutdown(wait=False)
            await self._lifecycle["ollama_client"].stop()
            kline = self._lifecycle.get("binance_kline")
            if kline is not None:
                try:
                    await kline.close()
                except Exception:
                    log.exception("binance_kline_close_failed")
            log.info("orchestrator_stopped")

    async def _scheduled_scan(self) -> None:
        """APScheduler callback — delegate to pipeline."""
        assert self.ctx is not None
        trace_id = str(uuid.uuid4())
        log.info("scheduled_scan_triggered", trace_id=trace_id)
        await scheduled_macro_scan(self.ctx, trace_id)

    async def _on_demand_scan(self, symbol: str) -> str:
        """Telegram /analyze callback."""
        from pipeline import on_demand_deep_scan
        assert self.ctx is not None
        trace_id = str(uuid.uuid4())
        return await on_demand_deep_scan(self.ctx, symbol, trace_id)

    async def _heartbeat_loop(self) -> None:
        assert self.engine is not None
        assert self._stop_event is not None
        # TODO(spec §4.8): if this insert fails N consecutive ticks,
        # escalate via halt.activate("heartbeat_db_failure", ...) instead
        # of relying on the external stale-heartbeat watchdog alone.
        while not self._stop_event.is_set():
            try:
                with self.engine.begin() as conn:
                    conn.execute(sa.text(
                        "INSERT INTO heartbeat (ts, trace_id) VALUES (:ts, :tid)"
                    ), {"ts": datetime.now(tz=timezone.utc).isoformat(), "tid": "heartbeat"})
            except Exception:
                log.exception("heartbeat_insert_failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=60)
            except asyncio.TimeoutError:
                continue

    async def _event_consumer_loop(self) -> None:
        """Drain broker events → BrokerEventRepo."""
        assert self.ctx is not None
        assert self._stop_event is not None
        events = self.ctx.broker.events()
        event_task: asyncio.Task[Any] | None = None
        try:
            while not self._stop_event.is_set():
                event_task = asyncio.create_task(events.__anext__())
                stop_task = asyncio.create_task(self._stop_event.wait())
                done, pending = await asyncio.wait(
                    [event_task, stop_task], return_when=asyncio.FIRST_COMPLETED
                )
                for p in pending:
                    p.cancel()
                if stop_task in done:
                    return
                try:
                    event = event_task.result()
                    self.ctx.event_repo.insert(event)
                except StopAsyncIteration:
                    return
                except Exception:
                    log.exception("event_persist_failed")
                event_task = None
        finally:
            # Make sure any suspended __anext__() has returned before we
            # aclose(), otherwise asyncio raises "generator is already
            # running". Critical once real broker backends land so WS
            # finally-blocks inside events() can close sockets.
            if event_task is not None and not event_task.done():
                event_task.cancel()
                try:
                    await event_task
                except BaseException:  # noqa: BLE001
                    pass
            try:
                await events.aclose()
            except Exception:
                log.exception("event_consumer_aclose_failed")

    async def _drift_monitor_loop(self) -> None:
        """Every N minutes, check current feature distribution vs reference."""
        assert self.ctx is not None
        assert self._stop_event is not None
        state = self._lifecycle["drift_state"]
        monitor = self._lifecycle["drift_monitor"]
        interval_s = self.cfg.drift_check_interval_minutes * 60
        while True:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval_s)
                return  # stop was set
            except asyncio.TimeoutError:
                pass  # woke up for periodic check
            try:
                if not monitor.reference:
                    continue  # Plan 5 populates reference from scan snapshots
                # TODO(plan-5): build `test` from the rolling feature buffer.
                # Until then, `has_breach` is intentionally skipped so the
                # loop cannot raise a halt from empty data.
                test: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
                breached = monitor.has_breach(test) if test else False
                state["breached"] = breached
                if breached:
                    self.ctx.halt.activate("feature_drift", reason="psi_or_ks_breach")
            except Exception:
                log.exception("drift_monitor_tick_failed")

    async def _telegram_loop(self) -> None:
        """Keep Telegram running under the TaskGroup."""
        assert self._telegram is not None
        assert self._stop_event is not None
        started = False
        try:
            await self._telegram.start()
            started = True
            await self._stop_event.wait()
        finally:
            # Only attempt stop() if start() completed — otherwise we'd
            # drive a half-initialized Application and mask the real cause.
            if started:
                try:
                    await self._telegram.stop()
                except Exception:
                    log.exception("telegram_stop_failed")

    async def _kline_refresh_loop(self) -> None:
        """Refreshes RollingKlineCache every minute. Failures are logged
        and swallowed; cache providers fall back to last known values."""
        if self.ctx is None or "kline_cache" not in self._lifecycle:
            log.warning("kline_refresh_skipped",
                        ctx_set=self.ctx is not None,
                        cache_in_lifecycle="kline_cache" in self._lifecycle)
            return
        cache = self._lifecycle["kline_cache"]
        symbol = self.ctx.symbols[0]
        timeframe = "1h"
        while not self.is_stopping():
            try:
                df = await self.ctx.data_source.fetch_latest(symbol, timeframe, 200)
                cache.ingest(symbol, timeframe, df)
            except Exception:
                log.warning("kline_refresh_failed", symbol=symbol, exc_info=True)
            try:
                if self._stop_event is None:
                    return
                await asyncio.wait_for(self._stop_event.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                continue
            return
