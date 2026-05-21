# Plan 4 STATUS — Orchestrator Integration (Paper Mode End-to-End)

**Date**: 2026-04-19
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: `0659da4` (Plan 4 plan doc)
**Head commit**: `c21bbe9` (Task 9; this STATUS is the next commit)

## Summary

All 10 tasks complete. `python -m src.cli` now boots a full paper-mode orchestrator: `build_scan_context` wires every Plan 1–3 component into a `ScanContext`, APScheduler fires the hourly `scheduled_macro_scan`, Telegram serves `/positions /status /halt /resume /analyze` against the real engine + broker, `HaltManager` re-evaluates `HeartbeatTrigger + DailyLossTrigger + FeatureDriftTrigger` on `/resume`, and `SIGTERM`/`SIGINT` unwinds the TaskGroup within ~5 s.

Test count: **253 passed** (excluding 1 pre-existing `xgboost`/`libomp` env failure unrelated to Plan 4). Plan 4 added **+8 tests** net over the Plan 3 baseline of 245.

## Task table

| # | Title | Commit(s) | Files |
|---|-------|-----------|-------|
| 1 | `messages.seq` stable ordering | `c8a9504` | Alembic migration + `interface/repositories.py` |
| 2 | Concrete `HaltTrigger` classes | `42f4209` | `decision/triggers.py` (new) |
| 3 | `DriftConfig` loader | `9361d1f` | `observability/drift_config.py` (new) |
| 4 | `build_scan_context` factory | `5b89f2f` | `wiring.py` (new) + `OrchestratorConfig` additions |
| 5 | Orchestrator full wiring | `f5fa430`, `d2e2020` | `orchestrator.py` — `ScanContext` + drift + event consumer + quality fixes |
| 6 | Graceful shutdown (SIGTERM/SIGINT) | `005d1aa`, `9336518` | `orchestrator.py` — `asyncio.Event` + signal handlers + quality fixes |
| 7 | `/status` real state | `2ec4600` | `interface/telegram_bot.py::_cmd_status` |
| 8 | `get_feature_snapshot` live | `1c63896` | `interface/tools.py::_query_feature_snapshot` |
| 9 | E2E orchestrator boot test | `c21bbe9` | `tests/e2e/test_orchestrator_boot.py` (new) |
| 10 | STATUS handoff | (this commit) | this doc |

## Verification checklist

- ✅ Full test suite green — 253 passed (245 Plan 3 baseline + 8 Plan 4 additions)
- ✅ Per-task unit tests + E2E: `test_run_uses_scan_context`, `test_stop_event_unwinds_taskgroup`, `test_cmd_status_*` (×2), `test_get_feature_snapshot_*` (×3), `test_orchestrator_boots_and_stops`
- ✅ `orchestrator.boot()` populates `self.ctx` (non-None ScanContext with 3 halt triggers) and `self._lifecycle` (ollama_client + drift_state + drift_monitor)
- ✅ `orchestrator.run()` starts the OllamaClient scheduler, installs SIGTERM/SIGINT handlers, opens the TaskGroup, stops cleanly within 5 s on `request_stop()`
- ✅ `messages.seq` monotonically increases per `conversation_id`; `history()` order is `seq DESC` then reversed in Python
- ✅ `FeatureDriftTrigger.state` is aliased into `build_scan_context`'s `lifecycle["drift_state"]` (verified by `test_drift_trigger_and_drift_state_alias_same_object`)
- ✅ `HaltManager.attempt_resume()` re-evaluates all 3 triggers (not an unconditional unlink anymore)
- ✅ `/status` reports real halt state + last heartbeat + open positions
- ✅ `get_feature_snapshot` returns the latest proposal's `feature_snapshot_json` (no more `not_implemented` stub)
- ✅ E2E boot-and-stop scenario persists a heartbeat row within 1.2 s and cleanly unwinds the TaskGroup

## Key design decisions landed

- **Plan doc ↔ actual code signature drift**: the plan was written before Plan 1–3 code was finalized. During execution we verified ~9 signatures against live code before dispatching Task 4 (notable corrections: `PaperBroker(cfg, rng, mid_provider)` not `(cfg, broker_events_repo)`; `SpreadGate(max_bps, spread_provider)`; `FixedFractionalSizer(fraction)`; `DailyLossKillSwitch(threshold_r)`; `MaxConcurrentPositions(cap)`; `ToolExecutor(engine, broker)` async not `(broker, proposal_repo, session_repo)` sync; proposals table is `proposals` not `trade_proposals`; `OllamaClient.stop()` not `close()`, and `OllamaClient.start()` is required before any `chat()/complete()` or they hang on the PriorityQueue).
- **Wiring cycle**: `wiring.py` imports `OrchestratorConfig` from `orchestrator.py`; `orchestrator.py` imports `build_scan_context` lazily inside `boot()` to dissolve the cycle.
- **Stop machinery**: `run()` owns the `asyncio.Event`; `request_stop()` is thread-safe-ish (sets event; loop awaits). Main `run()` body awaits the event inside the TaskGroup then `raise asyncio.CancelledError()` to unwind siblings; caught by `except* asyncio.CancelledError`.
- **Async-generator cleanup**: `_event_consumer_loop` tracks the pending `__anext__()` task at outer scope so `finally` can cancel-and-await it before `events.aclose()` — required to avoid `RuntimeError: asynchronous generator is already running` when CancelledError interrupts mid-`asyncio.wait()`.
- **Telegram half-init guard**: `_telegram_loop` only calls `stop()` if `start()` returned, and wraps the stop in `try/except` so a half-initialized Application can't mask the original start() exception.
- **Heartbeat resilience**: wrapped the SQL insert in `try/except log.exception` (responding to Task 5 code-review I5). TODO(spec §4.8): escalate to `halt.activate("heartbeat_db_failure", ...)` after N consecutive failures instead of silent log spam.
- **YAGNI on ProposalRepo**: Task 8 plan asked for `latest_for_symbol` on `ProposalRepo`, but only `get_feature_snapshot` needs that read path, and `tools.py` already bypasses repos for reads (`_query_proposals`, `_query_pnl`). Kept the direct-SQL pattern in `tools.py::_query_feature_snapshot` — no repo API growth.

## What is NOT done (Plan 5+ scope)

- Real `mid_provider` / `atr_provider` / `spread_provider` pulling from a live `data_source` cache (currently stubbed to `3000.0` / `15.0` / `0.0`).
- `BinanceKline` is NOT wired: `build_scan_context` is sync but `BinanceKline.open()` is async. Shipped `_StubDataSource` that returns an empty DataFrame. Plan 5 must either make `build_scan_context` async OR open Binance inside `orchestrator.run()` and inject into `ctx`.
- Drift-monitor reference population — `monitor.reference` is empty today; Plan 5 needs a rolling feature-buffer so `_drift_monitor_loop` can build the `test` dict and call `has_breach`.
- `ReplayBroker` + `LiveBroker` + contract suite.
- Walk-forward backtest + Deflated Sharpe (spec §10.1 gate 2).
- Pre-Live Gate module — 8 gates, exit code, live mode blocker (spec §10).
- `LiveConfirmViaTelegram` reconciliation policy (spec §7.4).
- `NetDirectionalCap` enforcement (enabled after 60+ days of paper data).
- Heartbeat-DB-failure HALT escalation (TODO comment in `_heartbeat_loop`).
- Streaming `ChatLLM` replies (current: full response).
- `mypy --strict` pass (Plan 1 baseline has ~80 errors; Plan 4 code is typed but not strictly verified).
- Deployment: launchd / systemd units; secrets management.

## Known follow-ups

- **Telegram race**: `_telegram_loop` calls `await self._telegram.start()` inside the TaskGroup. If `_scheduled_scan` or `_event_consumer_loop` attempted to use `ctx.telegram.send_message()` before `start()` completed, there's a small window where `Application` is half-initialized. In practice the first scheduled scan is +1h out, but Plan 5 should move `telegram.start()` to `run()` before the TaskGroup opens.
- **Broker `events()` Protocol typing**: `BrokerEventStream.events()` is typed as sync-returning in the Protocol but `PaperBroker.events` is an `async def` async-gen function. Both work because calling either returns an async iterator without awaiting, but `mypy --strict` will eventually flag the mismatch.
- **Drift-monitor test escalation**: when Plan 5 populates `monitor.reference`, it should also remove the `TODO(plan-5)` comment in `_drift_monitor_loop` and add an integration test that actually raises a breach + asserts `halt.activate` was called.
- **Heartbeat → DailyLossTrigger coupling**: `DailyLossTrigger` reads `session_state` by today's date; on the first boot of a new day before any position closes, `session_state` has no row. Trigger currently returns `False` (no breach) in that case — verify this matches Spec §4.7 expectations.

## Handoff

Plan 4 delivers the first end-to-end paper-mode orchestrator. Plan 5 owns live-data wiring, drift-monitor reference population, and the `LiveBroker` / `ReplayBroker` contract suite. Entry point for next iteration is `Orchestrator.boot()` → `build_scan_context()` — that's where the stub providers live.
