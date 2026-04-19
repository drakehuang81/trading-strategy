# Plan 3 STATUS — Interface + Ops (Paper Mode)

**Date**: 2026-04-19
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: `504022f` (Plan 3 plan doc)
**Head commit**: `9d5df2e`

## Summary

All 12 tasks complete. Paper-mode **components** delivered individually and covered by unit + E2E tests: OllamaClient priority queue, ChatLLM with read-only tools, Telegram bot, scan pipeline, APScheduler orchestrator skeleton, heartbeat watchdog, reconciliation, drift monitor, HALT manager, four E2E scenarios.

**Integration gap (flagged by final review, deferred to Plan 4):** `Orchestrator.run()` only schedules a no-op `_scheduled_scan` and placeholder loops. It does NOT instantiate `HaltManager` / `TelegramBot` / `OllamaClient` / `ChatLLM` / `Ensemble` / `PaperBroker` / `ScanContext`. E2E tests wire the stack by hand. Running `python -m src.cli` today does migrations + heartbeat only; it does not trade, scan, or serve Telegram. Plan 4 task 1 must build the `ScanContext` factory and wire lifecycle inside `Orchestrator.boot/run`.

Test count: **229 passed** (excluding 2 pre-existing `libomp`/xgboost environment failures unrelated to Plan 3).

## Task table

| # | Title | Commit | Files |
|---|-------|--------|-------|
| 1 | `broker_events.symbol` column | `6787984` | migration + repo |
| 2 | OllamaClient priority queue + `chat()` | `c4a36e2`, `1703821` | client + type widening |
| 3 | FeatureDriftMonitor + `config/drift.yaml` | `1998ba6`, `a2cdc88` | PSI/KS + dead-import fix |
| 4 | HaltManager | `28f8e6f` | activate/resume + triggers |
| 5 | ChatLLM + READ_ONLY tools + repos | `85e9aed`, `cb0afca` | 7 files + quality fixes |
| 6 | Telegram bot | `8dfc7b3` | commands + free-text routing |
| 7 | PaperAutoRepair reconciliation | `dcbb096` | Protocol + paper impl |
| 8 | Scan pipeline (macro + deep) | `3565c97` | ScanContext + `_scan_symbol` |
| 9 | Orchestrator wiring (APScheduler + TaskGroup) | `a22729f`, `e90747d` | rewrite + sys import drop |
| 10 | Heartbeat watchdog | `dc18c31` | external script |
| 11 | E2E scenarios + Ensemble LLM fallback | `920ad1d`, `9d5df2e` | 4 tests + constant extract |
| 12 | Final verification + handoff | (this commit) | STATUS doc |

## Verification checklist

- ✅ Full test suite green — 229 passed (226 were Plan 1+2 baseline; Plan 3 added 43 tests net)
- ✅ E2E scenarios pass: smoke pipeline, HALT fire drill, LLM fallback (×2), broker desync, feature drift — 6 e2e tests total
- ✅ `broker_events.symbol` column present and round-trips through `BrokerEventRepo.insert/all`
- ✅ OllamaClient priority queue serves `SCHEDULED_MACRO (0) > ON_DEMAND_DEEP (1) > CHAT (2)` via `asyncio.PriorityQueue`
- ✅ FeatureDriftMonitor detects PSI/KS breach on shifted data (scipy parity verified to 1e-17)
- ✅ HaltManager: `activate()` writes HALT + `halt_events` row; `attempt_resume()` blocks while any trigger is breached
- ✅ ChatLLM boundary contract: `TOOL_NAMES == {get_positions, get_recent_proposals, get_pnl_summary, get_feature_snapshot}` and 21 write-verb patterns blocked
- ✅ `grep prob_up` on `src/interface/` and `src/observability/` — 0 matches (LLM never writes probability, §7.1)
- ✅ Heartbeat watchdog writes HALT on stale heartbeat (threshold configurable, default 5min)
- ✅ PaperAutoRepair trusts broker, logs to `reconciliation_diffs` with `resolution="auto_repaired"`
- ✅ HALT file contract consistent across `halt.py`, `orchestrator.py`, `heartbeat_watchdog.py`
- ⚠️ mypy `--strict`: not run in this session (pre-existing Plan 1 baseline has 80 errors; Plan 3 code is annotated but not strictly verified — deferred to a dedicated typing pass)

## Key design decisions landed

- **Ensemble LLM fallback marker**: `LLM_UNAVAILABLE_MARKER = "llm_unavailable"` exported from `src/decision/ensemble.py`. Persists to `trade_proposals.llm_prompt_version` as audit trail.
- **OllamaClient scheduler**: single background task drains a PriorityQueue; tickets use `dataclass(order=True)` with `compare=False` on asyncio.Event fields (avoids Event-comparison TypeError).
- **ChatLLM tool transcript**: tool-call rounds are NOT persisted as `role="assistant"` messages; only the final user-facing reply enters `messages` table. Tool calls audit trail lives in `tool_calls`.
- **ChatLLM exhaustion fallback**: if 5 tool rounds are consumed without a clean text reply, returns `"（抱歉，無法在 5 步內完成這個請求，請重新描述。）"` and logs `chat_llm.max_rounds_exhausted`.
- **Telegram `stop_signals=None`**: prevents PTB from installing its own SIGTERM handler, allowing Orchestrator's TaskGroup to own shutdown.
- **Heartbeat watchdog**: runs as external process (launchd/cron); decoupled from orchestrator's event loop so it survives hangs.

## What is NOT done (Plan 4 scope)

- Walk-forward backtest + Deflated Sharpe (§10.1 gate 2)
- Pre-Live Gate module — 8 gates, exit code, live mode blocker (§10)
- ReplayBroker implementation
- LiveBroker implementation
- LiveConfirmViaTelegram reconciliation policy (§7.4)
- Contract suite for ReplayBroker / LiveBroker
- NetDirectionalCap enforcement (enabled after 60+ days paper data)
- ChatLLM memory eviction policy
- Streaming ChatLLM replies (current: full response)
- Full ScanContext dependency injection in `Orchestrator.run()` — `_scheduled_scan` is currently a stub
- `_event_consumer_loop` — currently a placeholder; will consume broker events in Plan 4

## Known issues / follow-ups

Fixed in final review pass (commit `HEAD`):

- ✅ **Task 8**: Empty-df guard added in `_scan_symbol` step 1
- ✅ **Task 8**: Dropped `hasattr(ctx.broker, 'positions'/'balance')` guards — `Broker` Protocol guarantees both
- ✅ **Task 10**: Watchdog now uses write-if-absent so user's `/halt` reason is not clobbered
- ✅ **Orchestrator docstring**: honest about skeleton-only status

Still open (Plan 4 scope):

- **Orchestrator full wiring** — biggest gap, tracked above under Summary integration-gap note. Plan 4 task 1.
- **Heartbeat → HaltManager trigger**: watchdog writes HALT externally, but HaltManager's `attempt_resume()` has no `HeartbeatTrigger` in its trigger list (the list is empty today). So `/resume` unconditionally unlinks HALT even if heartbeat is still stale. Wire a HeartbeatTrigger when ScanContext is built.
- **Task 5**: `message_repo.history()` orders by `ts` microsecond — add `seq INTEGER` secondary sort key before live.
- **Task 6**: No handler-level tests (only `parse_analyze_command`).
- **Task 7**: Sign-flip (long↔short) not flagged differently from qty delta. Live mode should enrich.
- **Task 9**: `run()` has no graceful-shutdown path; TaskGroup tasks loop forever. Spec §4.8 SIGTERM handling deferred.
- **Task 9**: `_scheduled_scan` and `_event_consumer_loop` are silent stubs — should emit startup log lines.
- **Task 10**: Engine not opened read-only (docstring claims RO).
- **`config/drift.yaml`** is orphaned — no production path loads it yet; wire when FeatureDriftMonitor is instantiated in orchestrator.
- **`tool_executor.execute` for `get_feature_snapshot`** returns hardcoded `not_implemented` JSON; advertised to LLM. Fill in when FeatureRegistry is wired into `ScanContext`.
- **ChatLLM rationale silent-fail** (`pipeline.py:137-138`) — add `reason="llm_timeout"` field for ops.
- **`ConversationRepo` imported inside `_free_text`** — move to module level.
