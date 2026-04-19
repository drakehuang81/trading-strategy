# Plan 2 STATUS — Model + Decision + End-to-End Scaffold

**Branch:** `pivot/foundation`
**Date completed:** 2026-04-19
**Test count:** 154 (Plan 1 end) → 186 (Plan 2 end), +32 tests
**All 19 tasks: DONE**

## Task Summary

| # | Task | Commit | Status |
|---|------|--------|--------|
| 1 | Decision + Execution Protocols | `3a77aa4` | ✅ |
| 2 | SQLite repos + rebuild_positions | `cef7aed` | ✅ |
| 3 | PaperBroker core | `c430bf6` | ✅ |
| 4 | PaperBroker funding | (in `c430bf6`) | ✅ |
| 5 | Broker contract test | `6a11bca` | ✅ |
| 6 | BinanceKline DataSource | `f311971` | ✅ |
| 7 | FundingRateWriter | `19055a1` | ✅ |
| 8 | TickRecorder | `2867a40` | ✅ |
| 9 | RiskPipeline + Day-1 checks | `3cd561c` | ✅ |
| 10 | SizingPipeline | `c4aa24d` | ✅ |
| 11 | Re-home trade_setup | `3d19dbb` | ✅ |
| 12 | ThresholdPolicy MVP | `88340db` | ✅ |
| 13 | XGBPredictor stub | `a9521c5` | ✅ |
| 14 | Training script + isotonic | `1ca5efd` | ✅ |
| 15 | GemmaContextProvider | `6c37482` | ✅ |
| 16 | Ensemble (ML + LLM flags) | `3bb4f05` | ✅ |
| 17 | Orchestrator + CLI | `99f8762` | ✅ |
| 18 | E2E smoke pipeline | `6e105ac` | ✅ |
| 19 | Final verification + handoff | (this commit) | ✅ |

## Files Added (Plan 2)

### src/
- `src/execution/base.py` — Order, BrokerEvent, Position, Balance, Broker Protocol
- `src/execution/paper_broker.py` — PaperBroker with latency, slippage, fees, partial fill, funding
- `src/execution/replay.py` — rebuild_positions (pure, idempotent on event_id)
- `src/execution/repositories.py` — BrokerEventRepo, ProposalRepo, SessionStateRepo
- `src/execution/tick_recorder.py` — TickRecorder → JSONL
- `src/decision/proposal.py` — TradeProposal, RiskCheckResult, PortfolioSnapshot
- `src/decision/policy.py` — Policy Protocol + ThresholdPolicy
- `src/decision/risk/base.py` — RiskCheck Protocol
- `src/decision/risk/pipeline.py` — RiskPipeline (short-circuit)
- `src/decision/risk/checks.py` — MandatorySL, SpreadGate, DailyLossKillSwitch, MaxConcurrentPositions
- `src/decision/sizing.py` — FixedFractionalSizer + SizingPipeline
- `src/decision/ensemble.py` — Ensemble (ML prob + LLM flags, never mix)
- `src/decision/trade_setup.py` — Re-homed from _legacy/
- `src/models/xgb_predictor.py` — XGBPredictor with stub + load + isotonic calibration
- `src/models/llm/ollama_client.py` — Minimal Ollama wrapper (instructor)
- `src/models/llm/gemma_context.py` — GemmaContextProvider (flags only)
- `src/data/binance_kline.py` — BinanceKline DataSource
- `src/data/funding.py` — FundingRateWriter + load_funding
- `src/orchestrator.py` — TaskGroup boot sequence (HALT check → migrations → heartbeat)
- `src/cli.py` — Ops-only CLI entry point

### tests/
- `tests/unit/execution/test_base_types.py`
- `tests/unit/execution/test_replay.py`
- `tests/unit/execution/test_repositories.py`
- `tests/unit/execution/test_paper_broker.py`
- `tests/unit/execution/test_tick_recorder.py`
- `tests/unit/decision/test_proposal.py`
- `tests/unit/decision/test_policy.py`
- `tests/unit/decision/test_risk_pipeline.py`
- `tests/unit/decision/test_sizing.py`
- `tests/unit/decision/test_ensemble.py`
- `tests/unit/decision/test_trade_setup.py`
- `tests/unit/data/test_binance_kline.py`
- `tests/unit/data/test_funding.py`
- `tests/unit/models/test_xgb_predictor.py`
- `tests/unit/models/test_xgb_predictor_real.py` (marked @slow)
- `tests/unit/models/test_gemma_context.py`
- `tests/unit/test_orchestrator.py`
- `tests/contracts/test_broker_contract.py`
- `tests/contracts/test_prompt_versioning.py`
- `tests/e2e/test_smoke_pipeline.py`

### Other
- `config/prompts/context_provider.md` — Gemma prompt text
- `scripts/train_xgb.py` — XGBoost training + isotonic calibration
- `models/.gitkeep` — Trained artifact directory (gitignored)

## Verification Checklist

- [x] Full test suite green: 186 passed, 0 failed
- [x] E2E smoke pipeline passes without skips
- [x] `broker_events.event_id` sole PK (spec §8.3)
- [x] `rebuild_positions` idempotent on duplicate event_ids
- [x] `GemmaContextProvider.prompt_version == sha256(prompt file)` — contract test
- [x] No LLM path produces `prob_up` (grep clean)
- [x] Model artifacts not committed to git
- [x] Plan-1 invariants hold: 6 features stable order, no-repainting tests green
- [x] No `decision._legacy` references in source
- [x] mypy: 0 new errors in Plan-2 code (80 pre-existing from Plan-1 features, Plan-3 scope)

## What is NOT done (Plan 3 scope)

- APScheduler hourly job wiring in Orchestrator
- Telegram bot integration
- ChatLLM conversational interface
- FeatureDriftMonitor
- Contract suite for ReplayBroker / LiveBroker
- Walk-forward backtest + Deflated Sharpe
- Pre-Live Gate module (spec §10.1)
- HALT fire-drill automation
- Heartbeat watchdog (currently just inserts; no alerting)
- `broker_events.symbol` column in schema (BrokerEvent model has it, repo doesn't persist it yet)
