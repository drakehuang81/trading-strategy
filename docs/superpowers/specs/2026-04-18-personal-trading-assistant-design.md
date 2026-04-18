# Personal Trading Assistant — Design Spec

- **Date**: 2026-04-18
- **Status**: Draft (pending user review)
- **Supersedes**: existing `strategy/` + `auto_bot.py` design
- **Scope**: full architectural pivot from rule-based auto-bot to a local-LLM-augmented personal trading assistant
- **Reviewers consulted**: two independent subagents — senior quant risk engineer + senior software architect

---

## 1. Context

The existing project at `/Users/drakehuang/SideProject/Trading/quant-trading-project` is a rule-based crypto trading bot with 116 passing unit tests covering SMC (swing, BOS/CHoCH, order blocks, FVG, POI), Fibonacci, liquidity zones, RSI divergence, funding-rate filter, 8-factor confidence scoring, and trade-setup. Runs against Binance ETHUSDT 15m.

The user wants to pivot to a **personal virtual trading assistant** built around a local LLM. Goals: conversational analysis, signal push, semi-auto confirmation, full-auto with reporting — all four interaction modes sharing one brain.

## 2. Goals & Non-Goals

### Goals

1. **Local-first** — runs entirely on MacBook Pro M1 Pro 16GB with zero cloud spend.
2. **Multi-mode interface** — same brain supports: chat analyst, push notification on signal, half-auto (propose + user confirm), full-auto (with reports).
3. **Paper-safe today, live-ready tomorrow** — paper trading is the default mode; architecture supports flipping to real Binance execution via a single config flag without touching business logic.
4. **Hybrid intelligence** — LLM for analysis / explanation / interface; numeric ML model for predictions; rule-based risk and sizing.
5. **Auditable** — every proposal, prediction, and event is logged append-only for post-hoc study.
6. **Operable from phone** — Telegram is the sole user interface; SSH/terminal is only for deploy.

### Non-Goals

- Sub-minute / streaming tick trading. Cadence is **1h background macro scan + on-demand 15m deep scan**.
- Multi-exchange arbitrage execution (data interface reserved, execution deferred).
- On-chain analysis as a feature (interface stubbed, implementation deferred).
- Transformer-based price predictors (XGBoost is the primary ML; Transformer deferred).
- Web UI / dashboard (CLI = `python -m orchestrator` for ops only; no browser-facing UI).

### Constraints

- **Hardware**: M1 Pro 16GB unified memory; Apple Silicon optimizations welcomed (MLX).
- **Budget**: $0 recurring cloud spend.
- **Language/runtime**: Python **3.11** (upgrade from current 3.9 is a prerequisite).
- **Risk**: paper trading initially; three pre-live gates (§10) must pass before mode flip.

## 3. High-Level Architecture

Six vertical layers + cross-cutting services. Each layer owns one concern, communicates via Python `Protocol` interfaces, is independently testable, and can be replaced without cascade.

```
┌──────────────────────────────────────────────────────────────┐
│ 6. Interface        Telegram bot · periodic reports          │
│                     CLI entry (python -m orchestrator)       │
├──────────────────────────────────────────────────────────────┤
│ 5. Execution        Broker Protocol · PaperBroker ↔          │
│                     ReplayBroker ↔ LiveBroker (same contract)│
├──────────────────────────────────────────────────────────────┤
│ 4. Decision         Ensemble · Policy · RiskPipeline ·       │
│                     SizingPipeline → TradeProposal           │
├──────────────────────────────────────────────────────────────┤
│ 3. Model            ML Predictor (XGBoost) ·                 │
│                     LLMContextProvider (Gemma 4 E4B, structural    │
│                     flags only) · both implement Predictor   │
├──────────────────────────────────────────────────────────────┤
│ 2. Feature          SMC · Fib · Liquidity · Divergence ·     │
│                     Funding · Sentiment · Confidence         │
│                     (all with point-in-time compute)         │
├──────────────────────────────────────────────────────────────┤
│ 1. Data             DataSource Protocol · Binance ·          │
│                     (OnChain · News · MultiExchange — stubs) │
└──────────────────────────────────────────────────────────────┘

Cross-cutting:
  Orchestrator   single-process asyncio event loop
  Scheduler      apscheduler AsyncIOScheduler (1h macro)
  State          SQLite (structured, Alembic-migrated) + Parquet (candles/features)
  Config         YAML + .env; mode = paper | live
  Observability  structlog → SQLite sink; trace_id per flow
  LLM            ChatLLM (user-facing, READ_ONLY_TOOLS only)
```

### 3.1 Repository Layout

```
quant-trading-project/
├── config/
│   ├── settings.yaml
│   └── prompts/
├── src/
│   ├── data/        # Layer 1: DataSource Protocol + impls
│   ├── features/    # Layer 2: moved from strategy/
│   ├── models/
│   │   ├── ml/      # XGBoost
│   │   └── llm/     # LLMContextProvider (flags) + shared ollama client
│   ├── decision/    # Ensemble, Policy, Risk, Sizing, Proposal
│   ├── execution/   # Broker Protocol + Paper/Replay/Live
│   ├── interface/   # Telegram + ChatLLM
│   ├── state/       # SQLite + Parquet repos, Alembic
│   ├── observability/  # structlog config, trace_id context
│   └── orchestrator.py  # asyncio entry + scheduler + event consumer
├── scripts/         # backtest, fire-drill, pre-live-check, train
├── tests/
│   ├── unit/        # per-module (70%)
│   ├── contracts/   # per-Protocol shared suites (20%)
│   └── e2e/         # scenario tests (10%)
└── docs/
    └── superpowers/specs/
```

**Migration note**: existing `strategy/*.py` modules move wholesale to `src/features/` and are wrapped as `Feature` Protocol implementations. Existing 116 tests move to `tests/unit/features/` and remain green.

## 4. Component Contracts

### 4.1 Data Layer

```python
# src/data/base.py
from typing import Protocol
from datetime import datetime
import pandas as pd

class DataSource(Protocol):
    name: str

    async def fetch(
        self, symbol: str, timeframe: str,
        since: datetime, until: datetime,
    ) -> pd.DataFrame: ...

    async def fetch_latest(
        self, symbol: str, timeframe: str, n: int
    ) -> pd.DataFrame: ...

    def supports(self, symbol: str, timeframe: str) -> bool: ...
```

- **Day-1 implementation**: `BinanceKline` (wraps existing `data_ingestion/`).
- **Stubs (interface only, `NotImplementedError`)**: `OnChain`, `News`, `FundingRate`, `MultiExchange`.

### 4.2 Feature Layer

```python
# src/features/base.py
class Feature(Protocol):
    name: str
    required_lookback: int

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict:
        """MUST only use df[df.index <= as_of]. No repainting.
        Every implementation is covered by test_no_repainting (§9)."""
```

Registry enumerates features and composes feature vectors. Existing SMC / Fib / Liquidity / Divergence / Funding / Confidence are each refactored to implement `Feature`.

### 4.3 Model Layer

```python
# src/models/base.py
from pydantic import BaseModel
from typing import Literal, Protocol

class PredictionBundle(BaseModel):
    direction: Literal["long", "short", "flat"]
    prob_up: float                     # from ML only
    horizon_bars: int
    size_multiplier: float = 1.0       # set by Ensemble; 0.0 on LLM context_veto
    veto_reason: str | None = None
    feature_snapshot_hash: str
    predictions_detail: dict           # per-predictor raw outputs for audit

class Predictor(Protocol):
    async def predict(self, features: dict) -> PredictionBundle: ...
```

Two concrete components — one `Predictor`, one separate context provider:

- **`XGBPredictor`** (implements `Predictor`, primary signal) — calibrated direction classifier; emits `direction`, `prob_up`.
- **`LLMContextProvider`** (**not** a `Predictor`; distinct role) — emits `LLMContextFlags(context_veto: bool, veto_reason: str|None, structural_flags: list[str])`. Uses Gemma 4 E4B via Ollama with `instructor` + Pydantic v2, `temperature=0`, JSON schema enforced. Does **not** output `prob_up`.

`Ensemble` implements `Predictor` and combines the two:

```python
# src/decision/ensemble.py
class Ensemble(Predictor):
    def __init__(self, ml: Predictor, llm: LLMContextProvider): ...

    async def predict(self, features: dict) -> PredictionBundle:
        ml_pred = await self.ml.predict(features)
        flags = await self.llm.flags(features)
        if flags.context_veto:
            return ml_pred.model_copy(update={
                "size_multiplier": 0.0,
                "veto_reason": flags.veto_reason,
            })
        return ml_pred
```

**Decision Layer consumes one `Predictor`** — it does not know about `ml` or `llm` directly. This keeps `Policy` pure and predictor-agnostic.

### 4.4 Decision Layer

```python
# src/decision/proposal.py
class TradeProposal(BaseModel):
    proposal_id: str                 # uuid, append-only key
    ts: datetime
    trace_id: str
    symbol: str
    direction: Literal["long", "short"]
    entry: float
    stop_loss: float
    take_profit: list[float]
    size: float
    confidence: float
    feature_snapshot: dict
    bundle: PredictionBundle
    risk_checks: list["RiskCheckResult"]
    rationale: str | None = None     # filled later by ChatLLM

# src/decision/policy.py
class Policy(Protocol):
    async def propose(
        self, features: dict, bundle: PredictionBundle,
        portfolio: "PortfolioSnapshot",
    ) -> TradeProposal | None: ...

# src/decision/risk.py
class RiskCheck(Protocol):
    name: str
    def check(
        self, proposal: TradeProposal,
        portfolio: "PortfolioSnapshot",
    ) -> "RiskCheckResult": ...
```

#### Risk Pipeline (ordered, any fail → reject)

| Level | Check | Default | Notes |
|---|---|---|---|
| 1 Trade | `FixedFractionalSizer` | 0.25% equity | entry |
| 1 Trade | `MandatoryStopLoss` | required | missing SL = reject |
| 1 Trade | `FundingAdjustedExpectedReturn` | > 0 after funding | |
| 1 Trade | `SpreadGate` | spread ≤ 20 bps | low-liquidity guard |
| 2 Daily | `DailyLossKillSwitch` | -2R | also writes `./HALT` |
| 2 Daily | `MaxTradesPerDay` | 10 | |
| 3 Portfolio | `CorrelatedExposureLimiter` | BTC/ETH/alts grouped | |
| 3 Portfolio | `MaxConcurrentPositions` | 3 | |
| 3 Portfolio | `NetDirectionalCap` | **enabled=false (stub)** | activated pre-live |
| 4 System | `HaltFileGate` | checks `./HALT` each cycle | |
| 4 System | `HeartbeatMonitor` | stale > 5min → alert | |
| 4 System | `FeatureDriftMonitor` | PSI/KS vs training window | auto-disables ML on breach |

#### Sizing Pipeline (applies modifiers in order)

```python
class SizingModifier(Protocol):
    def apply(self, size: float, state: SessionState) -> float: ...

class SizingPipeline:
    modifiers: list[SizingModifier]

# Day-1 ships with [IdentityModifier()].
# DailyWinStreakTaper NOT shipped (no statistical basis for calibrated models).
# Interface exists so future experiments are plug-in.
```

### 4.5 Execution Layer

```python
# src/execution/base.py
class Order(BaseModel):
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    type: Literal["market", "limit"]
    qty: float
    price: float | None = None
    stop_loss: float | None = None
    take_profit: list[float] | None = None

OrderId = str

class BrokerEvent(BaseModel):
    event_id: str                    # unique; idempotent consumption
    kind: Literal["submitted", "partially_filled", "filled",
                  "rejected", "cancelled", "funding_charged"]
    order_id: OrderId
    ts: datetime
    fill_price: float | None = None
    fill_qty: float | None = None
    fee: float | None = None
    reason: str | None = None

class Broker(Protocol):
    async def submit(self, order: Order) -> OrderId: ...
    async def cancel(self, order_id: OrderId) -> None: ...
    def events(self) -> AsyncIterator[BrokerEvent]: ...
    async def positions(self) -> list[Position]: ...
    async def balance(self) -> Balance: ...
```

Three implementations:

- **`PaperBroker`** (Day 1) — simulates latency (default 200 ms + jitter), fee split (maker/taker), slippage as `f(spread, size/ADV)`, probabilistic partial fill, rejection, funding charged every 8 hours on open positions.
- **`ReplayBroker`** — reads recorded live tick JSONL; the ONLY credible pre-live test.
- **`LiveBroker(Binance)`** — class stub on Day 1; filled when pre-live gate (§10) is green.

### 4.6 Interface Layer

- **Telegram bot** (`python-telegram-bot` v20+, asyncio native). Commands: `/analyze <symbol>`, `/positions`, `/status`, `/halt`, `/resume`, `/accept_broker`, plus free-text → ChatLLM.
- **ChatLLM** — Gemma 4 E4B (same Ollama process as LLMContextProvider but **different Python class, different prompt, different schema**). Has `READ_ONLY_TOOLS` only: `get_positions`, `get_recent_proposals`, `get_pnl_summary`, `get_feature_snapshot`. Cannot submit orders — enforced by tool registry and covered by contract test.
- **CLI** — single entry `python -m orchestrator`. Not a user UI; ops-only.

### 4.7 Reconciliation

```python
# src/execution/reconcile.py
class ReconciliationPolicy(Protocol):
    async def on_diff(self, diff: ReconciliationDiff) -> ReconcileAction: ...

class PaperAutoRepair:
    """paper mode: trust broker, overwrite local, log, continue."""

class LiveConfirmViaTelegram:
    """live mode: push diff + inline buttons to Telegram.
       Buttons: /accept_broker, /halt. Timeout (default 600s) → HALT."""
```

Policy is swapped at the paper↔live boundary alongside `Broker`.

## 5. Data Flows

### 5.1 Scheduled 1h Macro Scan

Triggered by apscheduler every hour.

1. Orchestrator assigns `trace_id = uuid4()`, `mode=scheduled_macro`.
2. For each `symbol` in `config.watchlist`:
   1. `BinanceKline.fetch_latest(symbol, "1h", n=200)`.
   2. `FeatureRegistry.compute_all(df, as_of=ts)` → `features`.
   3. `bundle = await ensemble.predict(features)`.
   4. `proposal = await policy.propose(features, bundle, portfolio)`.
   5. If proposal: `RiskPipeline` evaluates → append to `proposals` table whether pass or reject.
   6. On pass: `rationale = await chat_llm.explain(proposal)`; Telegram push; `if mode==auto: broker.submit(...)`.
3. `heartbeat.mark(ts, trace_id)`.
4. Every step emits `structlog` with `trace_id` → `log` table.

### 5.2 On-Demand 15m Deep Scan

Triggered by Telegram `/analyze <symbol>` or routed free text.

Same shape as 5.1 but timeframe `15m`, `n=300`, and step 6 replaces Telegram push with streaming `ChatLLM.converse(...)` reply that can call the proposal context as a tool.

### 5.3 Pure Conversation

User question with no new analysis trigger (e.g., "What's my PnL this week?").

`ChatLLM.converse(history, tools=READ_ONLY_TOOLS)` — LLM calls repos via tool-use, composes reply, streams to Telegram. No order can result from this path — enforced by tool registry.

### 5.4 Order Execution (PaperBroker internals)

1. `submit(order)` returns `OrderId` immediately.
2. Internal `asyncio.create_task(simulate_fill(order))`.
3. `simulate_fill`: sleep(latency), decide fill ratio, compute slippage + fee, emit `BrokerEvent`.
4. Funding background task every 8h charges open positions; emits `funding_charged` events.
5. Event consumer task updates `positions_repo` + `fills_repo`, pushes confirmation to Telegram. All event handlers are idempotent on `event_id`.

### 5.5 HALT / Kill-Switch

Triggers: manual `touch ./HALT` · Telegram `/halt` · `DailyLossKillSwitch` · `FeatureDriftMonitor` · heartbeat stale > 5 min.

Effects: `HaltFileGate` rejects new orders; scheduler stops new `scheduled_macro`; event consumer continues so existing positions can be closed; Telegram alert with reason.

Recovery: `/resume` (Telegram) or remove `./HALT` file plus `/resume` confirmation. Pre-live gate (§10) requires the HALT mechanism to have fired at least once in paper mode.

## 6. Error Handling

### 6.1 Three Modes (strictly distinct)

| Mode | Use | Pattern |
|---|---|---|
| **Retry** | transient I/O (429, 5xx, timeouts) | exponential backoff + jitter, cap 3 attempts |
| **Degrade** | non-essential component failure | circuit breaker: disable N minutes, fall through (e.g., News down → skip sentiment feature; LLMContextProvider down → ML-only decision) |
| **Hard Stop** | anything affecting capital safety | write `./HALT`, Telegram alert, no retry |

### 6.2 Per-Layer Handling

| Layer | Common failures | Strategy |
|---|---|---|
| Data | rate limit, data gaps, clock skew (>2s) | Retry + cache fallback + drift alert |
| Feature | lookback short, NaN/Inf, schema drift | skip tick + counter; 3 consecutive → alert |
| ML Model | corrupt model file, feature drift | circuit-break → disable ML predictor |
| LLM | Ollama down / OOM / parse fail / timeout | `instructor` retry 2×; fall through (bundle still valid from ML) |
| Decision | conflicting predictions, schema miss | handled by `Ensemble` (§4.3); schema → reject |
| Execution | order no callback, position desync | timeout watchdog; startup reconciliation; dead-letter table |
| Interface | Telegram throttled / offline | buffered queue, resend; unknown free text → graceful "I'm not sure" |
| Cross | SQLite lock / disk full / deadlock | predictive alert; every task has timeout; deadlock watchdog |

## 7. Key Design Decisions (synthesized from both reviewers)

### 7.1 LLM Never Writes Order Numbers

LLM is split at **contract** level (not instance — same Ollama process is fine):

- **`LLMContextProvider` / `LLMContextProvider`** — outputs `LLMContextFlags(context_veto: bool, veto_reason: str|None, structural_flags: list[str])`. **Does not output `prob_up`.** Gemma's numeric probability is uncalibrated linguistic style; feeding it into a calibrated pipeline injects noise.
- **`ChatLLM`** — streaming, READ_ONLY_TOOLS, user-facing. Reads `TradeProposal` to explain; never originates one.

`Ensemble` (implements `Predictor`) sees `ml_pred` and `flags`, returns a single `PredictionBundle`. `Policy` is predictor-agnostic.

### 7.2 Paper Broker Has Real Friction From Day 1

Not "paper then add realism later." From the first commit `PaperBroker` simulates: async fills, latency, slippage = `f(spread, size/ADV)`, maker/taker fees, funding every 8h, probabilistic partial fill, rejection. The backtest cost model IS `PaperBroker`'s cost model — single source of truth.

### 7.3 Append-Only Event Log

`proposals` (including rejected), `broker_events`, `log`, `prediction_disagreements`, `reconciliation_diffs`, `heartbeat` are all append-only SQLite tables. `positions_repo` is a **derived snapshot** from `broker_events` and can be rebuilt.

### 7.4 HALT + Heartbeat + Fire Drill

HALT file + 5-min heartbeat + five trigger sources. Must fire in paper at least once (E2E scenario #2) before live mode is permitted.

### 7.5 Reconciliation: Policy-Swapped by Mode

`PaperAutoRepair` (paper) vs `LiveConfirmViaTelegram` (live). Live never silently overwrites local state; Telegram inline buttons with 10-min timeout → HALT.

### 7.6 Deferred With Stub Interface

| Item | Reason | Interface lands Day 1 |
|---|---|---|
| `NetDirectionalCap` | redundant at paper (covered by correlated limiter + concurrent cap); LUNA-cascade insurance at live | `RiskCheck.check(proposal, portfolio)` signature; `enabled=false` config stub |
| `OnChain` / `News` / `MultiExchange` DataSource | no use-case yet, adds complexity | `DataSource` interface only, `NotImplementedError` |
| `Transformer` price predictor | XGBoost + LLM already two models | `Predictor` Protocol ready |
| `DailyWinStreakTaper` | no statistical basis for calibrated models; tapering wins throws away EV | `SizingPipeline` ships with `[IdentityModifier()]` |

### 7.7 Tool-use: Hand-Roll with `instructor`

No LangChain / LlamaIndex. `instructor` patches the Ollama client to return validated Pydantic models with automatic retry-on-validation-fail. Tool dispatch is a ~30-line router over Pydantic tool schemas.

### 7.8 Concurrency: Single Process Asyncio

One event loop. apscheduler AsyncIOScheduler for 1h macro. Telegram polling task. Broker event consumer task. Ollama gated by `asyncio.Semaphore(1)` (16 GB RAM can't handle concurrent LLM calls). No threads, no multiprocess.

### 7.9 Python 3.9 → 3.11 Upgrade

**Prerequisite** before the first new line of code. Needed for: Pydantic v2 performance, `asyncio.TaskGroup`, `tomllib`, structural pattern matching in Policy, current MLX / Ollama client.

## 8. State Management

### 8.1 SQLite (structured, Alembic-migrated from Day 1)

Tables (append-only unless noted):

- `proposals` — every proposal, pass or reject, with feature snapshot + bundle.
- `broker_events` — source of truth for positions.
- `positions` — derived snapshot (rebuildable).
- `fills` — derived from broker_events for convenience.
- `prediction_disagreements` — for post-hoc ensemble tuning.
- `reconciliation_diffs` — every startup reconcile outcome.
- `conversations` / `messages` / `tool_calls` — ChatLLM memory, first-class.
- `heartbeat` — per-tick row.
- `halt_events` — every HALT activation with reason, trigger source, and resume timestamp.
- `log` — structlog sink; `trace_id`-indexed.
- `backtest_runs` — walk-forward results, Deflated Sharpe, cost model version.
- `session_state` — consecutive wins, day PnL for SizingPipeline.

### 8.2 Parquet (heavy data, Day 1)

- Candles per symbol × timeframe.
- Feature cache per (symbol, timeframe, feature, as_of).
- Training dataset snapshots (keyed by model version).

## 9. Testing Strategy

### 9.1 Pyramid (70 / 20 / 10)

- **70 % Unit** — per-module; existing 116 preserved + new.
- **20 % Contract** — one shared suite per Protocol, applied to every implementation (Broker / DataSource / Feature / Predictor / RiskCheck / ReconciliationPolicy).
- **10 % E2E** — scenario tests running `ReplayBroker` over recorded market data.

### 9.2 No-Repainting Test (mandatory per Feature)

```python
@pytest.mark.parametrize("feature_cls", ALL_FEATURES)
def test_no_repainting(feature_cls, eth_1h_df):
    feature = feature_cls()
    for ts in random.sample(list(eth_1h_df.index), 50):
        truncated = eth_1h_df[eth_1h_df.index <= ts]
        assert feature.compute(eth_1h_df, as_of=ts) == \
               feature.compute(truncated, as_of=ts)
```

Until this is green for all features, walk-forward backtest results are not trustworthy.

### 9.3 LLM Testing

1. **Snapshot prompt + schema, not output.** Prompt drift is visible; output drift is expected.
2. **Schema validation via `instructor` + Pydantic** — every LLM boundary is a typed contract.
3. **vcrpy-style record/replay** of Ollama responses — fixtures live in `tests/fixtures/ollama/`, go through version control. CI never touches real Ollama.
4. **ChatLLM boundary test** — assert `READ_ONLY_TOOLS` does not expose any order-producing method (regression guard against future mis-registration).

### 9.4 Contract Tests

One abstract `XxxContract` class per Protocol. Each implementation subclasses and provides a fixture. This is the mechanism that makes paper↔live safe.

### 9.5 E2E Scenarios (must all be green)

1. **Happy path** — 7 days ETH 1h/15m; reasonable proposal count, HALT not triggered.
2. **HALT fire drill** — inject losing sequence → `-2R`; assert kill-switch, HALT file written, new orders rejected, positions can close, Telegram alert.
3. **LLM down** — mock Ollama timeout; assert fall-through to ML-only, pipeline uninterrupted.
4. **Broker desync** — inject mismatched startup positions; paper auto-repairs, live pushes Telegram confirm.
5. **Feature drift** — inject synthetic OOD data; PSI breaches, ML predictor auto-disables, rules-only mode continues.

### 9.6 Backtest (strategy-level)

`scripts/backtest.py`:

- Walk-forward with purged CV (López de Prado embargo).
- **Deflated Sharpe** reported (raw Sharpe is inflated by multiple trials).
- **OOS holdout** — last N months never seen by LLM (no prompts, no examples).
- Cost model is `PaperBroker`'s (same fees/slippage/funding logic).
- Results appended to `backtest_runs`; never modifies code.

### 9.7 CI / Local Workflow

- **pre-commit**: unit + no-repainting + `mypy --strict` + `ruff` (~1 min).
- **GitHub Actions on push**: unit + contract + vcrpy LLM + 5 E2E scenarios.
- **Manual Make targets**: `make backtest`, `make fire-drill`, `make pre-live-check`.

## 10. Pre-Live Gate

Mode can be flipped from paper to live only when all three gates are green. Enforcement is code-level: `src/execution/pre_live_gate.py` runs at startup in live mode and refuses to proceed otherwise.

1. ✅ **All features pass `test_no_repainting`**.
2. ✅ **Walk-forward OOS Deflated Sharpe > 0.5** with realistic cost model (PaperBroker parity).
3. ✅ **HALT fire-drill has fired at least once in paper** (recorded in `halt_events` table via E2E scenario #2).

## 11. Non-Obvious Additions (from architect review)

1. **`TradeProposal` event log is also future training data.** Long-term accumulation lets you retrain ML on your own distribution.
2. **Observability minimalism** — `structlog` + SQLite sink + a Jupyter notebook. No Grafana, no OTEL (wrong scale). Query by `trace_id`.
3. **Heartbeat written every tick** — a separate lightweight monitor (cron or launchd) reads SQLite; if stale > 5 min, Telegram alert. This is independent of the main process so a crashed main can still be noticed.

## 12. Red Flags To Keep Watching (from quant review)

- **8-factor confidence score** — currently un-calibrated. Must calibrate (reliability diagram) or drop before trusting.
- **Single asset, single timeframe ML** — expect IC near zero after costs; don't overfit hope onto early backtest wins.
- **LLM sounding confident ≠ LLM being right** — ChatLLM rationale is presentation, not evidence.

## 13. Migration Plan (from current codebase)

1. **Bump Python 3.9 → 3.11**; regenerate venv; confirm 116 tests still pass.
2. **Create `src/` skeleton** with empty layer directories and Protocols.
3. **Move `strategy/*.py` → `src/features/`** and wrap as `Feature` implementations; keep tests green.
4. **Add `no_repainting` test** for every migrated feature; fix any that fail (expected: SMC swing/BOS will need `as_of` clipping).
5. **Introduce SQLite + Alembic baseline migration**.
6. **Wire scaffolding end-to-end** — thin `BinanceKline` → one `Feature` → `XGBPredictor` stub → trivial `Policy` → `PaperBroker` → CLI log line. First working pipeline.
7. **Telegram bot + ChatLLM** — minimal `/status` and free-text chat.
8. **Ensemble with real XGBoost**, trained from historical features.
9. **Add `LLMContextProvider` / `LLMContextProvider` with `instructor`**; wire into Ensemble.
10. **Full risk pipeline + sizing pipeline**.
11. **Five E2E scenarios green**.
12. **Walk-forward backtest infrastructure + first Deflated Sharpe report**.
13. **Pre-live gate module** and HALT fire-drill.

Detailed implementation plan to follow via `superpowers:writing-plans`.

## 14. Open Questions (for future revision)

- Which calibration method for XGBoost `prob_up` (isotonic vs Platt)?
- ChatLLM conversation memory eviction policy when SQLite grows.
- Specific PSI/KS thresholds for `FeatureDriftMonitor` — TBD after first backtest.
- When `NetDirectionalCap` activates for live, what defaults are sensible for single-account crypto?

## 15. References

- Subagent review 1: senior quant risk engineer (architecture + 4-question follow-up)
- Subagent review 2: senior software architect (architecture + 4-question follow-up)
- Existing code: `/Users/drakehuang/SideProject/Trading/quant-trading-project`
- Gemma 4 — Google DeepMind release 2026-04-02, Apache 2.0, E4B 4B params multimodal 128K context
