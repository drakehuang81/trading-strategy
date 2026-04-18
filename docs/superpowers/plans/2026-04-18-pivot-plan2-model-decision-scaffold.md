# Pivot Plan 2 — Model + Decision + End-to-End Scaffold

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the first end-to-end scaffold from `BinanceKline` → `FeatureRegistry` → `XGBPredictor + Ensemble(+Gemma)` → `Policy` → `RiskPipeline` → `PaperBroker` → SQLite, with `TickRecorder` running from Day 1.

**Architecture:** Single asyncio process (Python 3.11 `TaskGroup`). All protocols from spec §4.4/§4.5/§4.3 land in code; `PaperBroker` ships with real friction; `rebuild_positions` is pure + idempotent on `event_id`. `XGBPredictor` uses isotonic calibration (resolves spec Q1 blocker). `GemmaContextProvider` emits boolean flags only — never a probability.

**Tech Stack:** Python 3.11, Pydantic v2, SQLAlchemy Core, python-binance AsyncClient, xgboost, scikit-learn (isotonic), ollama (python client), instructor, apscheduler, structlog, pytest, pytest-asyncio, mypy strict.

**Phase:** 2 of 3. Depends on Plan 1 (foundation). Next: Plan 3 (Interface + Ops + Pre-Live Gate) covers Telegram/ChatLLM + FeatureDriftMonitor + backtest/Deflated-Sharpe + Pre-Live Gate + HALT fire-drill + heartbeat watchdog.

**Spec:** [2026-04-18-personal-trading-assistant-design.md](../specs/2026-04-18-personal-trading-assistant-design.md) — authoritative. Re-read §4.3 (Model), §4.4 (Decision), §4.5 (Execution), §4.9 (TickRecorder), §7.1–§7.9 (Design decisions), §8.3 (Idempotency), §9.3 (LLM testing) before starting.

**Prerequisite:** Plan 1 completed and merged; `pivot/foundation` tests green (154 passed), `alembic current = 15fdbaffd2bf`.

---

## File Structure

New files under `src/` created by this plan:

```
src/
├── data/
│   ├── binance_kline.py           # Task 6  — DataSource impl
│   ├── funding.py                 # Task 7  — FundingRateDataSource + writer
├── decision/
│   ├── proposal.py                # Task 1  — TradeProposal, RiskCheckResult dataclasses
│   ├── policy.py                  # Task 1+12 — Policy Protocol + ThresholdPolicy
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── base.py                # Task 1  — RiskCheck Protocol
│   │   ├── pipeline.py            # Task 9  — RiskPipeline
│   │   ├── checks.py              # Task 9  — MandatorySL, SpreadGate, DailyLossKill, MaxConcurrent
│   ├── sizing.py                  # Task 10 — SizingModifier + SizingPipeline
│   ├── trade_setup.py             # Task 11 — re-homed from _legacy (Plan 1)
│   ├── ensemble.py                # Task 16 — ML + LLM flags → PredictionBundle
├── execution/
│   ├── base.py                    # Task 1  — Order, BrokerEvent, Broker, BrokerEventStream
│   ├── paper_broker.py            # Task 3+4 — PaperBroker with friction + funding
│   ├── replay.py                  # Task 2  — rebuild_positions pure fn
│   ├── repositories.py            # Task 2  — BrokerEventRepo, ProposalRepo, PositionRepo, SessionStateRepo
│   ├── tick_recorder.py           # Task 8  — WS ticks → daily JSONL
├── models/
│   ├── xgb_predictor.py           # Task 13+14 — XGBPredictor stub → calibrated
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama_client.py       # Task 15 — shared Ollama client (priority queue comes in Plan 3)
│   │   └── gemma_context.py       # Task 15 — GemmaContextProvider
├── orchestrator.py                # Task 17 — asyncio TaskGroup main
├── cli.py                         # Task 17 — `python -m src.cli`

config/
├── prompts/
│   └── context_provider.md        # Task 15 — Gemma prompt text

scripts/
├── train_xgb.py                   # Task 14 — training + isotonic calibration + model_versions row

tests/
├── contracts/
│   ├── test_broker_contract.py    # Task 5  — shared Broker contract
│   └── test_prompt_versioning.py  # Task 15 — prompt_version = sha256(file)
├── unit/
│   ├── data/
│   │   ├── test_binance_kline.py
│   │   └── test_funding.py
│   ├── decision/
│   │   ├── test_proposal.py
│   │   ├── test_policy.py
│   │   ├── test_risk_pipeline.py
│   │   ├── test_sizing.py
│   │   ├── test_trade_setup.py         # moved from unit/ in Task 11
│   │   └── test_ensemble.py
│   ├── execution/
│   │   ├── test_paper_broker.py
│   │   ├── test_replay.py
│   │   ├── test_repositories.py
│   │   └── test_tick_recorder.py
│   ├── models/
│   │   ├── test_xgb_predictor.py
│   │   ├── test_ollama_client.py
│   │   └── test_gemma_context.py
├── e2e/
│   └── test_smoke_pipeline.py     # Task 18 — full scaffold dry run
```

New dependencies added in Task 2 (requirements.txt):
- `xgboost>=2.0`, `scikit-learn>=1.4` (isotonic), `ollama>=0.2`, `instructor>=1.3`, `apscheduler>=3.10`, `aiosqlite>=0.19`

---

## Task 1: Decision + Execution Protocols

**Files:**
- Create: `src/execution/base.py`, `src/decision/proposal.py`, `src/decision/policy.py`, `src/decision/risk/__init__.py`, `src/decision/risk/base.py`
- Test: `tests/unit/decision/test_proposal.py`, `tests/unit/execution/test_base_types.py`

Rationale: Spec §4.4 + §4.5. These Protocols were explicitly deferred from Plan 1 Task 5. Everything downstream in this plan consumes them.

- [ ] **Step 1: Write failing tests for Order, BrokerEvent, TradeProposal shape**

```python
# tests/unit/execution/test_base_types.py
import pytest
from execution.base import Order, BrokerEvent

def test_order_requires_client_order_id_and_qty():
    with pytest.raises(Exception):
        Order(symbol="ETHUSDT", side="buy", type="market", qty=1.0)  # missing client_order_id
    o = Order(client_order_id="c1", symbol="ETHUSDT", side="buy", type="market", qty=1.0)
    assert o.side == "buy"

def test_broker_event_kind_enum():
    BrokerEvent(event_id="e1", kind="filled", order_id="o1", ts_epoch_ms=1, fill_price=1.0, fill_qty=1.0, fee=0.0)
    with pytest.raises(Exception):
        BrokerEvent(event_id="e1", kind="bogus", order_id="o1", ts_epoch_ms=1)
```

```python
# tests/unit/decision/test_proposal.py
from datetime import datetime, timezone
from decision.proposal import TradeProposal

def test_trade_proposal_round_trip():
    p = TradeProposal(
        proposal_id="p1", trace_id="t1",
        ts=datetime(2026, 4, 18, tzinfo=timezone.utc),
        symbol="ETHUSDT", direction="long",
        entry=2000.0, stop_loss=1980.0, take_profit=[2020.0, 2040.0],
        size=0.1, confidence=0.65,
        feature_snapshot={"smc": {}},
        bundle_json="{}",
        risk_checks=[],
        feature_registry_version="1.0.0",
        ml_model_version="stub",
        llm_prompt_version="stub",
    )
    assert p.model_dump()["direction"] == "long"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/drakehuang/SideProject/Trading/quant-trading-project/.worktrees/pivot-foundation
source venv/bin/activate
pytest tests/unit/execution/test_base_types.py tests/unit/decision/test_proposal.py -v
```
Expected: ImportError / ModuleNotFoundError.

- [ ] **Step 3: Implement `src/execution/base.py`**

```python
"""Execution layer Protocols and Pydantic payloads — spec §4.5."""
from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator, Literal, Protocol

from pydantic import BaseModel, Field


OrderId = str


class Order(BaseModel):
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    type: Literal["market", "limit"]
    qty: float = Field(gt=0)
    price: float | None = None
    stop_loss: float | None = None
    take_profit: list[float] | None = None


class BrokerEvent(BaseModel):
    event_id: str                    # spec §8.3 idempotency key
    kind: Literal[
        "submitted", "partially_filled", "filled",
        "rejected", "cancelled", "funding_charged",
    ]
    order_id: OrderId
    ts_epoch_ms: int
    fill_price: float | None = None
    fill_qty: float | None = None
    fee: float | None = None
    reason: str | None = None
    ml_model_version: str | None = None
    llm_prompt_version: str | None = None


class Position(BaseModel):
    symbol: str
    qty: float                       # signed: + long / - short
    avg_entry: float
    opened_at: datetime
    last_update_ts: datetime


class Balance(BaseModel):
    equity_usdt: float
    free_usdt: float


class Broker(Protocol):
    async def submit(self, order: Order) -> OrderId: ...
    async def cancel(self, order_id: OrderId) -> None: ...
    async def positions(self) -> list[Position]: ...
    async def balance(self) -> Balance: ...


class BrokerEventStream(Protocol):
    def events(self) -> AsyncIterator[BrokerEvent]: ...
```

- [ ] **Step 4: Implement `src/decision/proposal.py`**

```python
"""Decision layer payloads — spec §4.4."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class RiskCheckResult(BaseModel):
    name: str
    passed: bool
    detail: str | None = None


class TradeProposal(BaseModel):
    proposal_id: str
    trace_id: str
    ts: datetime
    symbol: str
    direction: Literal["long", "short"]
    entry: float
    stop_loss: float
    take_profit: list[float]
    size: float
    confidence: float
    feature_snapshot: dict[str, Any]
    bundle_json: str                 # serialized PredictionBundle — JSON
    risk_checks: list[RiskCheckResult]
    rationale: str | None = None
    feature_registry_version: str
    ml_model_version: str
    llm_prompt_version: str


class PortfolioSnapshot(BaseModel):
    equity_usdt: float
    open_positions: dict[str, float]  # symbol → signed qty
    day_pnl_r: float
    consecutive_wins: int
```

- [ ] **Step 5: Implement Policy + RiskCheck Protocols**

```python
# src/decision/policy.py
"""Policy Protocol — spec §4.4."""
from __future__ import annotations

from typing import Any, Protocol

from decision.proposal import PortfolioSnapshot, TradeProposal
from models.base import PredictionBundle


class Policy(Protocol):
    async def propose(
        self,
        features: dict[str, Any],
        bundle: PredictionBundle,
        portfolio: PortfolioSnapshot,
    ) -> TradeProposal | None: ...
```

```python
# src/decision/risk/base.py
"""RiskCheck Protocol — spec §4.4."""
from __future__ import annotations

from typing import Protocol

from decision.proposal import PortfolioSnapshot, RiskCheckResult, TradeProposal


class RiskCheck(Protocol):
    name: str

    def check(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioSnapshot,
    ) -> RiskCheckResult: ...
```

```python
# src/decision/risk/__init__.py
from decision.risk.base import RiskCheck

__all__ = ["RiskCheck"]
```

- [ ] **Step 6: Run tests — verify pass**

Run: `pytest tests/unit/execution/test_base_types.py tests/unit/decision/test_proposal.py -v`
Expected: all pass.

- [ ] **Step 7: mypy check**

Run: `venv/bin/python -m mypy src/execution/base.py src/decision/proposal.py src/decision/policy.py src/decision/risk/`
Expected: no new errors.

- [ ] **Step 8: Commit**

```bash
git add src/execution/base.py src/decision/proposal.py src/decision/policy.py src/decision/risk/
git add tests/unit/decision/test_proposal.py tests/unit/execution/test_base_types.py
git commit -m "feat: Decision/Execution Protocols (TradeProposal, Broker, Policy, RiskCheck)"
```

---

## Task 2: SQLite repositories + rebuild_positions

**Files:**
- Create: `src/execution/repositories.py`, `src/execution/replay.py`
- Test: `tests/unit/execution/test_repositories.py`, `tests/unit/execution/test_replay.py`
- Modify: `requirements.txt` (add xgboost, scikit-learn, ollama, instructor, apscheduler, aiosqlite)

Rationale: Spec §8.3. `rebuild_positions` is the idempotency oracle. Every other module will persist through these repos.

- [ ] **Step 1: Update `requirements.txt`**

```
# Plan 2 additions
xgboost>=2.0
scikit-learn>=1.4
ollama>=0.2
instructor>=1.3
apscheduler>=3.10
aiosqlite>=0.19
```

- [ ] **Step 2: `pip install`**

```bash
source venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 3: Write failing test for rebuild_positions idempotency**

```python
# tests/unit/execution/test_replay.py
from execution.base import BrokerEvent
from execution.replay import rebuild_positions


def ev(event_id: str, order_id: str, kind: str, symbol: str = "ETHUSDT",
       price: float = 2000, qty: float = 0.1, side: str = "buy", fee: float = 0.0) -> BrokerEvent:
    # Encode side via signed qty on filled events — rebuild_positions uses that convention.
    signed_qty = qty if side == "buy" else -qty
    return BrokerEvent(
        event_id=event_id, kind=kind, order_id=order_id,
        ts_epoch_ms=0, fill_price=price, fill_qty=signed_qty, fee=fee,
    )


def test_rebuild_positions_basic():
    events = [
        ev("e1", "o1", "filled", qty=0.1, side="buy", price=2000),
        ev("e2", "o1", "filled", qty=0.1, side="buy", price=2010),   # additional partial
        ev("e3", "o2", "filled", qty=0.05, side="sell", price=2050),
    ]
    snap = rebuild_positions(events)
    eth = snap["ETHUSDT"]
    assert abs(eth.qty - 0.15) < 1e-9
    # VWAP of the two buys, unaffected by the later sell
    assert abs(eth.avg_entry - (2000 * 0.1 + 2010 * 0.1) / 0.2) < 1e-6


def test_rebuild_positions_idempotent_on_duplicates():
    events = [
        ev("e1", "o1", "filled", qty=0.1, side="buy", price=2000),
        ev("e2", "o1", "filled", qty=0.1, side="buy", price=2010),
    ]
    snap_once = rebuild_positions(events)
    snap_twice = rebuild_positions(events + events)       # duplicates: same event_id
    assert snap_once == snap_twice
```

- [ ] **Step 4: Run test — verify fails**

Run: `pytest tests/unit/execution/test_replay.py -v`
Expected: ImportError.

- [ ] **Step 5: Implement `src/execution/replay.py`**

```python
"""Pure positions-rebuild function — spec §8.3 idempotency contract.

Source of truth for positions is broker_events; the positions table is
a cache. rebuild_positions is the oracle that defines what 'correct'
means.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from execution.base import BrokerEvent


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    qty: float
    avg_entry: float
    last_update_ts_ms: int


def rebuild_positions(events: Iterable[BrokerEvent]) -> dict[str, PositionSnapshot]:
    """Deterministic snapshot from a stream of BrokerEvents.

    Idempotent: duplicate event_ids are skipped (replay-safe).
    Only `filled` and `partially_filled` kinds affect position state.
    `fill_qty` is signed: buy=+qty, sell=-qty (caller encodes this).
    """
    seen: set[str] = set()
    state: dict[str, dict] = {}    # symbol -> dict(qty, cost_basis, last_ts_ms)

    for e in events:
        if e.event_id in seen:
            continue
        seen.add(e.event_id)
        if e.kind not in ("filled", "partially_filled"):
            continue
        if e.fill_price is None or e.fill_qty is None:
            continue
        # Symbol lives on Order, but for this unit Oracle we derive it later;
        # tests currently put symbol in a side-channel. We use a convention:
        # order_id is assumed to be distinct per symbol in tests, and the repo
        # layer stores symbol explicitly. For a pure function we need the symbol
        # passed through the event. Extend BrokerEvent with a symbol echo.
        raise NotImplementedError("see Step 6 — BrokerEvent.symbol echo needed")

    return {}
```

The above intentionally stops at `NotImplementedError` — Step 6 corrects the design.

- [ ] **Step 6: Extend `BrokerEvent` with `symbol` echo (append-only; no DB change yet)**

Edit `src/execution/base.py`: add `symbol: str | None = None` to `BrokerEvent` (Optional to keep migration strict-compat, but `PaperBroker` will always populate it). Then update Step 3's test helper to pass `symbol="ETHUSDT"`, and complete Step 5:

```python
# src/execution/replay.py  (final)
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from execution.base import BrokerEvent


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    qty: float
    avg_entry: float
    last_update_ts_ms: int


def rebuild_positions(events: Iterable[BrokerEvent]) -> dict[str, PositionSnapshot]:
    seen: set[str] = set()
    agg: dict[str, list[float]] = {}   # symbol -> [qty, cost_basis, last_ts]

    for e in events:
        if e.event_id in seen:
            continue
        seen.add(e.event_id)
        if e.kind not in ("filled", "partially_filled"):
            continue
        if e.fill_price is None or e.fill_qty is None or e.symbol is None:
            continue
        sym = e.symbol
        cur = agg.setdefault(sym, [0.0, 0.0, 0])
        qty, cost, _ = cur
        new_qty = qty + e.fill_qty
        # Only accumulate cost basis on adds in the same direction;
        # partial closes reduce qty but leave avg_entry anchored on remaining.
        if qty == 0 or (qty > 0) == (e.fill_qty > 0):
            new_cost = cost + e.fill_price * e.fill_qty
        else:
            # partial close: scale cost basis pro-rata so avg_entry stays stable
            if abs(new_qty) < 1e-12:
                new_cost = 0.0
            else:
                new_cost = cost * (new_qty / qty)
        cur[0], cur[1], cur[2] = new_qty, new_cost, max(cur[2], e.ts_epoch_ms)

    return {
        sym: PositionSnapshot(
            symbol=sym,
            qty=round(qty, 12),
            avg_entry=(cost / qty) if qty != 0 else 0.0,
            last_update_ts_ms=last_ts,
        )
        for sym, (qty, cost, last_ts) in agg.items()
        if abs(qty) > 1e-12
    }
```

Also update test helper to pass `symbol`:

```python
def ev(event_id, order_id, kind, symbol="ETHUSDT", price=2000, qty=0.1, side="buy", fee=0.0):
    signed_qty = qty if side == "buy" else -qty
    return BrokerEvent(
        event_id=event_id, kind=kind, order_id=order_id, symbol=symbol,
        ts_epoch_ms=0, fill_price=price, fill_qty=signed_qty, fee=fee,
    )
```

- [ ] **Step 7: Run `test_replay.py` — verify pass**

- [ ] **Step 8: Write test for `BrokerEventRepo` idempotent upsert**

```python
# tests/unit/execution/test_repositories.py
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from execution.base import BrokerEvent
from execution.repositories import BrokerEventRepo


@pytest.fixture
def migrated_db(tmp_path: Path) -> sa.Engine:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{tmp_path/'state.db'}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{tmp_path/'state.db'}")


def test_insert_or_ignore_is_idempotent(migrated_db):
    repo = BrokerEventRepo(migrated_db)
    e = BrokerEvent(event_id="e1", kind="filled", order_id="o1", symbol="ETHUSDT",
                    ts_epoch_ms=1, fill_price=2000.0, fill_qty=0.1, fee=0.05)
    assert repo.insert(e) is True
    assert repo.insert(e) is False                     # duplicate — no-op
    with migrated_db.connect() as conn:
        [count] = conn.execute(sa.text("SELECT COUNT(*) FROM broker_events")).one()
    assert count == 1
```

- [ ] **Step 9: Implement `src/execution/repositories.py`**

```python
"""SQLite repositories — thin SQLAlchemy Core wrappers around §8.1 tables.

All writes are append-only (spec §7.3); inserts on unique columns use
INSERT OR IGNORE to satisfy the §8.3 idempotency contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import sqlalchemy as sa

from decision.proposal import TradeProposal
from execution.base import BrokerEvent


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class BrokerEventRepo:
    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    def insert(self, event: BrokerEvent) -> bool:
        """INSERT OR IGNORE on event_id. Returns True if inserted, False if duplicate."""
        stmt = sa.text(
            "INSERT OR IGNORE INTO broker_events "
            "(event_id, kind, order_id, ts, fill_price, fill_qty, fee, reason, ml_model_version, llm_prompt_version) "
            "VALUES (:event_id, :kind, :order_id, :ts, :fill_price, :fill_qty, :fee, :reason, :mv, :pv)"
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt, {
                "event_id": event.event_id,
                "kind": event.kind,
                "order_id": event.order_id,
                "ts": datetime.fromtimestamp(event.ts_epoch_ms / 1000, tz=timezone.utc),
                "fill_price": event.fill_price,
                "fill_qty": event.fill_qty,
                "fee": event.fee,
                "reason": event.reason,
                "mv": event.ml_model_version,
                "pv": event.llm_prompt_version,
            })
            return result.rowcount == 1

    def all(self) -> list[BrokerEvent]:
        with self._engine.connect() as conn:
            rows = conn.execute(sa.text(
                "SELECT event_id, kind, order_id, ts, fill_price, fill_qty, fee, reason, "
                "ml_model_version, llm_prompt_version FROM broker_events ORDER BY ts"
            )).all()
        return [
            BrokerEvent(
                event_id=r[0], kind=r[1], order_id=r[2],
                ts_epoch_ms=int(r[3].timestamp() * 1000),
                fill_price=r[4], fill_qty=r[5], fee=r[6], reason=r[7],
                ml_model_version=r[8], llm_prompt_version=r[9],
            )
            for r in rows
        ]


class ProposalRepo:
    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    def insert(self, proposal: TradeProposal, accepted: bool) -> None:
        import json
        stmt = sa.text(
            "INSERT INTO proposals "
            "(proposal_id, trace_id, ts, symbol, direction, entry, stop_loss, take_profit_json, "
            " size, confidence, feature_snapshot_json, bundle_json, risk_checks_json, "
            " accepted, rationale, feature_registry_version, ml_model_version, llm_prompt_version) "
            "VALUES (:pid, :tid, :ts, :sym, :dir, :entry, :sl, :tp, :size, :conf, "
            " :feat, :bundle, :rc, :acc, :rat, :fv, :mv, :pv)"
        )
        with self._engine.begin() as conn:
            conn.execute(stmt, {
                "pid": proposal.proposal_id, "tid": proposal.trace_id, "ts": proposal.ts,
                "sym": proposal.symbol, "dir": proposal.direction,
                "entry": proposal.entry, "sl": proposal.stop_loss,
                "tp": json.dumps(proposal.take_profit),
                "size": proposal.size, "conf": proposal.confidence,
                "feat": json.dumps(proposal.feature_snapshot, default=str),
                "bundle": proposal.bundle_json,
                "rc": json.dumps([r.model_dump() for r in proposal.risk_checks]),
                "acc": accepted, "rat": proposal.rationale,
                "fv": proposal.feature_registry_version,
                "mv": proposal.ml_model_version,
                "pv": proposal.llm_prompt_version,
            })


class SessionStateRepo:
    """Holds consecutive_wins / day_pnl_r for SizingPipeline + DailyLossKill."""

    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    def get(self, d) -> tuple[int, float]:
        with self._engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT consecutive_wins, day_pnl_r FROM session_state WHERE date=:d"
            ), {"d": d}).first()
        return (row[0], row[1]) if row else (0, 0.0)

    def upsert(self, d, consecutive_wins: int, day_pnl_r: float) -> None:
        with self._engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO session_state (date, consecutive_wins, day_pnl_r, last_update_ts) "
                "VALUES (:d, :cw, :p, :ts) "
                "ON CONFLICT(date) DO UPDATE SET "
                "consecutive_wins=:cw, day_pnl_r=:p, last_update_ts=:ts"
            ), {"d": d, "cw": consecutive_wins, "p": day_pnl_r, "ts": _now()})
```

- [ ] **Step 10: Run repo tests — verify pass**

- [ ] **Step 11: Commit**

```bash
git add requirements.txt src/execution/base.py src/execution/replay.py src/execution/repositories.py
git add tests/unit/execution/test_replay.py tests/unit/execution/test_repositories.py
git commit -m "feat: rebuild_positions (pure, idempotent) + SQLite repos for broker_events/proposals/session_state"
```

---

## Task 3: PaperBroker core (latency + slippage + fees + partial fill)

**Files:**
- Create: `src/execution/paper_broker.py`
- Test: `tests/unit/execution/test_paper_broker.py`
- Modify: `config/settings.yaml` (add `fees` + `slippage` sections if not present)

Rationale: Spec §4.5 + §7.2. Real friction from day 1. Accepts `rng` so tests are deterministic.

- [ ] **Step 1: Write failing tests for determinism + fill event shape**

```python
# tests/unit/execution/test_paper_broker.py
import asyncio
import random

import pytest

from execution.base import Order
from execution.paper_broker import PaperBroker, PaperBrokerConfig


@pytest.fixture
def cfg():
    return PaperBrokerConfig(
        taker_bps=5.0,
        maker_bps=2.0,
        latency_ms_mean=10.0,
        latency_ms_stdev=1.0,
        partial_fill_prob=0.0,       # disable in unit tests
        rejection_prob=0.0,
    )


@pytest.mark.asyncio
async def test_market_order_fills_deterministically(cfg):
    rng = random.Random(42)
    broker = PaperBroker(cfg, rng=rng, mid_provider=lambda sym: 2000.0)
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
    order_id = await broker.submit(order)
    events = []
    async for e in broker.events():
        events.append(e)
        if e.kind == "filled":
            break
    assert events[0].kind == "submitted"
    assert events[-1].kind == "filled"
    assert abs(events[-1].fill_qty - 0.1) < 1e-9
    # deterministic: same seed → same fill_price
    broker2 = PaperBroker(cfg, rng=random.Random(42), mid_provider=lambda s: 2000.0)
    oid2 = await broker2.submit(order)
    events2 = [e async for e in _take_one_fill(broker2)]
    assert events2[-1].fill_price == events[-1].fill_price


async def _take_one_fill(broker):
    async for e in broker.events():
        yield e
        if e.kind == "filled":
            return


@pytest.mark.asyncio
async def test_slippage_and_fee_reflect_config(cfg):
    rng = random.Random(0)
    broker = PaperBroker(cfg, rng=rng, mid_provider=lambda sym: 2000.0)
    order = Order(client_order_id="c2", symbol="ETHUSDT", side="buy", type="market", qty=0.1)
    await broker.submit(order)
    async for e in broker.events():
        if e.kind == "filled":
            # fee in absolute: price * qty * bps/10000
            expected_fee = e.fill_price * 0.1 * cfg.taker_bps / 10000
            assert abs(e.fee - expected_fee) < 1e-6
            break
```

- [ ] **Step 2: Run failing**

- [ ] **Step 3: Implement `src/execution/paper_broker.py` (core — funding comes in Task 4)**

```python
"""PaperBroker — realistic friction from Day 1 (spec §4.5, §7.2).

- Latency: normal(mean, stdev) in ms.
- Fees: taker/maker bps → absolute via price * qty.
- Slippage: linear in order-size-vs-ADV proxy (ADV stubbed at 1000 for now;
  Plan 3 wires real ADV from TickRecorder aggregates).
- Partial fill: single-split probabilistic.
- Rejection: probabilistic (spread anomaly proxy).
- Accepts injected rng for determinism.
"""
from __future__ import annotations

import asyncio
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Callable

from execution.base import Balance, BrokerEvent, Order, OrderId, Position


@dataclass
class PaperBrokerConfig:
    taker_bps: float = 5.0
    maker_bps: float = 2.0
    latency_ms_mean: float = 200.0
    latency_ms_stdev: float = 50.0
    slippage_bps_base: float = 1.0
    slippage_bps_per_adv_unit: float = 20.0   # bps per (qty / ADV)
    adv_stub: float = 1000.0                  # Plan 3 wires real ADV
    partial_fill_prob: float = 0.15
    rejection_prob: float = 0.01


@dataclass
class PaperBroker:
    cfg: PaperBrokerConfig
    rng: random.Random
    mid_provider: Callable[[str], float]       # symbol -> current mid
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _orders: dict[OrderId, Order] = field(default_factory=dict)
    _positions: dict[str, Position] = field(default_factory=dict)
    _equity_usdt: float = 10_000.0

    async def submit(self, order: Order) -> OrderId:
        order_id = str(uuid.uuid4())
        self._orders[order_id] = order
        await self._emit("submitted", order_id, order)
        asyncio.create_task(self._simulate_fill(order_id, order))
        return order_id

    async def cancel(self, order_id: OrderId) -> None:
        await self._emit("cancelled", order_id, self._orders.get(order_id))

    async def positions(self) -> list[Position]:
        return list(self._positions.values())

    async def balance(self) -> Balance:
        return Balance(equity_usdt=self._equity_usdt, free_usdt=self._equity_usdt)

    async def events(self) -> AsyncIterator[BrokerEvent]:
        while True:
            e = await self._queue.get()
            yield e

    async def _simulate_fill(self, order_id: OrderId, order: Order) -> None:
        latency_s = max(0.0, self.rng.gauss(self.cfg.latency_ms_mean, self.cfg.latency_ms_stdev)) / 1000
        await asyncio.sleep(latency_s)

        if self.rng.random() < self.cfg.rejection_prob:
            await self._emit("rejected", order_id, order, reason="simulated_reject")
            return

        mid = self.mid_provider(order.symbol)
        sign = 1 if order.side == "buy" else -1
        slip_bps = (
            self.cfg.slippage_bps_base
            + self.cfg.slippage_bps_per_adv_unit * (order.qty / self.cfg.adv_stub)
        )
        fill_price = mid * (1 + sign * slip_bps / 10_000)

        if self.rng.random() < self.cfg.partial_fill_prob:
            first_qty = order.qty * self.rng.uniform(0.3, 0.7)
            await self._emit("partially_filled", order_id, order,
                             price=fill_price, qty=sign * first_qty)
            await asyncio.sleep(latency_s / 2)
            remain = order.qty - first_qty
            await self._emit("filled", order_id, order,
                             price=fill_price, qty=sign * remain)
        else:
            await self._emit("filled", order_id, order,
                             price=fill_price, qty=sign * order.qty)

    async def _emit(self, kind: str, order_id: OrderId, order: Order | None,
                    *, price: float | None = None, qty: float | None = None,
                    reason: str | None = None) -> None:
        fee = None
        if price is not None and qty is not None:
            fee_bps = self.cfg.taker_bps if (order and order.type == "market") else self.cfg.maker_bps
            fee = price * abs(qty) * fee_bps / 10_000
        event = BrokerEvent(
            event_id=str(uuid.uuid4()),
            kind=kind, order_id=order_id,
            symbol=order.symbol if order else "",
            ts_epoch_ms=int(datetime.now(tz=timezone.utc).timestamp() * 1000),
            fill_price=price, fill_qty=qty, fee=fee, reason=reason,
        )
        await self._queue.put(event)
```

- [ ] **Step 4: Run tests — verify pass**

```bash
pytest tests/unit/execution/test_paper_broker.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/execution/paper_broker.py tests/unit/execution/test_paper_broker.py
git commit -m "feat: PaperBroker with real friction (latency, slippage, fees, partial fill)"
```

---

## Task 4: PaperBroker funding task

**Files:**
- Modify: `src/execution/paper_broker.py`
- Test: `tests/unit/execution/test_paper_broker.py` (append)

Rationale: Spec §4.5 — `funding_charged` events every 8h from `data/funding/<symbol>.parquet`. Fixed rate is **test-only**; production reads the parquet (Task 7 writes it).

- [ ] **Step 1: Write failing funding test**

```python
# tests/unit/execution/test_paper_broker.py  (append)
import pandas as pd
from pathlib import Path


@pytest.mark.asyncio
async def test_funding_tick_emits_charged_event(tmp_path: Path, cfg):
    funding_df = pd.DataFrame(
        {"funding_rate": [0.0001]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-04-18T00:00", tz="UTC")]),
    )
    funding_df.to_parquet(tmp_path / "ETHUSDT.parquet")

    broker = PaperBroker(cfg, rng=random.Random(0), mid_provider=lambda s: 2000.0,
                        funding_dir=tmp_path)
    # seed a position
    from execution.base import Position
    from datetime import datetime, timezone
    broker._positions["ETHUSDT"] = Position(
        symbol="ETHUSDT", qty=0.1, avg_entry=2000.0,
        opened_at=datetime(2026, 4, 17, tzinfo=timezone.utc),
        last_update_ts=datetime(2026, 4, 17, tzinfo=timezone.utc),
    )

    await broker.tick_funding(pd.Timestamp("2026-04-18T00:00", tz="UTC"))
    # drain queue
    evt = await asyncio.wait_for(broker._queue.get(), timeout=0.1)
    assert evt.kind == "funding_charged"
    # fee = notional * funding_rate (long pays positive)
    expected = 0.1 * 2000.0 * 0.0001
    assert abs(evt.fee - expected) < 1e-8
```

- [ ] **Step 2: Extend PaperBroker**

Add to `PaperBroker` dataclass field:

```python
funding_dir: Path | None = None
```

Add method:

```python
async def tick_funding(self, ts: pd.Timestamp) -> None:
    """Called by orchestrator every 8h. Emits funding_charged per open position."""
    if self.funding_dir is None:
        return
    for sym, pos in self._positions.items():
        parquet = self.funding_dir / f"{sym}.parquet"
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet)
        row = df.loc[df.index == ts]
        if row.empty:
            continue
        rate = float(row["funding_rate"].iloc[0])
        mid = self.mid_provider(sym)
        fee = pos.qty * mid * rate       # long pays positive funding
        event = BrokerEvent(
            event_id=str(uuid.uuid4()),
            kind="funding_charged", order_id=f"funding_{sym}_{int(ts.timestamp())}",
            symbol=sym, ts_epoch_ms=int(ts.timestamp() * 1000),
            fee=fee, reason=f"rate={rate}",
        )
        await self._queue.put(event)
```

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git add src/execution/paper_broker.py tests/unit/execution/test_paper_broker.py
git commit -m "feat(paper-broker): funding_charged tick driven by data/funding/<symbol>.parquet"
```

---

## Task 5: Broker contract test

**Files:**
- Create: `tests/contracts/test_broker_contract.py`

Rationale: Spec §9.4 — one abstract contract per Protocol. `PaperBroker` is the first impl; `ReplayBroker` and `LiveBroker` in Plan 3 plug into the same suite.

- [ ] **Step 1: Write contract suite**

```python
"""Shared Broker contract — every implementation must pass this suite.

Lives under tests/contracts/ per spec §9.4. A concrete Broker adds a
pytest fixture named `broker` and a `mid_provider` fixture; the contract
tests run automatically.
"""
from __future__ import annotations

import asyncio
import random

import pytest

from execution.base import Order
from execution.paper_broker import PaperBroker, PaperBrokerConfig


@pytest.fixture
def paper_broker():
    return PaperBroker(
        cfg=PaperBrokerConfig(
            latency_ms_mean=5.0, latency_ms_stdev=1.0,
            partial_fill_prob=0.0, rejection_prob=0.0,
        ),
        rng=random.Random(1),
        mid_provider=lambda s: 2000.0,
    )


class BrokerContractTests:
    broker_fixture = "paper_broker"       # override in subclasses

    @pytest.mark.asyncio
    async def test_submit_yields_submitted_then_filled(self, request):
        broker = request.getfixturevalue(self.broker_fixture)
        o = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
        await broker.submit(o)
        seen = []
        async for e in broker.events():
            seen.append(e.kind)
            if e.kind in ("filled", "rejected"):
                break
        assert seen[0] == "submitted"
        assert seen[-1] == "filled"

    @pytest.mark.asyncio
    async def test_event_ids_are_unique(self, request):
        broker = request.getfixturevalue(self.broker_fixture)
        await broker.submit(Order(client_order_id="c1", symbol="ETHUSDT",
                                  side="buy", type="market", qty=0.1))
        await broker.submit(Order(client_order_id="c2", symbol="ETHUSDT",
                                  side="sell", type="market", qty=0.05))
        ids = set()
        # Collect 4 events (2 submitted + 2 filled)
        for _ in range(4):
            e = await asyncio.wait_for(broker._queue.get(), timeout=1.0)
            assert e.event_id not in ids
            ids.add(e.event_id)


class TestPaperBrokerContract(BrokerContractTests):
    broker_fixture = "paper_broker"
```

- [ ] **Step 2: Run contract suite — verify pass**

```bash
pytest tests/contracts/test_broker_contract.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/contracts/test_broker_contract.py
git commit -m "test(contracts): Broker contract suite; PaperBroker passes"
```

---

## Task 6: BinanceKline DataSource

**Files:**
- Create: `src/data/binance_kline.py`
- Test: `tests/unit/data/test_binance_kline.py`

Rationale: Spec §4.1 — Day-1 `DataSource` impl. Wraps `python-binance`'s `AsyncClient`. Tests use a faked client — no network.

- [ ] **Step 1: Write failing test with fake client**

```python
# tests/unit/data/test_binance_kline.py
import pytest
import pandas as pd
from datetime import datetime, timezone

from data.binance_kline import BinanceKline


class FakeAsyncClient:
    async def get_klines(self, *, symbol, interval, startTime=None, endTime=None, limit=500):
        return [
            # [open_time, open, high, low, close, volume, close_time, qav, n_trades, tbv, tqv, ignore]
            [1700000000000, "2000", "2010", "1990", "2005", "10", 1700003599999,
             "0", 100, "0", "0", "0"],
            [1700003600000, "2005", "2015", "2000", "2012", "12", 1700007199999,
             "0", 120, "0", "0", "0"],
        ]

    async def close_connection(self): ...


@pytest.mark.asyncio
async def test_fetch_latest_returns_dataframe():
    ds = BinanceKline(client=FakeAsyncClient())
    df = await ds.fetch_latest("ETHUSDT", "1h", n=2)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")
    assert df["open"].iloc[0] == 2000.0


def test_supports_only_known_intervals():
    ds = BinanceKline(client=None)
    assert ds.supports("ETHUSDT", "1h") is True
    assert ds.supports("ETHUSDT", "7m") is False
```

- [ ] **Step 2: Run — fails (ImportError)**

- [ ] **Step 3: Implement `src/data/binance_kline.py`**

```python
"""BinanceKline — first DataSource implementation (spec §4.1).

Wraps python-binance AsyncClient. Public constructor accepts an injected
client (for tests). Real client lifecycle is managed by
`BinanceKline.open(api_key, api_secret)` classmethod.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd


_VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}


class BinanceKline:
    name = "binance_kline"

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    async def open(cls, api_key: str = "", api_secret: str = "") -> "BinanceKline":
        from binance import AsyncClient
        client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
        return cls(client)

    async def close(self) -> None:
        await self._client.close_connection()

    def supports(self, symbol: str, timeframe: str) -> bool:
        return timeframe in _VALID_INTERVALS

    async def fetch_latest(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        raw = await self._client.get_klines(symbol=symbol, interval=timeframe, limit=n)
        return self._to_df(raw)

    async def fetch(
        self, symbol: str, timeframe: str, since: datetime, until: datetime
    ) -> pd.DataFrame:
        start_ms = int(since.timestamp() * 1000)
        end_ms = int(until.timestamp() * 1000)
        raw = await self._client.get_klines(
            symbol=symbol, interval=timeframe,
            startTime=start_ms, endTime=end_ms, limit=1000,
        )
        return self._to_df(raw)

    @staticmethod
    def _to_df(raw: Iterable[list]) -> pd.DataFrame:
        rows = list(raw)
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "n_trades", "tbv", "tqv", "_ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df[["open", "high", "low", "close", "volume"]]
```

- [ ] **Step 4: Run — pass**

- [ ] **Step 5: Commit**

```bash
git add src/data/binance_kline.py tests/unit/data/test_binance_kline.py
git commit -m "feat(data): BinanceKline DataSource impl (spec §4.1)"
```

---

## Task 7: FundingRateDataSource + writer

**Files:**
- Create: `src/data/funding.py`
- Test: `tests/unit/data/test_funding.py`

Rationale: Spec §4.1 + §4.5. Writes `data/funding/<symbol>.parquet`. Activates `FundingFeature` from Plan 1 (which returns `{}` without the parquet).

- [ ] **Step 1: Write failing test**

```python
# tests/unit/data/test_funding.py
import pandas as pd
import pytest
from pathlib import Path

from data.funding import FundingRateWriter, load_funding


class FakeFundingClient:
    async def futures_funding_rate(self, *, symbol, startTime=None, endTime=None, limit=1000):
        return [
            {"fundingTime": 1700000000000, "fundingRate": "0.0001"},
            {"fundingTime": 1700028800000, "fundingRate": "0.0002"},
        ]
    async def close_connection(self): ...


@pytest.mark.asyncio
async def test_writer_persists_parquet(tmp_path: Path):
    w = FundingRateWriter(client=FakeFundingClient(), out_dir=tmp_path)
    n = await w.update("ETHUSDT")
    assert n == 2
    df = load_funding(tmp_path / "ETHUSDT.parquet")
    assert list(df.columns) == ["funding_rate"]
    assert df.index.tz is not None
```

- [ ] **Step 2: Implement `src/data/funding.py`**

```python
"""Funding rate ingestion — spec §4.1.

Binance futures funding rate every 8h. Stored at data/funding/<symbol>.parquet
with DatetimeIndex (UTC) and single column funding_rate (float).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class FundingRateWriter:
    def __init__(self, client: Any, out_dir: Path) -> None:
        self._client = client
        self._out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    async def update(self, symbol: str) -> int:
        """Fetch new funding rows since last persisted ts; upsert to parquet."""
        out = self._out_dir / f"{symbol}.parquet"
        existing = load_funding(out) if out.exists() else pd.DataFrame()
        start_ms = int(existing.index.max().timestamp() * 1000) + 1 if not existing.empty else None
        raw = await self._client.futures_funding_rate(symbol=symbol, startTime=start_ms, limit=1000)
        if not raw:
            return 0
        new_df = pd.DataFrame(raw)
        new_df["ts"] = pd.to_datetime(new_df["fundingTime"], unit="ms", utc=True)
        new_df = new_df.set_index("ts")
        new_df["funding_rate"] = new_df["fundingRate"].astype(float)
        new_df = new_df[["funding_rate"]]
        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(out)
        return len(new_df)


def load_funding(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
```

- [ ] **Step 3: Run tests — verify pass**

- [ ] **Step 4: Integration test — activate Plan-1 FundingFeature**

Add to `tests/unit/features/test_funding_rate.py`:

```python
def test_funding_feature_non_empty_when_parquet_present(tmp_path, monkeypatch, eth_1h_df):
    import pandas as pd
    from features.funding_rate import FundingFeature

    # write a minimal parquet at the path FundingFeature looks up
    funding_dir = tmp_path / "data" / "funding"
    funding_dir.mkdir(parents=True)
    pd.DataFrame(
        {"funding_rate": [0.0001]},
        index=pd.DatetimeIndex([eth_1h_df.index[0]], name="ts"),
    ).to_parquet(funding_dir / "ETHUSDT.parquet")

    monkeypatch.chdir(tmp_path)
    f = FundingFeature(symbol="ETHUSDT")
    result = f.compute(eth_1h_df, as_of=eth_1h_df.index[10])
    assert result != {}
```

(If `FundingFeature` hard-codes its parquet path, the test will guide a follow-up refactor; acceptable to mark xfail if immediate integration is out of scope.)

- [ ] **Step 5: Commit**

```bash
git add src/data/funding.py tests/unit/data/test_funding.py tests/unit/features/test_funding_rate.py
git commit -m "feat(data): FundingRateWriter + parquet contract (activates Plan-1 FundingFeature)"
```

---

## Task 8: TickRecorder

**Files:**
- Create: `src/execution/tick_recorder.py`
- Test: `tests/unit/execution/test_tick_recorder.py`

Rationale: Spec §4.9. Runs from Day 1 so `ReplayBroker` has fuel by the time pre-live gate is reached.

- [ ] **Step 1: Write test — recorder writes JSONL lines**

```python
# tests/unit/execution/test_tick_recorder.py
import asyncio
import json
from pathlib import Path

import pytest

from execution.tick_recorder import TickRecorder


class FakeTradeStream:
    def __init__(self, trades):
        self._trades = trades

    async def __aiter__(self):
        for t in self._trades:
            yield t
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_records_to_daily_jsonl(tmp_path: Path):
    stream = FakeTradeStream([
        {"t": 1700000000000, "p": "2000", "q": "0.1", "m": False},
        {"t": 1700000001000, "p": "2001", "q": "0.05", "m": True},
    ])
    rec = TickRecorder(symbol="ETHUSDT", out_dir=tmp_path, stream_factory=lambda sym: stream)
    await rec.record_once_for_test()    # test hook: drains one aiter pass

    day_file = tmp_path / "ETHUSDT" / "2023-11-14.jsonl"
    assert day_file.exists()
    lines = day_file.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["p"] == "2000"
```

- [ ] **Step 2: Implement `src/execution/tick_recorder.py`**

```python
"""TickRecorder — spec §4.9. Fuel for ReplayBroker.

Subscribes to Binance trades WS, appends raw ticks to
data/ticks/<symbol>/<YYYY-MM-DD>.jsonl. Rollover at UTC midnight.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable


class TickRecorder:
    def __init__(
        self,
        symbol: str,
        out_dir: Path,
        stream_factory: Callable[[str], AsyncIterator[dict[str, Any]]],
    ) -> None:
        self.symbol = symbol
        self.out_dir = out_dir / symbol
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._stream_factory = stream_factory

    async def run(self) -> None:
        """Run forever. Orchestrator owns this task's lifecycle."""
        stream = self._stream_factory(self.symbol)
        async for tick in stream:
            self._append(tick)

    async def record_once_for_test(self) -> None:
        stream = self._stream_factory(self.symbol)
        async for tick in stream:
            self._append(tick)

    def _append(self, tick: dict[str, Any]) -> None:
        ts = datetime.fromtimestamp(tick["t"] / 1000, tz=timezone.utc)
        path = self.out_dir / f"{ts.date().isoformat()}.jsonl"
        with path.open("a") as fh:
            fh.write(json.dumps(tick, separators=(",", ":")) + "\n")
```

- [ ] **Step 3: Run, commit**

```bash
git add src/execution/tick_recorder.py tests/unit/execution/test_tick_recorder.py
git commit -m "feat(execution): TickRecorder → data/ticks/<symbol>/<date>.jsonl (spec §4.9)"
```

---

## Task 9: RiskPipeline (ordered, reject on first fail)

**Files:**
- Create: `src/decision/risk/pipeline.py`, `src/decision/risk/checks.py`
- Test: `tests/unit/decision/test_risk_pipeline.py`

Rationale: Spec §4.4 — Trade + Daily + Portfolio + System levels. Day-1 implements the essentials: `MandatoryStopLoss`, `SpreadGate`, `DailyLossKillSwitch`, `MaxConcurrentPositions`. Plan 3 adds Drift / Correlated / NetDirectional.

- [ ] **Step 1: Write tests for each check + pipeline ordering**

```python
# tests/unit/decision/test_risk_pipeline.py
from datetime import datetime, timezone

import pytest

from decision.proposal import PortfolioSnapshot, TradeProposal
from decision.risk.pipeline import RiskPipeline
from decision.risk.checks import (
    DailyLossKillSwitch, MandatoryStopLoss, MaxConcurrentPositions, SpreadGate,
)


def _prop(**kw) -> TradeProposal:
    base = dict(
        proposal_id="p1", trace_id="t1",
        ts=datetime(2026, 4, 18, tzinfo=timezone.utc),
        symbol="ETHUSDT", direction="long",
        entry=2000.0, stop_loss=1980.0, take_profit=[2020.0],
        size=0.1, confidence=0.65,
        feature_snapshot={}, bundle_json="{}", risk_checks=[],
        feature_registry_version="1.0.0",
        ml_model_version="stub", llm_prompt_version="stub",
    )
    base.update(kw)
    return TradeProposal(**base)


def _port(**kw) -> PortfolioSnapshot:
    base = dict(equity_usdt=10_000, open_positions={}, day_pnl_r=0.0, consecutive_wins=0)
    base.update(kw)
    return PortfolioSnapshot(**base)


def test_mandatory_sl_rejects_when_missing():
    c = MandatoryStopLoss()
    p = _prop(stop_loss=0.0)
    assert c.check(p, _port()).passed is False


def test_spread_gate_rejects_wide_spread():
    c = SpreadGate(max_bps=20.0, spread_provider=lambda sym: 25.0)
    assert c.check(_prop(), _port()).passed is False


def test_daily_loss_kill_switch_rejects_below_threshold():
    c = DailyLossKillSwitch(threshold_r=-2.0)
    assert c.check(_prop(), _port(day_pnl_r=-2.5)).passed is False


def test_max_concurrent_rejects_at_cap():
    c = MaxConcurrentPositions(cap=3)
    full = _port(open_positions={"BTC": 0.1, "SOL": 0.2, "LINK": 1.0})
    assert c.check(_prop(), full).passed is False


def test_pipeline_rejects_on_first_fail_and_records_all_names():
    p = RiskPipeline([
        MandatoryStopLoss(),
        SpreadGate(max_bps=20.0, spread_provider=lambda s: 50.0),  # will reject
        DailyLossKillSwitch(threshold_r=-2.0),
    ])
    results = p.evaluate(_prop(), _port())
    assert any(r.name == "SpreadGate" and not r.passed for r in results)
    assert p.is_accepted(results) is False
```

- [ ] **Step 2: Implement checks**

```python
# src/decision/risk/checks.py
"""Day-1 risk checks — spec §4.4.

Spread is queried from an injected provider so tests don't need a live
L2 book. Production will plug in a MidProvider backed by TickRecorder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from decision.proposal import PortfolioSnapshot, RiskCheckResult, TradeProposal


@dataclass
class MandatoryStopLoss:
    name: str = "MandatoryStopLoss"

    def check(self, p: TradeProposal, port: PortfolioSnapshot) -> RiskCheckResult:
        if p.stop_loss <= 0:
            return RiskCheckResult(name=self.name, passed=False, detail="stop_loss<=0")
        if p.direction == "long" and p.stop_loss >= p.entry:
            return RiskCheckResult(name=self.name, passed=False, detail="long SL>=entry")
        if p.direction == "short" and p.stop_loss <= p.entry:
            return RiskCheckResult(name=self.name, passed=False, detail="short SL<=entry")
        return RiskCheckResult(name=self.name, passed=True)


@dataclass
class SpreadGate:
    max_bps: float
    spread_provider: Callable[[str], float]    # symbol -> current bps
    name: str = "SpreadGate"

    def check(self, p: TradeProposal, port: PortfolioSnapshot) -> RiskCheckResult:
        bps = self.spread_provider(p.symbol)
        if bps > self.max_bps:
            return RiskCheckResult(name=self.name, passed=False, detail=f"{bps:.1f}bps>{self.max_bps}")
        return RiskCheckResult(name=self.name, passed=True)


@dataclass
class DailyLossKillSwitch:
    threshold_r: float     # e.g., -2.0
    name: str = "DailyLossKillSwitch"

    def check(self, p: TradeProposal, port: PortfolioSnapshot) -> RiskCheckResult:
        if port.day_pnl_r <= self.threshold_r:
            return RiskCheckResult(name=self.name, passed=False,
                                   detail=f"day_pnl_r={port.day_pnl_r}<={self.threshold_r}")
        return RiskCheckResult(name=self.name, passed=True)


@dataclass
class MaxConcurrentPositions:
    cap: int
    name: str = "MaxConcurrentPositions"

    def check(self, p: TradeProposal, port: PortfolioSnapshot) -> RiskCheckResult:
        # if the proposal's symbol is already open, adding to it doesn't count as new.
        currently_open_distinct = set(port.open_positions.keys())
        would_be = currently_open_distinct | {p.symbol}
        if len(would_be) > self.cap:
            return RiskCheckResult(name=self.name, passed=False,
                                   detail=f"{len(would_be)}>{self.cap}")
        return RiskCheckResult(name=self.name, passed=True)
```

- [ ] **Step 3: Implement pipeline**

```python
# src/decision/risk/pipeline.py
"""Risk pipeline — evaluate checks in order; any fail → reject."""
from __future__ import annotations

from typing import Iterable

from decision.proposal import PortfolioSnapshot, RiskCheckResult, TradeProposal
from decision.risk.base import RiskCheck


class RiskPipeline:
    def __init__(self, checks: Iterable[RiskCheck]) -> None:
        self._checks = list(checks)

    def evaluate(self, p: TradeProposal, port: PortfolioSnapshot) -> list[RiskCheckResult]:
        results: list[RiskCheckResult] = []
        for c in self._checks:
            r = c.check(p, port)
            results.append(r)
            if not r.passed:
                return results               # short-circuit on first fail
        return results

    @staticmethod
    def is_accepted(results: list[RiskCheckResult]) -> bool:
        return all(r.passed for r in results)
```

- [ ] **Step 4: Run tests, commit**

```bash
git add src/decision/risk/pipeline.py src/decision/risk/checks.py tests/unit/decision/test_risk_pipeline.py
git commit -m "feat(risk): RiskPipeline + Day-1 checks (MandatorySL, Spread, DailyLoss, MaxConcurrent)"
```

---

## Task 10: SizingPipeline (fixed fractional)

**Files:**
- Create: `src/decision/sizing.py`
- Test: `tests/unit/decision/test_sizing.py`

Rationale: Spec §4.4 — Day-1 ships `[IdentityModifier()]` plus the `FixedFractional` base sizer. No win-streak taper (spec §7.6).

- [ ] **Step 1: Tests**

```python
# tests/unit/decision/test_sizing.py
from decision.sizing import FixedFractionalSizer, SizingPipeline, IdentityModifier


def test_fixed_fractional_honours_risk_budget():
    # 0.25% of 10_000 = 25 USDT risk budget.
    # SL distance 20 USDT → size = 25 / 20 = 1.25 units.
    s = FixedFractionalSizer(fraction=0.0025)
    size = s.size(equity_usdt=10_000, entry=2000, stop_loss=1980)
    assert abs(size - 1.25) < 1e-9


def test_sizing_pipeline_with_identity_is_noop():
    p = SizingPipeline([IdentityModifier()])
    assert p.apply(5.0, consecutive_wins=10, day_pnl_r=-0.5) == 5.0
```

- [ ] **Step 2: Implement**

```python
# src/decision/sizing.py
"""Sizing pipeline — spec §4.4.

FixedFractionalSizer converts (equity, entry, SL) → contract units.
SizingPipeline applies post-sizing modifiers; Day-1 ships identity only
because win-streak tapering has no basis for calibrated models (§7.6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass
class FixedFractionalSizer:
    fraction: float       # e.g., 0.0025 = 0.25% risk per trade

    def size(self, equity_usdt: float, entry: float, stop_loss: float) -> float:
        risk_budget = equity_usdt * self.fraction
        distance = abs(entry - stop_loss)
        if distance <= 0:
            raise ValueError("entry == stop_loss; cannot size")
        return risk_budget / distance


class SizingModifier(Protocol):
    name: str

    def apply(self, size: float, consecutive_wins: int, day_pnl_r: float) -> float: ...


@dataclass
class IdentityModifier:
    name: str = "IdentityModifier"

    def apply(self, size: float, consecutive_wins: int, day_pnl_r: float) -> float:
        return size


class SizingPipeline:
    def __init__(self, modifiers: Iterable[SizingModifier]) -> None:
        self._modifiers = list(modifiers)

    def apply(self, size: float, *, consecutive_wins: int, day_pnl_r: float) -> float:
        for m in self._modifiers:
            size = m.apply(size, consecutive_wins, day_pnl_r)
        return size
```

- [ ] **Step 3: Commit**

```bash
git add src/decision/sizing.py tests/unit/decision/test_sizing.py
git commit -m "feat(sizing): FixedFractionalSizer + SizingPipeline with identity modifier"
```

---

## Task 11: Re-home trade_setup from _legacy/ to decision/

**Files:**
- Move: `src/decision/_legacy/trade_setup.py` → `src/decision/trade_setup.py`
- Move: `tests/unit/test_trade_setup_legacy.py` → `tests/unit/decision/test_trade_setup.py`
- Delete: `src/decision/_legacy/` (if empty after move)

Rationale: Plan 1 parked this here as a temporary shim. With Policy + RiskPipeline now in place, it finds its permanent home.

- [ ] **Step 1: Move files, fix imports**

```bash
git mv src/decision/_legacy/trade_setup.py src/decision/trade_setup.py
git mv tests/unit/test_trade_setup_legacy.py tests/unit/decision/test_trade_setup.py
```

Edit `tests/unit/decision/test_trade_setup.py`:

```python
# change: from decision._legacy.trade_setup import ...
# to:     from decision.trade_setup import ...
```

Remove empty `_legacy` dir:

```bash
rmdir src/decision/_legacy 2>/dev/null || true
rm -f src/decision/_legacy/__init__.py
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/unit/decision/test_trade_setup.py -v
```
Expected: pass.

- [ ] **Step 3: Grep for any lingering `_legacy` imports**

```bash
grep -r "decision._legacy" src tests && echo "FOUND" || echo "clean"
```
Expected: `clean`.

- [ ] **Step 4: Commit**

```bash
git add src/decision/ tests/unit/decision/
git commit -m "refactor(decision): re-home trade_setup from _legacy to decision/ (Plan 2 complete)"
```

---

## Task 12: Policy MVP — threshold policy producing TradeProposal

**Files:**
- Create: `src/decision/policy.py` (extend with `ThresholdPolicy` class)
- Test: `tests/unit/decision/test_policy.py`

Rationale: Spec §4.4. The simplest Policy that exercises the full pipeline: `prob_up > accept_long_threshold` → long; `prob_up < accept_short_threshold` → short; else `None`. Uses `trade_setup` (Task 11) for entry/SL/TP.

- [ ] **Step 1: Tests**

```python
# tests/unit/decision/test_policy.py
import asyncio
from datetime import datetime, timezone

import pytest

from decision.policy import ThresholdPolicy
from decision.proposal import PortfolioSnapshot
from models.base import PredictionBundle


def _bundle(direction="long", prob_up=0.70) -> PredictionBundle:
    return PredictionBundle(
        direction=direction, prob_up=prob_up, horizon_bars=4,
        size_multiplier=1.0, feature_snapshot_hash="h",
        feature_registry_version="1.0.0",
        ml_model_version="stub", llm_prompt_version="stub",
        predictions_detail={},
    )


@pytest.mark.asyncio
async def test_policy_emits_long_on_high_prob():
    pol = ThresholdPolicy(long_threshold=0.55, short_threshold=0.45,
                          symbol="ETHUSDT", mid_provider=lambda s: 2000.0,
                          atr_provider=lambda s: 10.0)
    features = {"smc": {}, "fib": {}, "liquidity": {}, "divergence": {},
                "funding": {}, "confidence": {}}
    port = PortfolioSnapshot(equity_usdt=10_000, open_positions={}, day_pnl_r=0.0, consecutive_wins=0)
    proposal = await pol.propose(features, _bundle(), port)
    assert proposal is not None
    assert proposal.direction == "long"
    assert proposal.stop_loss < proposal.entry


@pytest.mark.asyncio
async def test_policy_returns_none_on_flat():
    pol = ThresholdPolicy(long_threshold=0.55, short_threshold=0.45,
                          symbol="ETHUSDT", mid_provider=lambda s: 2000.0,
                          atr_provider=lambda s: 10.0)
    flat = _bundle(direction="flat", prob_up=0.50)
    port = PortfolioSnapshot(equity_usdt=10_000, open_positions={}, day_pnl_r=0.0, consecutive_wins=0)
    assert await pol.propose({}, flat, port) is None
```

- [ ] **Step 2: Append `ThresholdPolicy` to `src/decision/policy.py`**

```python
# src/decision/policy.py  (append below the Protocol definition)
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from decision.proposal import PortfolioSnapshot, TradeProposal
from models.base import PredictionBundle


@dataclass
class ThresholdPolicy:
    long_threshold: float
    short_threshold: float
    symbol: str
    mid_provider: Callable[[str], float]
    atr_provider: Callable[[str], float]           # stop distance proxy (USDT)
    tp_multiples: tuple[float, ...] = (1.5, 3.0)

    async def propose(
        self,
        features: dict[str, Any],
        bundle: PredictionBundle,
        portfolio: PortfolioSnapshot,
    ) -> TradeProposal | None:
        if bundle.direction == "flat" or bundle.size_multiplier == 0.0:
            return None
        if bundle.direction == "long" and bundle.prob_up < self.long_threshold:
            return None
        if bundle.direction == "short" and bundle.prob_up > (1 - self.short_threshold):
            return None

        mid = self.mid_provider(self.symbol)
        atr = self.atr_provider(self.symbol)
        sign = 1 if bundle.direction == "long" else -1
        entry = mid
        stop_loss = mid - sign * atr
        take_profit = [mid + sign * atr * m for m in self.tp_multiples]

        return TradeProposal(
            proposal_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            ts=datetime.now(tz=timezone.utc),
            symbol=self.symbol,
            direction=bundle.direction,
            entry=entry, stop_loss=stop_loss, take_profit=take_profit,
            size=1.0,                                # sized later by SizingPipeline
            confidence=bundle.prob_up if bundle.direction == "long" else 1 - bundle.prob_up,
            feature_snapshot=features,
            bundle_json=bundle.model_dump_json(),
            risk_checks=[],
            feature_registry_version=bundle.feature_registry_version,
            ml_model_version=bundle.ml_model_version,
            llm_prompt_version=bundle.llm_prompt_version,
        )
```

- [ ] **Step 3: Run tests, commit**

```bash
git add src/decision/policy.py tests/unit/decision/test_policy.py
git commit -m "feat(decision): ThresholdPolicy — first concrete Policy producing TradeProposal"
```

---

## Task 13: XGBPredictor stub

**Files:**
- Create: `src/models/xgb_predictor.py`
- Test: `tests/unit/models/test_xgb_predictor.py`

Rationale: Unblocks orchestrator wiring (Task 17) without requiring trained model. Task 14 replaces the stub with a real trained + calibrated predictor.

- [ ] **Step 1: Test — stub returns fixed direction/prob**

```python
# tests/unit/models/test_xgb_predictor.py
import pytest

from models.xgb_predictor import XGBPredictor


@pytest.mark.asyncio
async def test_stub_returns_bundle_with_fixed_prob():
    pred = XGBPredictor.stub(prob_up=0.62, ml_model_version="stub-v0")
    bundle = await pred.predict({"smc": {}, "confidence": {"score": 5}})
    assert bundle.prob_up == 0.62
    assert bundle.direction == "long"
    assert bundle.ml_model_version == "stub-v0"
    assert bundle.feature_snapshot_hash        # non-empty
```

- [ ] **Step 2: Implement stub interface (real loader comes in Task 14)**

```python
# src/models/xgb_predictor.py
"""XGBPredictor — spec §4.3.

Ships in two forms:
  - XGBPredictor.stub(prob_up) → fixed predictor for scaffolding tests
  - XGBPredictor.load(path, calibrator_path) → real trained + calibrated
The `predict()` signature is stable across both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from features.registry import canonical_hash
from models.base import PredictionBundle


@dataclass
class XGBPredictor:
    _model: Any = None
    _calibrator: Any = None
    _feature_order: tuple[str, ...] = ()
    ml_model_version: str = "stub-v0"
    _fixed_prob: float | None = None

    @classmethod
    def stub(cls, prob_up: float, ml_model_version: str = "stub-v0") -> "XGBPredictor":
        return cls(ml_model_version=ml_model_version, _fixed_prob=prob_up)

    async def predict(self, features: dict[str, Any]) -> PredictionBundle:
        prob_up = self._fixed_prob if self._fixed_prob is not None else self._run_model(features)
        direction = "long" if prob_up > 0.52 else ("short" if prob_up < 0.48 else "flat")
        return PredictionBundle(
            direction=direction,
            prob_up=float(prob_up),
            horizon_bars=4,
            size_multiplier=1.0,
            feature_snapshot_hash=canonical_hash(features),
            feature_registry_version="1.0.0",
            ml_model_version=self.ml_model_version,
            llm_prompt_version="none",
            predictions_detail={"xgb_prob_up": prob_up},
        )

    def _run_model(self, features: dict[str, Any]) -> float:
        raise NotImplementedError("Task 14 wires the real trained model")
```

- [ ] **Step 3: Commit**

```bash
git add src/models/xgb_predictor.py tests/unit/models/test_xgb_predictor.py
git commit -m "feat(models): XGBPredictor stub (unblocks orchestrator; Task 14 trains real model)"
```

---

## Task 14: Training script + isotonic calibration

**Files:**
- Create: `scripts/train_xgb.py`
- Modify: `src/models/xgb_predictor.py` (wire real `_run_model`)
- Create: `tests/unit/models/test_xgb_predictor_real.py`
- New directory: `models/` at repo root (not under src/ — trained artifacts), add to `.gitignore`

Rationale: Spec §4.3 + §10.1 gate 3 (calibration is a pre-live blocker). Isotonic chosen over Platt because it doesn't assume a sigmoid shape; trade-off is requiring more data, which we accept given 60-day paper runtime.

- [ ] **Step 1: Append to `.gitignore`**

```
models/
!models/.gitkeep
```

Touch `models/.gitkeep` to anchor the directory.

- [ ] **Step 2: Write `scripts/train_xgb.py`**

```python
"""Trains XGBoost on historical ETHUSDT features + isotonic calibration.

Labels: 4-bar forward return > 0. Features: build_default_registry().
Writes:
  - models/xgb_<model_version>.json (booster)
  - models/calib_<model_version>.pkl (IsotonicRegression)
  - row in SQLite model_versions

Usage:
  python scripts/train_xgb.py --data data/ETHUSDT_1h_long.csv
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

from features.registry import build_default_registry


def _make_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    reg = build_default_registry()
    rows = []
    ys = []
    for i, ts in enumerate(df.index):
        if i < 200 or i > len(df) - 5:
            continue
        feats = reg.compute_all(df, as_of=ts)
        flat = _flatten(feats)
        rows.append(flat)
        # label: 4-bar forward return > 0
        y = int(df["close"].iloc[i + 4] > df["close"].iloc[i])
        ys.append(y)
    X = pd.DataFrame(rows).fillna(0.0)
    return X, pd.Series(ys, name="y")


def _flatten(features: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in features.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=kk + "."))
        elif isinstance(v, (int, float)):
            out[kk] = float(v)
        elif isinstance(v, bool):
            out[kk] = float(v)
    return out


def train(data_path: Path, out_dir: Path) -> str:
    df = pd.read_csv(data_path, parse_dates=["open_time"]).set_index("open_time")
    X, y = _make_dataset(df)

    # walk-forward CV; use last fold for calibration
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, calib_idx = list(tscv.split(X))[-1]
    booster = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", tree_method="hist",
    )
    booster.fit(X.iloc[train_idx], y.iloc[train_idx])
    raw_prob = booster.predict_proba(X.iloc[calib_idx])[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_prob, y.iloc[calib_idx])

    out_dir.mkdir(parents=True, exist_ok=True)
    model_version = hashlib.sha256(booster.get_booster().save_raw()).hexdigest()[:12]
    booster.save_model(out_dir / f"xgb_{model_version}.json")
    with open(out_dir / f"calib_{model_version}.pkl", "wb") as fh:
        pickle.dump({"isotonic": isotonic, "feature_order": list(X.columns)}, fh)

    _register(model_version, X.index.min(), X.index.max(), out_dir)
    return model_version


def _register(model_version: str, start, end, out_dir: Path) -> None:
    engine = sa.create_engine("sqlite:///data/state.db")
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT OR REPLACE INTO model_versions "
            "(ml_model_version, path, training_window_start, training_window_end, "
            " calibration_method, deployed_at) "
            "VALUES (:mv, :path, :s, :e, :cm, :ts)"
        ), {
            "mv": model_version,
            "path": str(out_dir / f"xgb_{model_version}.json"),
            "s": start, "e": end,
            "cm": "isotonic",
            "ts": datetime.now(tz=timezone.utc),
        })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=Path("models"), type=Path)
    args = ap.parse_args()
    mv = train(args.data, args.out)
    print(f"Trained model_version={mv}")
```

- [ ] **Step 3: Wire real `_run_model` and `load` classmethod in `src/models/xgb_predictor.py`**

```python
@classmethod
def load(cls, model_path: str, calib_path: str) -> "XGBPredictor":
    import pickle
    import xgboost as xgb
    booster = xgb.XGBClassifier()
    booster.load_model(model_path)
    with open(calib_path, "rb") as fh:
        meta = pickle.load(fh)
    version = Path(model_path).stem.removeprefix("xgb_")
    return cls(
        _model=booster,
        _calibrator=meta["isotonic"],
        _feature_order=tuple(meta["feature_order"]),
        ml_model_version=version,
    )

def _run_model(self, features: dict[str, Any]) -> float:
    from scripts.train_xgb import _flatten
    flat = _flatten(features)
    row = [flat.get(k, 0.0) for k in self._feature_order]
    raw = self._model.predict_proba([row])[0, 1]
    return float(self._calibrator.transform([raw])[0])
```

(Move `_flatten` to `src/models/xgb_predictor.py` if you prefer avoiding script imports from library code. Either works — keep consistent.)

- [ ] **Step 4: Test real path with a tiny fixture**

```python
# tests/unit/models/test_xgb_predictor_real.py
import pytest
from pathlib import Path


@pytest.mark.slow
def test_trained_predictor_roundtrip(tmp_path: Path):
    """Offline integration: trains on the fixture CSV, loads, predicts."""
    from scripts.train_xgb import train
    import pandas as pd
    # reuse the 1h fixture from Plan 1
    fixture = Path("tests/fixtures/ethusdt_1h_sample.csv")
    out_dir = tmp_path / "models"
    mv = train(fixture, out_dir)
    from models.xgb_predictor import XGBPredictor
    pred = XGBPredictor.load(
        str(out_dir / f"xgb_{mv}.json"),
        str(out_dir / f"calib_{mv}.pkl"),
    )
    import asyncio
    bundle = asyncio.run(pred.predict({"smc": {}, "fib": {}, "liquidity": {},
                                       "divergence": {}, "funding": {}, "confidence": {}}))
    assert 0 <= bundle.prob_up <= 1
```

Mark `slow` so it doesn't run on every `pytest` — opt-in via `-m slow`.

- [ ] **Step 5: Run fast tests + commit**

```bash
pytest -q
git add scripts/train_xgb.py src/models/xgb_predictor.py tests/unit/models/test_xgb_predictor_real.py .gitignore models/
git commit -m "feat(models): XGBoost + isotonic calibration (resolves spec Q1 pre-live blocker)"
```

---

## Task 15: GemmaContextProvider (Ollama + instructor)

**Files:**
- Create: `config/prompts/context_provider.md`
- Create: `src/models/llm/__init__.py`, `src/models/llm/ollama_client.py`, `src/models/llm/gemma_context.py`
- Test: `tests/unit/models/test_gemma_context.py`, `tests/contracts/test_prompt_versioning.py`

Rationale: Spec §4.3 + §9.3. Emits boolean flags only — never `prob_up`. Prompt version = `sha256(prompt_file_bytes)` enforced by contract test.

- [ ] **Step 1: Write the prompt file** (terse; concrete revisions land as the bot runs)

```markdown
<!-- config/prompts/context_provider.md -->
You are a structural market-context classifier. Given the feature snapshot JSON,
emit flags only — never probabilities.

Return a JSON object matching this schema:
  - context_veto: bool   # true if the market regime is hostile to our ML signal
  - veto_reason: string | null
  - structural_flags: list of strings, each one a short structural tag

Guardrails:
  - Never output numeric probabilities.
  - Never propose an entry, stop, or size.
  - If uncertain, set context_veto=false and leave structural_flags empty.
```

- [ ] **Step 2: Implement `OllamaClient` (basic — priority queue lands in Plan 3)**

```python
# src/models/llm/ollama_client.py
"""Minimal Ollama client wrapper — Plan-2 scope.

Plan-3 replaces this with a priority-queue-aware client (spec §4.6.1).
"""
from __future__ import annotations

import asyncio
from typing import Any

import instructor
import ollama
from pydantic import BaseModel


class OllamaClient:
    def __init__(self, model: str = "gemma2:4b", host: str = "http://localhost:11434") -> None:
        self._model = model
        self._client = instructor.from_openai(
            ollama.AsyncClient(host=host),
            mode=instructor.Mode.JSON,
        )
        self._sem = asyncio.Semaphore(1)

    async def complete(self, prompt: str, schema: type[BaseModel], **kw: Any) -> BaseModel:
        async with self._sem:
            return await self._client.chat.completions.create(
                model=self._model,
                response_model=schema,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                **kw,
            )
```

- [ ] **Step 3: Implement `GemmaContextProvider`**

```python
# src/models/llm/gemma_context.py
"""GemmaContextProvider — spec §4.3. Emits flags; never probabilities."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.base import LLMContextFlags
from models.llm.ollama_client import OllamaClient


PROMPT_PATH = Path("config/prompts/context_provider.md")


def _load_prompt() -> tuple[str, str]:
    body = PROMPT_PATH.read_bytes()
    return body.decode(), hashlib.sha256(body).hexdigest()


@dataclass
class GemmaContextProvider:
    client: OllamaClient
    prompt_version: str = ""
    _system: str = ""

    def __post_init__(self) -> None:
        self._system, self.prompt_version = _load_prompt()

    async def flags(self, features: dict[str, Any]) -> LLMContextFlags:
        prompt = f"{self._system}\n\nFeatures:\n{json.dumps(features, default=str)[:8000]}"
        return await self.client.complete(prompt, schema=LLMContextFlags)
```

- [ ] **Step 4: Prompt-version contract test**

```python
# tests/contracts/test_prompt_versioning.py
import hashlib
from pathlib import Path

from models.llm.gemma_context import GemmaContextProvider, PROMPT_PATH


def test_gemma_prompt_version_matches_file_hash():
    expected = hashlib.sha256(Path(PROMPT_PATH).read_bytes()).hexdigest()
    # Construct the provider with a dummy client; __post_init__ sets prompt_version
    provider = GemmaContextProvider(client=None)      # type: ignore[arg-type]
    assert provider.prompt_version == expected
```

- [ ] **Step 5: Unit test with mocked Ollama**

```python
# tests/unit/models/test_gemma_context.py
import pytest
from unittest.mock import AsyncMock

from models.base import LLMContextFlags
from models.llm.gemma_context import GemmaContextProvider


@pytest.mark.asyncio
async def test_returns_flags(monkeypatch):
    fake_client = AsyncMock()
    fake_client.complete = AsyncMock(return_value=LLMContextFlags(
        context_veto=False, veto_reason=None, structural_flags=["trend"],
    ))
    provider = GemmaContextProvider(client=fake_client)
    out = await provider.flags({"smc": {}})
    assert out.structural_flags == ["trend"]
```

- [ ] **Step 6: Run tests, commit**

```bash
pytest tests/unit/models/test_gemma_context.py tests/contracts/test_prompt_versioning.py -v
git add config/prompts/ src/models/llm/ tests/unit/models/test_gemma_context.py tests/contracts/test_prompt_versioning.py
git commit -m "feat(llm): GemmaContextProvider + prompt_version hash contract"
```

---

## Task 16: Ensemble

**Files:**
- Create: `src/decision/ensemble.py`
- Test: `tests/unit/decision/test_ensemble.py`

Rationale: Spec §4.3. ML `prob_up` + LLM boolean flags → single `PredictionBundle`. Spec §7.1 enforces: LLM never contributes a probability.

- [ ] **Step 1: Test**

```python
# tests/unit/decision/test_ensemble.py
import pytest
from unittest.mock import AsyncMock

from decision.ensemble import Ensemble
from models.base import LLMContextFlags, PredictionBundle


def _bundle(prob_up=0.7):
    return PredictionBundle(
        direction="long", prob_up=prob_up, horizon_bars=4, size_multiplier=1.0,
        feature_snapshot_hash="h", feature_registry_version="1.0.0",
        ml_model_version="m", llm_prompt_version="none", predictions_detail={},
    )


@pytest.mark.asyncio
async def test_no_veto_passes_through():
    ml = AsyncMock(); ml.predict = AsyncMock(return_value=_bundle(0.7))
    llm = AsyncMock(); llm.flags = AsyncMock(return_value=LLMContextFlags(
        context_veto=False, veto_reason=None, structural_flags=[]))
    llm.prompt_version = "v1"
    ens = Ensemble(ml=ml, llm_ctx=llm)
    out = await ens.predict({})
    assert out.prob_up == 0.7 and out.size_multiplier == 1.0
    assert out.llm_prompt_version == "v1"


@pytest.mark.asyncio
async def test_veto_zeros_size_multiplier_but_keeps_prob():
    ml = AsyncMock(); ml.predict = AsyncMock(return_value=_bundle(0.7))
    llm = AsyncMock(); llm.flags = AsyncMock(return_value=LLMContextFlags(
        context_veto=True, veto_reason="regime_mismatch", structural_flags=[]))
    llm.prompt_version = "v1"
    ens = Ensemble(ml=ml, llm_ctx=llm)
    out = await ens.predict({})
    assert out.size_multiplier == 0.0
    assert out.veto_reason == "regime_mismatch"
    assert out.prob_up == 0.7              # prob untouched
```

- [ ] **Step 2: Implement**

```python
# src/decision/ensemble.py
"""Ensemble — spec §4.3. ML prob + LLM flags → PredictionBundle.

LLM never writes a probability (§7.1). On veto, we set size_multiplier=0
but leave prob_up intact so audit logs show what the ML model said.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.base import LLMContextProvider, PredictionBundle, Predictor


@dataclass
class Ensemble:
    ml: Predictor
    llm_ctx: LLMContextProvider

    async def predict(self, features: dict[str, Any]) -> PredictionBundle:
        ml_pred = await self.ml.predict(features)
        flags = await self.llm_ctx.flags(features)
        update: dict[str, Any] = {"llm_prompt_version": self.llm_ctx.prompt_version}
        if flags.context_veto:
            update["size_multiplier"] = 0.0
            update["veto_reason"] = flags.veto_reason
        return ml_pred.model_copy(update=update)
```

- [ ] **Step 3: Run, commit**

```bash
git add src/decision/ensemble.py tests/unit/decision/test_ensemble.py
git commit -m "feat(decision): Ensemble — ML prob + LLM boolean flags (spec §4.3, §7.1)"
```

---

## Task 17: Orchestrator + CLI

**Files:**
- Create: `src/orchestrator.py`, `src/cli.py`
- Test: `tests/unit/test_orchestrator.py` (boot sequence only — E2E is Task 18)

Rationale: Spec §4.8. Single asyncio TaskGroup. Boot sequence: HALT check → migrations → reconcile → ping Ollama → start tasks. Plan 2 ships a minimal loop; apscheduler hourly job added in Task 18's E2E dry run.

- [ ] **Step 1: Test boot sequence exits on HALT file**

```python
# tests/unit/test_orchestrator.py
from pathlib import Path

import pytest

from orchestrator import Orchestrator, OrchestratorConfig


@pytest.mark.asyncio
async def test_halt_file_aborts_boot(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "HALT").write_text("manual")
    orch = Orchestrator(OrchestratorConfig(sqlite_path=str(tmp_path / "state.db")))
    with pytest.raises(SystemExit) as e:
        await orch.boot()
    assert "HALT" in str(e.value)
```

- [ ] **Step 2: Implement a minimal orchestrator**

```python
# src/orchestrator.py
"""Orchestrator — spec §4.8. Single asyncio TaskGroup main.

Plan-2 minimum: boot sequence + manual-trigger scan method. Hourly
scheduler + Telegram + event consumer lifecycle land in Plan 3.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
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
        from datetime import datetime, timezone
        while True:
            with self.engine.begin() as conn:
                conn.execute(sa.text(
                    "INSERT INTO heartbeat (ts, trace_id) VALUES (:ts, :tid)"
                ), {"ts": datetime.now(tz=timezone.utc), "tid": "boot"})
            await asyncio.sleep(60)
```

- [ ] **Step 3: Minimal CLI**

```python
# src/cli.py
"""CLI entry — `python -m src.cli`.

Not a user UI. Ops-only.
"""
from __future__ import annotations

import argparse
import asyncio

import structlog

from orchestrator import Orchestrator, OrchestratorConfig


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def main() -> None:
    _configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default="data/state.db")
    args = ap.parse_args()
    orch = Orchestrator(OrchestratorConfig(sqlite_path=args.sqlite))
    asyncio.run(orch.run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, commit**

```bash
pytest tests/unit/test_orchestrator.py -v
git add src/orchestrator.py src/cli.py tests/unit/test_orchestrator.py
git commit -m "feat(orchestrator): TaskGroup boot sequence + structlog CLI entry (spec §4.8)"
```

---

## Task 18: E2E smoke pipeline

**Files:**
- Create: `tests/e2e/test_smoke_pipeline.py`

Rationale: Exercise the full wire: `FeatureRegistry` → `XGBPredictor.stub + Ensemble (LLM flags=mock)` → `ThresholdPolicy` → `RiskPipeline` → `PaperBroker` → `BrokerEventRepo` → `rebuild_positions`. Proves the scaffold holds together.

- [ ] **Step 1: Write the test**

```python
# tests/e2e/test_smoke_pipeline.py
import asyncio
import random
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd
import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from decision.ensemble import Ensemble
from decision.policy import ThresholdPolicy
from decision.proposal import PortfolioSnapshot
from decision.risk.checks import (
    DailyLossKillSwitch, MandatoryStopLoss, MaxConcurrentPositions, SpreadGate,
)
from decision.risk.pipeline import RiskPipeline
from decision.sizing import FixedFractionalSizer
from execution.base import Order
from execution.paper_broker import PaperBroker, PaperBrokerConfig
from execution.repositories import BrokerEventRepo, ProposalRepo
from execution.replay import rebuild_positions
from features.registry import build_default_registry
from models.base import LLMContextFlags
from models.xgb_predictor import XGBPredictor


@pytest.mark.asyncio
async def test_smoke_pipeline_runs_end_to_end(tmp_path: Path):
    # --- state ---
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    engine = sa.create_engine(f"sqlite:///{db}")
    event_repo = BrokerEventRepo(engine)
    proposal_repo = ProposalRepo(engine)

    # --- data (fixture CSV from Plan 1) ---
    df = pd.read_csv("tests/fixtures/ethusdt_1h_sample.csv",
                     parse_dates=["open_time"]).set_index("open_time")
    as_of = df.index[-5]
    registry = build_default_registry()
    features = registry.compute_all(df, as_of=as_of)

    # --- predictor + ensemble ---
    ml = XGBPredictor.stub(prob_up=0.65, ml_model_version="stub-v0")
    llm = AsyncMock()
    llm.flags = AsyncMock(return_value=LLMContextFlags(
        context_veto=False, veto_reason=None, structural_flags=[]))
    llm.prompt_version = "mock-v0"
    ensemble = Ensemble(ml=ml, llm_ctx=llm)
    bundle = await ensemble.predict(features)

    # --- policy ---
    policy = ThresholdPolicy(
        long_threshold=0.55, short_threshold=0.45,
        symbol="ETHUSDT",
        mid_provider=lambda s: float(df["close"].iloc[-5]),
        atr_provider=lambda s: float(df["close"].iloc[-5]) * 0.005,
    )
    proposal = await policy.propose(features, bundle,
                                    PortfolioSnapshot(equity_usdt=10_000,
                                                      open_positions={},
                                                      day_pnl_r=0.0, consecutive_wins=0))
    assert proposal is not None

    # --- risk ---
    risk = RiskPipeline([
        MandatoryStopLoss(),
        SpreadGate(max_bps=20.0, spread_provider=lambda s: 5.0),
        DailyLossKillSwitch(threshold_r=-2.0),
        MaxConcurrentPositions(cap=3),
    ])
    results = risk.evaluate(proposal, PortfolioSnapshot(equity_usdt=10_000,
                                                         open_positions={},
                                                         day_pnl_r=0.0, consecutive_wins=0))
    assert RiskPipeline.is_accepted(results)
    proposal = proposal.model_copy(update={"risk_checks": results})
    proposal_repo.insert(proposal, accepted=True)

    # --- size ---
    sized = FixedFractionalSizer(fraction=0.0025).size(
        equity_usdt=10_000, entry=proposal.entry, stop_loss=proposal.stop_loss,
    )
    assert sized > 0

    # --- execute ---
    broker = PaperBroker(
        cfg=PaperBrokerConfig(latency_ms_mean=5, latency_ms_stdev=1,
                              partial_fill_prob=0.0, rejection_prob=0.0),
        rng=random.Random(7),
        mid_provider=lambda s: proposal.entry,
    )
    side = "buy" if proposal.direction == "long" else "sell"
    order_id = await broker.submit(Order(
        client_order_id=proposal.proposal_id,
        symbol=proposal.symbol, side=side, type="market", qty=sized,
    ))

    # --- drain events, persist, rebuild positions ---
    events = []
    async for e in broker.events():
        events.append(e)
        event_repo.insert(e)
        if e.kind == "filled":
            break
    snap = rebuild_positions(event_repo.all())
    assert "ETHUSDT" in snap
    assert snap["ETHUSDT"].qty == sized if side == "buy" else -sized
```

- [ ] **Step 2: Run the smoke test**

```bash
pytest tests/e2e/test_smoke_pipeline.py -v
```
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_smoke_pipeline.py
git commit -m "test(e2e): smoke pipeline — features → ensemble → policy → risk → paper broker → SQLite"
```

---

## Task 19: Final verification + handoff

Rationale: Close out Plan 2. Verify green baseline; write handoff STATUS.md; summarize.

- [ ] **Step 1: Full test suite**

```bash
pytest -q
```
Expected: `N passed, 0 failed` where `N ≥ 154 + Plan-2 additions (~30+)`. Record the exact number.

- [ ] **Step 2: mypy strict over new src modules**

```bash
venv/bin/python -m mypy src/execution src/decision src/models src/data
```
Expected: 0 errors on new Plan-2 files. Legacy feature files' pre-existing 71 errors are unaffected (Plan 3 tightens).

- [ ] **Step 3: Alembic current = head**

```bash
source venv/bin/activate && alembic current && alembic heads
```
Expected: both show `15fdbaffd2bf`.

- [ ] **Step 4: `grep` — no `_legacy` references remain**

```bash
grep -r "decision._legacy" src tests || echo clean
```
Expected: `clean`.

- [ ] **Step 5: Write Plan-2 STATUS.md**

Create `docs/superpowers/plans/2026-04-18-pivot-plan2-STATUS.md` following the Plan-1 STATUS template, listing:
- 19 tasks + commit hashes
- Test count before/after
- Files added under `src/`, `tests/`, `config/prompts/`, `scripts/`
- What is NOT done (intentional — Plan 3 scope): apscheduler hourly job wiring, Telegram bot, ChatLLM, FeatureDriftMonitor, contract suite for ReplayBroker/LiveBroker, walk-forward backtest + Deflated Sharpe, Pre-Live Gate module, HALT fire-drill, heartbeat watchdog

- [ ] **Step 6: Report handoff**

Draft a one-screen summary naming:
- What is green: E2E smoke test, Ensemble, PaperBroker contract, calibrated XGB model optional via `scripts/train_xgb.py`
- What is NOT done (Plan 3)
- Propose starting Plan 3 or pausing

---

## Ready-to-hand-off invariants (must be true before starting Plan 3)

- [ ] Full test suite green.
- [ ] `tests/e2e/test_smoke_pipeline.py` passes without skips.
- [ ] `broker_events.event_id` still the sole PK (no regression on spec §8.3).
- [ ] `rebuild_positions` idempotent on duplicate event_ids (Task 2 test is authoritative).
- [ ] `GemmaContextProvider.prompt_version == sha256(config/prompts/context_provider.md)` via contract test.
- [ ] No LLM path in code produces a `prob_up` value (grep `prob_up` across `src/models/llm/` — must only appear in comments).
- [ ] `models/xgb_*.json` + `models/calib_*.pkl` not committed to git (in `.gitignore`).
- [ ] Plan-1 invariants still hold: 6 features in stable order, no-repainting tests green on all seeds.

Any un-ticked box blocks starting Plan 3.
