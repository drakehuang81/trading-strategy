# Plan 5B-2 — ReplayBroker + Broker Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic `ReplayBroker` that walks through historical klines + funding parquet under an external clock, sharing the cost model (slippage + fees) with `PaperBroker`. Also add a `LiveBroker` stub (refuses orders) so wiring's `broker_kind=live` flag has somewhere to go. Both broker classes added to the existing `tests/contracts/test_broker_contract.py` shared suite. Wiring gains a `broker_kind: paper|replay|live` selector.

**Architecture:** Extract slippage + fee computation from `PaperBroker` into `src/execution/cost_model.py` (3 small pure functions). `PaperBroker` and the new `ReplayBroker` both call them — guarantees backtest cost model = paper cost model = single source of truth (spec §7.2 "PaperBroker's cost model IS backtest's cost model"). `ReplayBroker` is the simplest possible Broker: synchronous fill at current bar's close ± slippage; no latency / partial / rejection (all are "realism noise" appropriate for paper, not for backtest where we want determinism). External clock via `set_time(ts)` lets the Plan 5B-3 backtest harness drive the loop. `LiveBroker` is a stub that raises on `submit()` and returns empty state elsewhere — safety brake until Plan 5D wires real Binance.

**Tech Stack:** Python 3.11, asyncio, pandas, pytest with `asyncio_mode=auto`. No new dependencies.

**Decisions baked in:**
- **Cost model is shared**, not parallel implementations. Guarantees backtest fills match paper fills bit-for-bit (spec §7.2).
- **ReplayBroker has no latency, no partial fill, no rejection.** Backtest determinism > realism noise. PaperBroker's randomness is for paper ops where we want to surface real-world friction.
- **External clock**: backtest harness owns the loop; `ReplayBroker.set_time(ts)` is the only way time advances.
- **Fill price = current bar `close` ± slippage_bps.** Single-bar approximation; assumes "trade at this bar's close, see result on next bar's open" semantics common in backtest frameworks.
- **`broker_kind` config field** added to `OrchestratorConfig`. Default stays `paper`; setting `replay` or `live` requires explicit opt-in.
- **No new SQL schema** — `model_versions` and `proposals` are unchanged. Replay backtest's results land in `backtest_runs` (Plan 5B-3 territory).

**Out of Plan 5B-2 scope (deferred):**
- The actual backtest harness (Plan 5B-3) — this plan only delivers the broker.
- Real Binance integration (Plan 5D).
- LiveConfirmViaTelegram (Plan 5D).
- Replay funding tick scheduling logic for backtest harness — `ReplayBroker.set_time` emits funding events for any 8h boundaries crossed, but the harness loop is Plan 5B-3.

---

## File map

### Created
- `src/execution/cost_model.py` — `slippage_fill_price`, `taker_or_maker_fee`, `compute_fill_price_and_fee` (3 pure functions)
- `src/execution/replay_broker.py` — `ReplayBroker` class
- `src/execution/live_broker.py` — `LiveBroker` stub class
- `tests/unit/execution/test_cost_model.py`
- `tests/unit/execution/test_replay_broker.py`
- `tests/unit/execution/test_live_broker.py`
- `docs/superpowers/plans/2026-04-26-pivot-plan5b2-STATUS.md` (handoff)

### Modified
- `src/execution/paper_broker.py` — `_simulate_fill` and `_emit` use `cost_model.*` instead of inline math.
- `tests/contracts/test_broker_contract.py` — add `replay_broker` fixture + `TestReplayBrokerContract`. (LiveBroker NOT added to the contract suite because it can't `fill`.)
- `src/orchestrator.py` — `OrchestratorConfig` gains `broker_kind: Literal["paper","replay","live"] = "paper"`; `replay_kline_path: str = "data/history/ETHUSDT_1h.parquet"`; `replay_funding_path: str = "data/funding/ETHUSDT.parquet"`.
- `src/wiring.py` — branch on `cfg.broker_kind` to construct the right broker.

### Untouched (verified intentionally)
- `src/execution/base.py` — Broker / BrokerEvent / Position protocols stable.
- `src/execution/replay.py` — `rebuild_positions` is unrelated (event-stream replay, not a broker).
- `src/execution/repositories.py`, `src/execution/reconcile.py`, `src/execution/tick_recorder.py` — unrelated.
- Existing PaperBroker tests — should remain green after refactor.

---

## Task 1: Extract `cost_model.py`; refactor PaperBroker to use it

**Why first:** Establishes the shared cost-model contract that ReplayBroker will reuse. Refactoring PaperBroker now (rather than after ReplayBroker exists) means we don't have to coordinate two changes later. Pure refactor — no behavior change.

**Files:**
- Create: `src/execution/cost_model.py`
- Create: `tests/unit/execution/test_cost_model.py`
- Modify: `src/execution/paper_broker.py:80-84` (slippage formula) and `:124-127` (fee formula)

- [ ] **Step 1: Write the failing test for cost_model**

```python
# tests/unit/execution/test_cost_model.py
"""Pure cost-model functions — Plan 5B-2 Task 1."""
from __future__ import annotations

import math

import pytest

from execution.cost_model import (
    slippage_fill_price,
    taker_or_maker_fee,
    SlippageConfig,
)


def test_buy_slippage_pushes_price_up():
    cfg = SlippageConfig(slippage_bps_base=1.0, slippage_bps_per_adv_unit=20.0, adv_stub=1000.0)
    # qty 100, base+20*0.1 = 3 bps total -> mid * 1.0003
    fill = slippage_fill_price(mid=2000.0, side="buy", qty=100.0, cfg=cfg)
    assert math.isclose(fill, 2000.0 * 1.0003, rel_tol=1e-12)


def test_sell_slippage_pushes_price_down():
    cfg = SlippageConfig(slippage_bps_base=1.0, slippage_bps_per_adv_unit=20.0, adv_stub=1000.0)
    fill = slippage_fill_price(mid=2000.0, side="sell", qty=100.0, cfg=cfg)
    assert math.isclose(fill, 2000.0 * 0.9997, rel_tol=1e-12)


def test_zero_qty_gives_only_base_slippage():
    cfg = SlippageConfig(slippage_bps_base=2.0, slippage_bps_per_adv_unit=10.0, adv_stub=1000.0)
    fill = slippage_fill_price(mid=2000.0, side="buy", qty=0.0, cfg=cfg)
    assert math.isclose(fill, 2000.0 * 1.0002, rel_tol=1e-12)


def test_taker_fee_for_market_order():
    fee = taker_or_maker_fee(price=2000.0, qty=0.5, order_type="market",
                             taker_bps=5.0, maker_bps=2.0)
    # 2000 * 0.5 * 5/10000 = 0.5
    assert math.isclose(fee, 0.5, rel_tol=1e-12)


def test_maker_fee_for_limit_order():
    fee = taker_or_maker_fee(price=2000.0, qty=0.5, order_type="limit",
                             taker_bps=5.0, maker_bps=2.0)
    # 2000 * 0.5 * 2/10000 = 0.2
    assert math.isclose(fee, 0.2, rel_tol=1e-12)


def test_fee_uses_abs_qty_for_short_fills():
    fee = taker_or_maker_fee(price=2000.0, qty=-0.5, order_type="market",
                             taker_bps=5.0, maker_bps=2.0)
    assert math.isclose(fee, 0.5, rel_tol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/execution/test_cost_model.py -v`
Expected: ImportError on `execution.cost_model`.

- [ ] **Step 3: Implement cost_model.py**

```python
# src/execution/cost_model.py
"""Shared slippage + fee math — Plan 5B-2 Task 1.

Single source of truth so PaperBroker (live paper) and ReplayBroker
(backtest) produce IDENTICAL fill prices for the same (mid, side, qty,
config) tuple. Spec §7.2: "PaperBroker's cost model IS backtest's
cost model — single source of truth."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SlippageConfig:
    slippage_bps_base: float
    slippage_bps_per_adv_unit: float
    adv_stub: float


def slippage_fill_price(*, mid: float, side: Literal["buy", "sell"],
                        qty: float, cfg: SlippageConfig) -> float:
    """Apply linear slippage: base + per-unit-of-ADV.

    Buy fills at higher price (worse), sell at lower price (worse).
    """
    sign = 1 if side == "buy" else -1
    slip_bps = cfg.slippage_bps_base + cfg.slippage_bps_per_adv_unit * (qty / cfg.adv_stub)
    return mid * (1 + sign * slip_bps / 10_000)


def taker_or_maker_fee(*, price: float, qty: float,
                       order_type: Literal["market", "limit"],
                       taker_bps: float, maker_bps: float) -> float:
    """Absolute fee in quote currency. Uses abs(qty) so short fills
    (qty < 0) charge symmetric fees."""
    fee_bps = taker_bps if order_type == "market" else maker_bps
    return price * abs(qty) * fee_bps / 10_000.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/execution/test_cost_model.py -v`
Expected: 6 passed.

- [ ] **Step 5: Refactor PaperBroker to use cost_model**

In `src/execution/paper_broker.py`, add the import:
```python
from execution.cost_model import (
    SlippageConfig,
    slippage_fill_price,
    taker_or_maker_fee,
)
```

Replace the slippage block in `_simulate_fill` (currently lines 79-84):
```python
        mid = self.mid_provider(order.symbol)
        slip_cfg = SlippageConfig(
            slippage_bps_base=self.cfg.slippage_bps_base,
            slippage_bps_per_adv_unit=self.cfg.slippage_bps_per_adv_unit,
            adv_stub=self.cfg.adv_stub,
        )
        fill_price = slippage_fill_price(
            mid=mid, side=order.side, qty=order.qty, cfg=slip_cfg,
        )
        sign = 1 if order.side == "buy" else -1
```

Replace the fee block in `_emit` (currently lines 124-127):
```python
        fee = None
        if price is not None and qty is not None:
            fee = taker_or_maker_fee(
                price=price, qty=qty,
                order_type=order.type if order else "market",
                taker_bps=self.cfg.taker_bps,
                maker_bps=self.cfg.maker_bps,
            )
```

- [ ] **Step 6: Run PaperBroker tests + full suite to confirm zero behavior change**

Run: `pytest tests/unit/execution/test_paper_broker.py -v && pytest -q`
Expected: All PaperBroker tests pass identically (same fill prices, same fees — pure refactor). 302 total (296 + 6 new).

If any PaperBroker test fails on a numeric value, you've changed behavior — diff the formula output against the original.

- [ ] **Step 7: Commit**

```bash
git add src/execution/cost_model.py tests/unit/execution/test_cost_model.py src/execution/paper_broker.py
git commit -m "refactor(execution): extract slippage+fee to cost_model; PaperBroker uses it"
```

---

## Task 2: `ReplayBroker`

**Why:** This is the meat. A deterministic broker the Plan 5B-3 backtest harness will drive via `set_time()` to walk historical klines bar by bar, getting reproducible fills using the same cost model as paper trading.

**Files:**
- Create: `src/execution/replay_broker.py`
- Create: `tests/unit/execution/test_replay_broker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/execution/test_replay_broker.py
"""ReplayBroker — deterministic discrete-time broker for backtest."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from execution.base import Order
from execution.paper_broker import PaperBrokerConfig
from execution.replay_broker import ReplayBroker


def _klines() -> pd.DataFrame:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(5)], name="open_time",
    )
    return pd.DataFrame({
        "open":   [3000.0, 3010.0, 3020.0, 3030.0, 3040.0],
        "high":   [3015.0, 3025.0, 3035.0, 3045.0, 3055.0],
        "low":    [2985.0, 2995.0, 3005.0, 3015.0, 3025.0],
        "close":  [3005.0, 3015.0, 3025.0, 3035.0, 3045.0],
        "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
    }, index=idx)


@pytest.fixture
def cfg():
    return PaperBrokerConfig(
        taker_bps=5.0, maker_bps=2.0,
        slippage_bps_base=1.0, slippage_bps_per_adv_unit=0.0, adv_stub=1000.0,
    )


@pytest.mark.asyncio
async def test_set_time_required_before_submit(cfg):
    rb = ReplayBroker(cfg=cfg, klines=_klines())
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
    with pytest.raises(RuntimeError, match="set_time"):
        await rb.submit(order)


@pytest.mark.asyncio
async def test_submit_fills_at_current_bar_close_with_slippage(cfg):
    klines = _klines()
    rb = ReplayBroker(cfg=cfg, klines=klines)
    rb.set_time(klines.index[2])         # 3rd bar, close = 3025.0
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
    await rb.submit(order)

    seen = []
    for _ in range(2):
        e = await asyncio.wait_for(rb._queue.get(), timeout=0.5)
        seen.append(e)
    assert seen[0].kind == "submitted"
    assert seen[1].kind == "filled"
    # 1 bps slippage on a buy: 3025 * 1.0001 = 3025.3025
    assert abs(seen[1].fill_price - 3025.3025) < 1e-9
    assert seen[1].fill_qty == 0.1   # buy is positive
    # fee = 3025.3025 * 0.1 * 5/10000 = 0.15126512...
    assert abs(seen[1].fee - 3025.3025 * 0.1 * 5.0 / 10_000) < 1e-9


@pytest.mark.asyncio
async def test_sell_signs_fill_qty_negative(cfg):
    klines = _klines()
    rb = ReplayBroker(cfg=cfg, klines=klines)
    rb.set_time(klines.index[0])
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="sell",
                  type="market", qty=0.05)
    await rb.submit(order)

    seen = []
    for _ in range(2):
        seen.append(await asyncio.wait_for(rb._queue.get(), timeout=0.5))
    assert seen[1].fill_qty == -0.05


@pytest.mark.asyncio
async def test_set_time_advances_funding_for_open_positions(cfg, tmp_path):
    klines = _klines()
    base = klines.index[0]
    # Funding tick at base + 0h (one tick that should fire when set_time crosses it)
    funding = pd.DataFrame(
        {"funding_rate": [0.0001]},
        index=pd.DatetimeIndex([base], name="ts"),
    )
    rb = ReplayBroker(cfg=cfg, klines=klines, funding=funding)
    rb.set_time(base - timedelta(seconds=1))  # before the tick
    # Open a position so funding has something to charge
    rb.set_time(base - timedelta(seconds=1))  # idempotent re-set
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
    rb.set_time(base - timedelta(seconds=1))
    await rb.submit(order)
    # Drain submit/fill events
    for _ in range(2):
        await asyncio.wait_for(rb._queue.get(), timeout=0.5)
    # Now advance past the funding tick
    rb.set_time(base + timedelta(hours=1))
    funding_event = await asyncio.wait_for(rb._queue.get(), timeout=0.5)
    assert funding_event.kind == "funding_charged"
    assert funding_event.symbol == "ETHUSDT"


@pytest.mark.asyncio
async def test_positions_aggregate_fills(cfg):
    klines = _klines()
    rb = ReplayBroker(cfg=cfg, klines=klines)
    rb.set_time(klines.index[0])
    o1 = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
               type="market", qty=0.1)
    o2 = Order(client_order_id="c2", symbol="ETHUSDT", side="buy",
               type="market", qty=0.05)
    await rb.submit(o1)
    await rb.submit(o2)
    # Drain 4 events
    for _ in range(4):
        await asyncio.wait_for(rb._queue.get(), timeout=0.5)
    pos = await rb.positions()
    assert len(pos) == 1
    assert abs(pos[0].qty - 0.15) < 1e-9


@pytest.mark.asyncio
async def test_balance_returns_initial_equity(cfg):
    rb = ReplayBroker(cfg=cfg, klines=_klines(), initial_equity_usdt=5_000.0)
    bal = await rb.balance()
    assert bal.equity_usdt == 5_000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/execution/test_replay_broker.py -v`
Expected: ImportError on `execution.replay_broker`.

- [ ] **Step 3: Implement ReplayBroker**

```python
# src/execution/replay_broker.py
"""ReplayBroker — deterministic discrete-time broker for backtest.

Differs from PaperBroker:
- No latency, no partial fill, no rejection — backtest determinism.
- External clock via `set_time(ts)`; `submit()` fills synchronously at
  the current bar's close ± slippage.
- Funding events emitted when `set_time` crosses 8h boundaries.

Same as PaperBroker:
- Slippage and fee computed via `execution.cost_model` (single source).
- Emits the same BrokerEvent shape ("submitted"/"filled"/"funding_charged").
- Same Broker Protocol surface (submit/cancel/positions/balance/events).
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

import pandas as pd

from execution.base import Balance, BrokerEvent, Order, OrderId, Position
from execution.cost_model import (
    SlippageConfig,
    slippage_fill_price,
    taker_or_maker_fee,
)
from execution.paper_broker import PaperBrokerConfig


@dataclass
class ReplayBroker:
    cfg: PaperBrokerConfig
    klines: pd.DataFrame
    funding: pd.DataFrame | None = None
    initial_equity_usdt: float = 10_000.0
    _current_ts: pd.Timestamp | None = None
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _orders: dict[OrderId, Order] = field(default_factory=dict)
    _positions: dict[str, Position] = field(default_factory=dict)
    _equity_usdt: float = 0.0

    def __post_init__(self) -> None:
        self._equity_usdt = self.initial_equity_usdt

    def set_time(self, ts: pd.Timestamp | datetime) -> None:
        """Advance internal clock. Emits funding events for any 8h
        boundaries strictly between previous time and new time."""
        new_ts = pd.Timestamp(ts)
        if self.funding is not None and self._current_ts is not None:
            window = self.funding[
                (self.funding.index > self._current_ts)
                & (self.funding.index <= new_ts)
            ]
            for ts_funding, row in window.iterrows():
                self._charge_funding(ts_funding, float(row["funding_rate"]))
        self._current_ts = new_ts

    async def submit(self, order: Order) -> OrderId:
        if self._current_ts is None:
            raise RuntimeError(
                "ReplayBroker.set_time must be called before submit"
            )
        order_id = str(uuid.uuid4())
        self._orders[order_id] = order
        await self._emit("submitted", order_id, order)

        mid = self._current_close(order.symbol)
        slip_cfg = SlippageConfig(
            slippage_bps_base=self.cfg.slippage_bps_base,
            slippage_bps_per_adv_unit=self.cfg.slippage_bps_per_adv_unit,
            adv_stub=self.cfg.adv_stub,
        )
        fill_price = slippage_fill_price(
            mid=mid, side=order.side, qty=order.qty, cfg=slip_cfg,
        )
        sign = 1 if order.side == "buy" else -1
        signed_qty = sign * order.qty
        await self._emit(
            "filled", order_id, order,
            price=fill_price, qty=signed_qty,
        )
        self._update_position(order.symbol, signed_qty, fill_price)
        return order_id

    async def cancel(self, order_id: OrderId) -> None:
        await self._emit("cancelled", order_id, self._orders.get(order_id))

    async def positions(self) -> list[Position]:
        return list(self._positions.values())

    async def balance(self) -> Balance:
        return Balance(equity_usdt=self._equity_usdt, free_usdt=self._equity_usdt)

    async def events(self) -> AsyncIterator[BrokerEvent]:
        while True:
            yield await self._queue.get()

    def _current_close(self, symbol: str) -> float:
        # Single-symbol replay; ignore symbol arg.
        if self._current_ts is None:
            raise RuntimeError("clock not set")
        sub = self.klines[self.klines.index <= self._current_ts]
        if sub.empty:
            raise RuntimeError(
                f"no kline at or before {self._current_ts}"
            )
        return float(sub["close"].iloc[-1])

    def _update_position(self, symbol: str, signed_qty: float,
                         fill_price: float) -> None:
        existing = self._positions.get(symbol)
        if existing is None:
            self._positions[symbol] = Position(
                symbol=symbol, qty=signed_qty, avg_entry=fill_price,
                opened_at=self._current_ts.to_pydatetime(),
                last_update_ts=self._current_ts.to_pydatetime(),
            )
            return
        new_qty = existing.qty + signed_qty
        if abs(new_qty) < 1e-12:
            del self._positions[symbol]
            return
        if (existing.qty > 0) == (signed_qty > 0):
            new_avg = (
                existing.avg_entry * existing.qty + fill_price * signed_qty
            ) / new_qty
        else:
            new_avg = existing.avg_entry  # reducing position keeps entry basis
        self._positions[symbol] = existing.model_copy(update={
            "qty": new_qty,
            "avg_entry": new_avg,
            "last_update_ts": self._current_ts.to_pydatetime(),
        })

    def _charge_funding(self, ts: pd.Timestamp, rate: float) -> None:
        for sym, pos in self._positions.items():
            mid = self._current_close(sym)
            fee = pos.qty * mid * rate
            self._queue.put_nowait(BrokerEvent(
                event_id=str(uuid.uuid4()),
                kind="funding_charged",
                order_id=f"funding_{sym}_{int(ts.timestamp())}",
                symbol=sym,
                ts_epoch_ms=int(ts.timestamp() * 1000),
                fee=fee, reason=f"rate={rate}",
            ))

    async def _emit(self, kind: str, order_id: OrderId, order: Order | None,
                    *, price: float | None = None, qty: float | None = None,
                    reason: str | None = None) -> None:
        fee = None
        if price is not None and qty is not None:
            fee = taker_or_maker_fee(
                price=price, qty=qty,
                order_type=order.type if order else "market",
                taker_bps=self.cfg.taker_bps,
                maker_bps=self.cfg.maker_bps,
            )
        event = BrokerEvent(
            event_id=str(uuid.uuid4()),
            kind=kind, order_id=order_id,
            symbol=order.symbol if order else "",
            ts_epoch_ms=int(self._current_ts.timestamp() * 1000)
                if self._current_ts is not None else 0,
            fill_price=price, fill_qty=qty, fee=fee, reason=reason,
        )
        await self._queue.put(event)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/execution/test_replay_broker.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 308 passed (302 + 6 new).

- [ ] **Step 6: Commit**

```bash
git add src/execution/replay_broker.py tests/unit/execution/test_replay_broker.py
git commit -m "feat(execution): ReplayBroker — deterministic broker driven by external clock"
```

---

## Task 3: `LiveBroker` stub

**Why:** Wiring's `broker_kind=live` flag needs somewhere to dispatch. A stub class that implements the Broker Protocol but refuses orders is the safety brake until Plan 5D wires the real Binance integration. Also documents the API contract for the future implementation.

**Files:**
- Create: `src/execution/live_broker.py`
- Create: `tests/unit/execution/test_live_broker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/execution/test_live_broker.py
"""LiveBroker stub — refuses orders; Plan 5D will replace."""
from __future__ import annotations

import asyncio

import pytest

from execution.base import Order
from execution.live_broker import LiveBroker, LiveBrokerNotImplemented


@pytest.mark.asyncio
async def test_submit_raises_not_implemented():
    lb = LiveBroker()
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
    with pytest.raises(LiveBrokerNotImplemented):
        await lb.submit(order)


@pytest.mark.asyncio
async def test_positions_returns_empty():
    lb = LiveBroker()
    assert await lb.positions() == []


@pytest.mark.asyncio
async def test_balance_returns_zero_equity():
    lb = LiveBroker()
    bal = await lb.balance()
    assert bal.equity_usdt == 0.0
    assert bal.free_usdt == 0.0


@pytest.mark.asyncio
async def test_cancel_raises_not_implemented():
    lb = LiveBroker()
    with pytest.raises(LiveBrokerNotImplemented):
        await lb.cancel("any_id")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/execution/test_live_broker.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement LiveBroker stub**

```python
# src/execution/live_broker.py
"""LiveBroker — Plan 5B-2 stub; Plan 5D will replace with real Binance.

Refuses submit/cancel; positions/balance return empty/zero so wiring code
that probes them at boot doesn't crash. The wiring path
`cfg.broker_kind="live"` MUST be gated behind Plan 5B-4's Pre-Live Gate
before this stub is replaced — running with this stub in production
would silently no-op every order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator

from execution.base import Balance, BrokerEvent, Order, OrderId, Position


class LiveBrokerNotImplemented(NotImplementedError):
    """Raised by LiveBroker stub when an order-flow method is called.

    Catch this in wiring/startup code if you want a friendly error
    message; otherwise let it propagate.
    """


@dataclass
class LiveBroker:
    """Stub. Plan 5D will implement against python-binance live endpoints."""

    async def submit(self, order: Order) -> OrderId:
        raise LiveBrokerNotImplemented(
            "LiveBroker.submit not implemented; gated behind Plan 5D + Pre-Live Gate"
        )

    async def cancel(self, order_id: OrderId) -> None:
        raise LiveBrokerNotImplemented(
            "LiveBroker.cancel not implemented; gated behind Plan 5D + Pre-Live Gate"
        )

    async def positions(self) -> list[Position]:
        return []

    async def balance(self) -> Balance:
        return Balance(equity_usdt=0.0, free_usdt=0.0)

    async def events(self) -> AsyncIterator[BrokerEvent]:
        # Empty stream — never yields. Caller must handle this.
        if False:
            yield   # type: ignore[unreachable]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/execution/test_live_broker.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/execution/live_broker.py tests/unit/execution/test_live_broker.py
git commit -m "feat(execution): LiveBroker stub — refuses orders, returns empty state"
```

---

## Task 4: Add ReplayBroker to broker contract test suite

**Why:** `tests/contracts/test_broker_contract.py` already runs a shared 2-test suite against PaperBroker. Plan 5B-2's ReplayBroker must pass the same suite to prove it's interchangeable. (LiveBroker is intentionally NOT added — its `submit` raises, so it can't pass `test_submit_yields_submitted_then_filled`.)

**Files:**
- Modify: `tests/contracts/test_broker_contract.py`

- [ ] **Step 1: Read the existing contract suite**

Open `tests/contracts/test_broker_contract.py` and confirm current shape: a `BrokerContractTests` base class with two tests, a `paper_broker` fixture, and `TestPaperBrokerContract(BrokerContractTests)` subclass. The fixture name is set via `broker_fixture = "paper_broker"` class attribute. Subclassing pattern is established.

- [ ] **Step 2: Add `replay_broker` fixture + `TestReplayBrokerContract`**

Append to `tests/contracts/test_broker_contract.py`:

```python
import pandas as pd
from datetime import datetime, timedelta, timezone

from execution.replay_broker import ReplayBroker


@pytest.fixture
def replay_broker():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    klines = pd.DataFrame({
        "open":   [3000.0, 3010.0, 3020.0],
        "high":   [3015.0, 3025.0, 3035.0],
        "low":    [2985.0, 2995.0, 3005.0],
        "close":  [3005.0, 3015.0, 3025.0],
        "volume": [1.0, 1.0, 1.0],
    }, index=pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(3)], name="open_time",
    ))
    rb = ReplayBroker(
        cfg=PaperBrokerConfig(
            taker_bps=5.0, maker_bps=2.0,
            slippage_bps_base=1.0, slippage_bps_per_adv_unit=0.0, adv_stub=1000.0,
        ),
        klines=klines,
    )
    rb.set_time(klines.index[1])
    return rb


class TestReplayBrokerContract(BrokerContractTests):
    broker_fixture = "replay_broker"
```

- [ ] **Step 3: Run contract tests**

Run: `pytest tests/contracts/test_broker_contract.py -v`
Expected: 4 passed (2 original × Paper + 2 new × Replay).

- [ ] **Step 4: Run full suite**

Run: `pytest -q`
Expected: 314 passed (308 + 4 contract = wait, 312; we already counted Replay's 6 unit tests). Actually: 312 = 308 Replay-unit-suite + 4 LiveBroker + 0 from this task because the 2 PaperContract tests were already in the 296 baseline. Let me recompute: Plan 5B-1 left 296. Task 1 added 6 cost_model tests → 302. Task 2 added 6 replay-broker unit tests → 308. Task 3 added 4 live-broker tests → 312. Task 4 adds 2 contract tests for Replay → 314.

Expected: 314 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/contracts/test_broker_contract.py
git commit -m "test(contract): ReplayBroker passes the shared Broker contract suite"
```

---

## Task 5: Wiring `broker_kind` switch + STATUS handoff

**Why:** Final integration. `OrchestratorConfig` gains `broker_kind` and two parquet paths; `wiring.py` branches to construct the right broker. Default stays `paper` (no behavior change for current callers).

**Files:**
- Modify: `src/orchestrator.py:27-47` — add 3 fields to `OrchestratorConfig`
- Modify: `src/wiring.py` — branch on `cfg.broker_kind`
- Create: `tests/unit/test_wiring_broker_kind.py`
- Create: `docs/superpowers/plans/2026-04-26-pivot-plan5b2-STATUS.md`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_wiring_broker_kind.py
"""Wiring switches broker by cfg.broker_kind — Plan 5B-2 Task 5."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import sqlalchemy as sa

from orchestrator import OrchestratorConfig
from wiring import build_scan_context


def _seed_klines(path: Path) -> None:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "open":   [3000.0] * 5, "high": [3010.0] * 5, "low": [2990.0] * 5,
        "close":  [3005.0] * 5, "volume": [1.0] * 5,
    }, index=pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(5)], name="open_time",
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _seed_funding(path: Path) -> None:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {"funding_rate": [0.0001] * 3},
        index=pd.DatetimeIndex(
            [base + timedelta(hours=i * 8) for i in range(3)], name="ts",
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _bootstrap_engine(cfg) -> sa.Engine:
    import alembic.command, alembic.config
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{cfg.sqlite_path}")
    alembic.command.upgrade(ac, "head")
    return sa.create_engine(f"sqlite:///{cfg.sqlite_path}")


@pytest.mark.asyncio
async def test_default_broker_kind_is_paper(tmp_path):
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
    )
    engine = _bootstrap_engine(cfg)
    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=pd.DataFrame())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        ctx, _ = await build_scan_context(cfg, engine)
    from execution.paper_broker import PaperBroker
    assert isinstance(ctx.broker, PaperBroker)


@pytest.mark.asyncio
async def test_broker_kind_replay_uses_replay_broker(tmp_path):
    kline_path = tmp_path / "klines.parquet"
    funding_path = tmp_path / "funding.parquet"
    _seed_klines(kline_path)
    _seed_funding(funding_path)
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        broker_kind="replay",
        replay_kline_path=str(kline_path),
        replay_funding_path=str(funding_path),
    )
    engine = _bootstrap_engine(cfg)
    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=pd.DataFrame())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        ctx, _ = await build_scan_context(cfg, engine)
    from execution.replay_broker import ReplayBroker
    assert isinstance(ctx.broker, ReplayBroker)


@pytest.mark.asyncio
async def test_broker_kind_live_uses_live_broker(tmp_path):
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        broker_kind="live",
    )
    engine = _bootstrap_engine(cfg)
    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=pd.DataFrame())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        ctx, _ = await build_scan_context(cfg, engine)
    from execution.live_broker import LiveBroker
    assert isinstance(ctx.broker, LiveBroker)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_wiring_broker_kind.py -v`
Expected: 3 failures — `OrchestratorConfig` has no `broker_kind` field; the test for replay reaches "PaperBroker" instance and fails the isinstance check.

- [ ] **Step 3: Add fields to `OrchestratorConfig`**

In `src/orchestrator.py`, near the `OrchestratorConfig` block (after `paper_broker_seed`):

```python
    broker_kind: Literal["paper", "replay", "live"] = "paper"
    replay_kline_path: str = "data/history/ETHUSDT_1h.parquet"
    replay_funding_path: str = "data/funding/ETHUSDT.parquet"
```

Also add the import at the top of the file if not already present:
```python
from typing import Any, Literal
```

(Read the existing top imports before editing — `Any` is already there; `Literal` may need to be added.)

- [ ] **Step 4: Add broker switch in wiring**

In `src/wiring.py`, find the existing `broker = PaperBroker(...)` block. Replace with:

```python
    if cfg.broker_kind == "paper":
        broker = PaperBroker(
            cfg=PaperBrokerConfig(),
            rng=rng,
            mid_provider=mid_provider,
        )
    elif cfg.broker_kind == "replay":
        from execution.replay_broker import ReplayBroker
        replay_klines = pd.read_parquet(cfg.replay_kline_path)
        replay_funding = (
            pd.read_parquet(cfg.replay_funding_path)
            if Path(cfg.replay_funding_path).exists()
            else None
        )
        broker = ReplayBroker(
            cfg=PaperBrokerConfig(),
            klines=replay_klines,
            funding=replay_funding,
        )
    elif cfg.broker_kind == "live":
        from execution.live_broker import LiveBroker
        broker = LiveBroker()
    else:
        raise ValueError(f"unknown broker_kind: {cfg.broker_kind!r}")
```

Add the import at the top of `src/wiring.py` if not present:
```python
import pandas as pd
```

(Check existing imports first — likely already there for `RollingKlineCache`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_wiring_broker_kind.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run full suite**

Run: `pytest -q`
Expected: 317 passed (314 + 3 wiring tests).

- [ ] **Step 7: Commit**

```bash
git add src/orchestrator.py src/wiring.py tests/unit/test_wiring_broker_kind.py
git commit -m "feat(wiring): broker_kind selects paper/replay/live broker at boot"
```

- [ ] **Step 8: Write Plan 5B-2 STATUS**

Create `docs/superpowers/plans/2026-04-26-pivot-plan5b2-STATUS.md`:

```markdown
# Plan 5B-2 STATUS — ReplayBroker + Broker Contract

**Date**: 2026-04-26
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: (Plan 5B-1 STATUS commit, e.g., `4d66f0e`)
**Head commit**: (this commit)

## Summary

`PaperBroker`'s slippage + fee math extracted to `src/execution/cost_model.py` (3 pure functions). `ReplayBroker` (deterministic, external clock) and `LiveBroker` (refuses orders) added; both share the cost model. `tests/contracts/test_broker_contract.py` now runs against PaperBroker AND ReplayBroker. `OrchestratorConfig.broker_kind` selects which broker the wiring constructs at boot.

Test count: **317 passed** (Plan 5B-1 baseline 296 + 21 new across 5 tasks).

## Task table

| # | Title | Commit | Files |
|---|-------|--------|-------|
| 1 | Extract `cost_model.py`; PaperBroker refactor | (commit SHA) | `cost_model.py`, `paper_broker.py`, test |
| 2 | `ReplayBroker` | (commit SHA) | `replay_broker.py`, test |
| 3 | `LiveBroker` stub | (commit SHA) | `live_broker.py`, test |
| 4 | Broker contract: ReplayBroker added | (commit SHA) | `tests/contracts/test_broker_contract.py` |
| 5 | Wiring `broker_kind` switch + STATUS | (commit SHA) | `orchestrator.py`, `wiring.py`, test |

## Decisions landed

- **Cost model is shared** (`execution.cost_model`). Backtest fills now provably match paper fills bit-for-bit (spec §7.2).
- **ReplayBroker has zero realism noise** — no latency, no partial fill, no rejection. Determinism over realism for backtest.
- **External clock**: `ReplayBroker.set_time(ts)` is the only mutator of internal time. The Plan 5B-3 backtest harness will own the loop.
- **LiveBroker raises `LiveBrokerNotImplemented`** on submit/cancel; positions/balance return empty/zero so wiring boot doesn't crash. Plan 5D replaces.

## What is NOT done (Plan 5B-3+ scope)

- **Plan 5B-3**: Walk-forward backtest harness that drives `ReplayBroker.set_time` over historical klines + records `backtest_runs` rows + computes Deflated Sharpe.
- **Plan 5B-4**: Pre-Live Gate module (§10 8 gates).
- **Plan 5D**: Replace `LiveBroker` stub with real Binance live integration.

## Known follow-ups

- **`ReplayBroker` is single-symbol** by construction (`klines` is one DataFrame). Multi-symbol replay would require `klines: dict[symbol, df]`. Defer.
- **Funding tick semantic**: `set_time` emits funding events for boundaries strictly between previous and new time. If the harness skips `set_time` calls (e.g., advances by 8h directly), funding still fires correctly.
- **`LiveBroker.events()` never yields** — caller must handle the empty-async-iterator case before iterating.
- **No backward-compat warning when `cfg.broker_kind` is unknown** — `wiring.py` raises `ValueError` immediately, which is the right behavior but worth noting.
```

- [ ] **Step 9: Final commit**

```bash
git add docs/superpowers/plans/2026-04-26-pivot-plan5b2-STATUS.md
git commit -m "docs: Plan 5B-2 handoff STATUS"
```

---

## Self-review notes

- **Spec coverage**: Plan 5B-1 STATUS listed "ReplayBroker + Broker contract test suite" as 5B-2 scope. Delivered. Plus LiveBroker stub (which 5B-1 STATUS implied as part of the broker contract work) and the wiring switch.
- **Type consistency**: `ReplayBroker(cfg: PaperBrokerConfig, klines, funding=None, initial_equity_usdt=10_000)` is the only public constructor. `LiveBroker()` takes no args. Both implement the Broker Protocol from `src/execution/base.py`.
- **No placeholders**: Every step has working code; every test has assertions; every commit message is concrete.
- **Test math is verifiable**: Task 2's `fill_price = 3025 * 1.0001 = 3025.3025` and `fee = 3025.3025 * 0.1 * 5/10000` are exact arithmetic; reviewer can compute by hand.
- **Backward compat**: `OrchestratorConfig.broker_kind` defaults to `"paper"`. All existing callers (orchestrator boot, wiring tests, e2e tests) get PaperBroker as before. No silent behavior change.
