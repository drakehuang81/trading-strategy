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
from execution.cost_model import slippage_fill_price, taker_or_maker_fee
from execution.paper_broker import PaperBrokerConfig


@dataclass
class ReplayBroker:
    cfg: PaperBrokerConfig
    klines: pd.DataFrame
    symbol: str = "ETHUSDT"
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
        """Advance internal clock. Emits funding events for any boundaries
        strictly between previous time and new time."""
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
        if order.symbol != self.symbol:
            raise ValueError(
                f"ReplayBroker is configured for {self.symbol!r} but received order for {order.symbol!r}"
            )
        order_id = str(uuid.uuid4())
        self._orders[order_id] = order
        await self._emit("submitted", order_id, order)

        mid = self._current_close()
        fill_price = slippage_fill_price(
            mid=mid, side=order.side, qty=order.qty, cfg=self.cfg.slippage,
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

    def drain_events(self) -> list[BrokerEvent]:
        """Synchronously drain all currently queued events.

        Returns events in FIFO order, then leaves the queue empty.
        Used by the backtest harness to process per-bar event batches
        without async iteration. Production paper mode should still use
        `events()` for streaming.
        """
        out: list[BrokerEvent] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return out

    @property
    def current_ts(self) -> "pd.Timestamp | None":
        """Public accessor for the broker's clock — used by backtest harness."""
        return self._current_ts

    def current_mid(self) -> float:
        """Returns the current bar's close (mid). Raises if clock not set."""
        return self._current_close()

    def _close_at(self, ts: pd.Timestamp) -> float:
        """Returns close of the kline whose index is <= ts (last bar at or before ts)."""
        sub = self.klines[self.klines.index <= ts]
        if sub.empty:
            raise RuntimeError(f"no kline at or before {ts}")
        return float(sub["close"].iloc[-1])

    def _current_close(self) -> float:
        if self._current_ts is None:
            raise RuntimeError("clock not set")
        return self._close_at(self._current_ts)

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
            # Same side: VWAP across fills.
            new_avg = (
                existing.avg_entry * existing.qty + fill_price * signed_qty
            ) / new_qty
        elif (existing.qty > 0) != (new_qty > 0):
            # Crossed zero: closed old position, opened new opposite at fill_price.
            new_avg = fill_price
        else:
            # Partial reduction (still same side as existing): keep entry basis.
            new_avg = existing.avg_entry
        self._positions[symbol] = existing.model_copy(update={
            "qty": new_qty,
            "avg_entry": new_avg,
            "last_update_ts": self._current_ts.to_pydatetime(),
        })

    def _charge_funding(self, ts: pd.Timestamp, rate: float) -> None:
        for sym, pos in self._positions.items():
            mid = self._close_at(ts)
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
