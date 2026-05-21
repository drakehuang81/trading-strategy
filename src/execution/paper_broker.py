"""PaperBroker — realistic friction from Day 1 (spec §4.5, §7.2).

- Latency: normal(mean, stdev) in ms.
- Fees: taker/maker bps → absolute via price * qty.
- Slippage: linear in order-size-vs-ADV proxy.
- Partial fill: single-split probabilistic.
- Rejection: probabilistic.
- Accepts injected rng for determinism.
"""
from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable

import pandas as pd

from typing import Literal
from execution.base import Balance, BrokerEvent, Order, OrderId, Position
from execution.cost_model import (
    SlippageConfig,
    slippage_fill_price,
    taker_or_maker_fee,
)


@dataclass
class PaperBrokerConfig:
    taker_bps: float = 5.0
    maker_bps: float = 2.0
    latency_ms_mean: float = 200.0
    latency_ms_stdev: float = 50.0
    slippage: SlippageConfig = field(default_factory=lambda: SlippageConfig(
        slippage_bps_base=1.0,
        slippage_bps_per_adv_unit=20.0,
        adv_stub=1000.0,
    ))
    partial_fill_prob: float = 0.15
    rejection_prob: float = 0.01


@dataclass
class PaperBroker:
    cfg: PaperBrokerConfig
    rng: random.Random
    mid_provider: Callable[[str], float]
    funding_dir: Path | None = None
    _queue: asyncio.Queue[BrokerEvent] = field(default_factory=asyncio.Queue)
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
        fill_price = slippage_fill_price(
            mid=mid, side=order.side, qty=order.qty, cfg=self.cfg.slippage,
        )
        sign = 1 if order.side == "buy" else -1

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
            fee = pos.qty * mid * rate
            event = BrokerEvent(
                event_id=str(uuid.uuid4()),
                kind="funding_charged", order_id=f"funding_{sym}_{int(ts.timestamp())}",
                symbol=sym, ts_epoch_ms=int(ts.timestamp() * 1000),
                fee=fee, reason=f"rate={rate}",
            )
            await self._queue.put(event)

    async def _emit(
        self,
        kind: Literal["submitted", "partially_filled", "filled", "rejected", "cancelled", "funding_charged"],
        order_id: OrderId,
        order: Order | None,
        *,
        price: float | None = None,
        qty: float | None = None,
        reason: str | None = None,
    ) -> None:
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
            ts_epoch_ms=int(datetime.now(tz=timezone.utc).timestamp() * 1000),
            fill_price=price, fill_qty=qty, fee=fee, reason=reason,
        )
        await self._queue.put(event)
