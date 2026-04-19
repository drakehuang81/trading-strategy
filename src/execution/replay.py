"""Pure positions-rebuild function — spec §8.3 idempotency contract."""
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
    agg: dict[str, list[float]] = {}

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
        if qty == 0 or (qty > 0) == (e.fill_qty > 0):
            new_cost = cost + e.fill_price * e.fill_qty
        else:
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
