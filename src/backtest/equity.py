"""Equity curve from broker events — Plan 5B-3 Task 3."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd

from execution.base import BrokerEvent


def equity_curve(*, events: Iterable[BrokerEvent],
                 initial_equity: float,
                 mark_prices: dict[datetime, float]) -> pd.Series:
    """Build a per-bar equity Series from a stream of broker events.

    `mark_prices` provides per-bar marks (DatetimeIndex → close price).
    `mark_prices` keys must be tz-aware datetimes (UTC), matching
    BrokerEvent.ts_epoch_ms semantics.
    Each event lands at the bar matching its `ts_epoch_ms`.

    Equity update rules:
    - filled (qty signed): pay fee; update position; realised PnL = 0
      (entry) or `(price - avg_entry) * qty_closed` (close-out portion).
      Position avg_entry uses weighted avg on adds, kept on reduces.
    - funding_charged: pay fee, no position change.

    Then mark-to-market at each bar in `mark_prices` against current
    open position: `unrealised = (mark - avg_entry) * qty`.

    Returns a Series indexed by mark_prices keys (sorted), values are
    the running equity = cash + unrealised. Initial point is
    `initial_equity` at the earliest mark price (or alone if empty).
    """
    sorted_marks = sorted(mark_prices.keys())
    if sorted_marks and sorted_marks[0].tzinfo is None:
        raise ValueError(
            "mark_prices keys must be tz-aware datetimes (UTC). "
            "Naive datetimes silently misalign with BrokerEvent.ts_epoch_ms (UTC)."
        )
    if not sorted_marks:
        return pd.Series([initial_equity])

    cash = initial_equity
    pos_qty = 0.0
    pos_avg = 0.0

    # Group events by their bar timestamp (ms-aligned to nearest mark).
    # For simplicity we bucket events to the next mark whose ts >= event ts.
    events_sorted = sorted(events, key=lambda e: e.ts_epoch_ms)
    ev_idx = 0

    out_index: list[datetime] = []
    out_values: list[float] = []

    for mark_ts in sorted_marks:
        mark_ms = int(mark_ts.timestamp() * 1000)
        # Apply all events with ts <= this mark.
        while ev_idx < len(events_sorted) and events_sorted[ev_idx].ts_epoch_ms <= mark_ms:
            e = events_sorted[ev_idx]
            if e.kind == "filled" and e.fill_qty is not None and e.fill_price is not None:
                fee = e.fee or 0.0
                cash -= fee
                # Realise PnL on the closing portion if this fill reduces position.
                if pos_qty != 0 and (pos_qty > 0) != (e.fill_qty > 0):
                    closing = min(abs(e.fill_qty), abs(pos_qty))
                    sign_pos = 1 if pos_qty > 0 else -1
                    realised = (e.fill_price - pos_avg) * closing * sign_pos
                    cash += realised
                    new_qty = pos_qty + e.fill_qty
                    if abs(new_qty) < 1e-12:
                        pos_qty, pos_avg = 0.0, 0.0
                    elif (new_qty > 0) != (pos_qty > 0):
                        # Crossed zero: residual opens new opposite position at fill_price.
                        pos_qty = new_qty
                        pos_avg = e.fill_price
                    else:
                        # Partial reduce, same direction: realised PnL on the
                        # closing leg already moved into cash above; remaining
                        # qty is still MtM'd against the original entry basis,
                        # so avg_entry intentionally stays unchanged.
                        pos_qty = new_qty
                        # avg_entry stays
                else:
                    # Same-side add (or new position).
                    new_qty = pos_qty + e.fill_qty
                    if pos_qty == 0:
                        pos_avg = e.fill_price
                    else:
                        pos_avg = (pos_avg * pos_qty + e.fill_price * e.fill_qty) / new_qty
                    pos_qty = new_qty
            elif e.kind == "funding_charged":
                cash -= (e.fee or 0.0)
            ev_idx += 1

        unrealised = (mark_prices[mark_ts] - pos_avg) * pos_qty if pos_qty != 0 else 0.0
        out_index.append(mark_ts)
        out_values.append(cash + unrealised)

    return pd.Series(out_values, index=pd.DatetimeIndex(out_index, name="ts"))
