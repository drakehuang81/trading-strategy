"""Equity curve from broker events — Plan 5B-3 Task 3."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from execution.base import BrokerEvent
from backtest.equity import equity_curve


def _ev(kind: str, ts: datetime, *, price: float | None = None,
        qty: float | None = None, fee: float | None = None) -> BrokerEvent:
    return BrokerEvent(
        event_id=f"ev_{ts.isoformat()}_{kind}",
        kind=kind, order_id="oid",
        symbol="ETHUSDT",
        ts_epoch_ms=int(ts.timestamp() * 1000),
        fill_price=price, fill_qty=qty, fee=fee,
    )


def test_empty_events_returns_initial_equity_only():
    curve = equity_curve(events=[], initial_equity=10_000.0,
                          mark_prices={})
    assert len(curve) == 1
    assert curve.iloc[0] == 10_000.0


def test_buy_then_mark_to_market():
    """Buy 1.0 ETH @3000 (fee 1.5), mark at 3050 → equity = 10_000 - 1.5 + (3050-3000)*1.0 = 10_048.5."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    events = [
        _ev("filled", base, price=3000.0, qty=1.0, fee=1.5),
    ]
    mark_prices = {base: 3000.0, base + timedelta(hours=1): 3050.0}
    curve = equity_curve(events=events, initial_equity=10_000.0,
                          mark_prices=mark_prices)
    assert curve[base + timedelta(hours=1)] == pytest.approx(10_048.5)


def test_buy_then_sell_realised_pnl():
    """Buy 1.0 @3000 (fee 1.5), sell 1.0 @3100 (fee 1.55) → realised 100 - 3.05 = 96.95."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    events = [
        _ev("filled", base, price=3000.0, qty=1.0, fee=1.5),
        _ev("filled", base + timedelta(hours=1), price=3100.0, qty=-1.0, fee=1.55),
    ]
    mark_prices = {
        base: 3000.0,
        base + timedelta(hours=1): 3100.0,
        base + timedelta(hours=2): 3100.0,
    }
    curve = equity_curve(events=events, initial_equity=10_000.0,
                          mark_prices=mark_prices)
    # After close, equity = 10_000 - 1.5 + 100 - 1.55 = 10_096.95
    assert curve[base + timedelta(hours=2)] == pytest.approx(10_096.95)


def test_funding_charge_decreases_equity():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    events = [
        _ev("filled", base, price=3000.0, qty=1.0, fee=1.5),
        _ev("funding_charged", base + timedelta(hours=8), fee=0.3),
    ]
    mark_prices = {
        base: 3000.0,
        base + timedelta(hours=8): 3000.0,
        base + timedelta(hours=9): 3000.0,
    }
    curve = equity_curve(events=events, initial_equity=10_000.0,
                          mark_prices=mark_prices)
    # 10_000 - 1.5 (entry fee) - 0.3 (funding) - 0 (no MtM change) = 9_998.2
    assert curve[base + timedelta(hours=9)] == pytest.approx(9_998.2)
