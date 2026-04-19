from execution.base import BrokerEvent
from execution.replay import rebuild_positions


def ev(event_id: str, order_id: str, kind: str, symbol: str = "ETHUSDT",
       price: float = 2000, qty: float = 0.1, side: str = "buy", fee: float = 0.0) -> BrokerEvent:
    signed_qty = qty if side == "buy" else -qty
    return BrokerEvent(
        event_id=event_id, kind=kind, order_id=order_id, symbol=symbol,
        ts_epoch_ms=0, fill_price=price, fill_qty=signed_qty, fee=fee,
    )


def test_rebuild_positions_basic():
    events = [
        ev("e1", "o1", "filled", qty=0.1, side="buy", price=2000),
        ev("e2", "o1", "filled", qty=0.1, side="buy", price=2010),
        ev("e3", "o2", "filled", qty=0.05, side="sell", price=2050),
    ]
    snap = rebuild_positions(events)
    eth = snap["ETHUSDT"]
    assert abs(eth.qty - 0.15) < 1e-9
    assert abs(eth.avg_entry - (2000 * 0.1 + 2010 * 0.1) / 0.2) < 1e-6


def test_rebuild_positions_idempotent_on_duplicates():
    events = [
        ev("e1", "o1", "filled", qty=0.1, side="buy", price=2000),
        ev("e2", "o1", "filled", qty=0.1, side="buy", price=2010),
    ]
    snap_once = rebuild_positions(events)
    snap_twice = rebuild_positions(events + events)
    assert snap_once == snap_twice
