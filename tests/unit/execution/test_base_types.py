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
