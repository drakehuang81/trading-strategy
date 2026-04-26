"""LiveBroker — Plan 5B-2 stub; Plan 5D will replace with real Binance.

Refuses submit/cancel; positions/balance return empty/zero so wiring code
that probes them at boot doesn't crash. The wiring path
`cfg.broker_kind="live"` MUST be gated behind Plan 5B-4's Pre-Live Gate
before this stub is replaced — running with this stub in production
would silently no-op every order.
"""
from __future__ import annotations

from dataclasses import dataclass
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
