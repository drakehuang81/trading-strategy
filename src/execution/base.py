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
