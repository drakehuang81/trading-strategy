"""OllamaClient priority queue tests."""
from __future__ import annotations

import asyncio

import pytest

from models.llm.ollama_client import Priority, _Ticket


def test_tickets_ordered_by_priority():
    """Lower priority number = higher priority = served first."""
    t_chat = _Ticket(priority=Priority.CHAT.value, ts=1.0)
    t_macro = _Ticket(priority=Priority.SCHEDULED_MACRO.value, ts=2.0)
    assert t_macro < t_chat


def test_fifo_within_same_priority():
    t1 = _Ticket(priority=Priority.CHAT.value, ts=1.0)
    t2 = _Ticket(priority=Priority.CHAT.value, ts=2.0)
    assert t1 < t2


@pytest.mark.asyncio
async def test_priority_queue_ordering():
    """PriorityQueue serves highest-priority ticket first."""
    q: asyncio.PriorityQueue[_Ticket] = asyncio.PriorityQueue()
    t_chat = _Ticket(priority=Priority.CHAT.value, ts=1.0)
    t_deep = _Ticket(priority=Priority.ON_DEMAND_DEEP.value, ts=2.0)
    t_macro = _Ticket(priority=Priority.SCHEDULED_MACRO.value, ts=3.0)
    await q.put(t_chat)
    await q.put(t_deep)
    await q.put(t_macro)
    assert (await q.get()) is t_macro
    assert (await q.get()) is t_deep
    assert (await q.get()) is t_chat


@pytest.mark.asyncio
async def test_scheduler_serves_high_priority_first():
    """When multiple requests queue while one is active, highest priority wins."""
    from models.llm.ollama_client import OllamaClient

    client = OllamaClient.__new__(OllamaClient)
    client._queue = asyncio.PriorityQueue()
    client._scheduler_task = None
    await client.start()

    served: list[str] = []

    # Block the scheduler with a low-priority ticket
    blocker = await client._acquire(Priority.CHAT)

    async def worker(pri: Priority, name: str) -> None:
        ticket = await client._acquire(pri)
        try:
            served.append(name)
        finally:
            ticket.done.set()

    t_chat = asyncio.create_task(worker(Priority.CHAT, "chat"))
    t_macro = asyncio.create_task(worker(Priority.SCHEDULED_MACRO, "macro"))
    t_deep = asyncio.create_task(worker(Priority.ON_DEMAND_DEEP, "deep"))
    await asyncio.sleep(0.01)  # let all tasks enqueue

    blocker.done.set()  # release → scheduler serves queued in priority order
    await asyncio.gather(t_chat, t_deep, t_macro)

    assert served == ["macro", "deep", "chat"]
    await client.stop()
