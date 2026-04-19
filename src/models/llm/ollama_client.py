"""OllamaClient with priority queue — spec §4.6.1.

Single client owning a PriorityQueue. All LLM calls route here.
Priority: scheduled_macro (high) > on_demand_deep (med) > chat (low).
A background scheduler task pulls tickets in priority order, ensuring
only one LLM call is active at any time.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import instructor
import ollama
from pydantic import BaseModel


class Priority(IntEnum):
    SCHEDULED_MACRO = 0   # highest
    ON_DEMAND_DEEP = 1
    CHAT = 2              # lowest


@dataclass(order=True)
class _Ticket:
    priority: int
    ts: float
    ready: asyncio.Event = field(compare=False, default_factory=asyncio.Event)
    done: asyncio.Event = field(compare=False, default_factory=asyncio.Event)


class OllamaClient:
    def __init__(self, model: str = "gemma2:4b", host: str = "http://localhost:11434") -> None:
        self._model = model
        self._raw = ollama.AsyncClient(host=host)
        self._instructor = instructor.from_openai(
            self._raw,
            mode=instructor.Mode.JSON,
        )
        self._queue: asyncio.PriorityQueue[_Ticket] = asyncio.PriorityQueue()
        self._scheduler_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the priority scheduler. Must be called before complete()/chat()."""
        if self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(self._scheduler())

    async def stop(self) -> None:
        """Cancel the scheduler task."""
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

    async def _scheduler(self) -> None:
        """Pull tickets in priority order; signal ready, wait for done."""
        while True:
            ticket = await self._queue.get()
            ticket.ready.set()
            await ticket.done.wait()

    async def _acquire(self, priority: Priority) -> _Ticket:
        """Enqueue a ticket and wait until the scheduler grants access."""
        ticket = _Ticket(priority=priority.value, ts=time.monotonic())
        await self._queue.put(ticket)
        await ticket.ready.wait()
        return ticket

    async def complete(
        self,
        prompt: str,
        schema: type[BaseModel],
        priority: Priority = Priority.CHAT,
        **kw: Any,
    ) -> BaseModel:
        """Structured JSON output via instructor (used by GemmaContextProvider)."""
        ticket = await self._acquire(priority)
        try:
            result: BaseModel = await self._instructor.chat.completions.create(
                model=self._model,
                response_model=schema,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                **kw,
            )
            return result
        finally:
            ticket.done.set()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        priority: Priority = Priority.CHAT,
        tools: list[dict[str, Any]] | None = None,
    ) -> ollama.ChatResponse:
        """Free-form chat (used by ChatLLM). Returns raw ollama response."""
        ticket = await self._acquire(priority)
        try:
            kw: dict[str, Any] = {"model": self._model, "messages": messages}
            if tools:
                kw["tools"] = tools
            response: ollama.ChatResponse = await self._raw.chat(**kw)
            return response
        finally:
            ticket.done.set()
