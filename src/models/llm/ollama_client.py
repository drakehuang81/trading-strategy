"""Minimal Ollama client wrapper — Plan-2 scope.

Plan-3 replaces this with a priority-queue-aware client (spec §4.6.1).
"""
from __future__ import annotations

import asyncio
from typing import Any

import instructor
import ollama
from pydantic import BaseModel


class OllamaClient:
    def __init__(self, model: str = "gemma2:4b", host: str = "http://localhost:11434") -> None:
        self._model = model
        self._client = instructor.from_openai(
            ollama.AsyncClient(host=host),
            mode=instructor.Mode.JSON,
        )
        self._sem = asyncio.Semaphore(1)

    async def complete(self, prompt: str, schema: type[BaseModel], **kw: Any) -> BaseModel:
        async with self._sem:
            return await self._client.chat.completions.create(
                model=self._model,
                response_model=schema,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                **kw,
            )
