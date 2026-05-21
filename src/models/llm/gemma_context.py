"""GemmaContextProvider — spec §4.3. Emits flags; never probabilities."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.base import LLMContextFlags
from models.llm.ollama_client import OllamaClient, Priority


PROMPT_PATH = Path("config/prompts/context_provider.md")


def _load_prompt() -> tuple[str, str]:
    body = PROMPT_PATH.read_bytes()
    return body.decode(), hashlib.sha256(body).hexdigest()


@dataclass
class GemmaContextProvider:
    client: OllamaClient
    prompt_version: str = ""
    _system: str = ""

    def __post_init__(self) -> None:
        self._system, self.prompt_version = _load_prompt()

    async def flags(self, features: dict[str, Any]) -> LLMContextFlags:
        prompt = f"{self._system}\n\nFeatures:\n{json.dumps(features, default=str)[:8000]}"
        result = await self.client.complete(prompt, schema=LLMContextFlags, priority=Priority.SCHEDULED_MACRO)
        assert isinstance(result, LLMContextFlags)
        return result
