"""ChatLLM — spec §4.6. Conversational interface backed by Ollama.

Distinct from GemmaContextProvider: different prompt, different schema,
supports tool calls. Uses READ_ONLY_TOOLS only — cannot submit orders.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from interface.repositories import MessageRepo, ToolCallRepo
from interface.tools import TOOL_SCHEMAS, ToolExecutor
from models.llm.ollama_client import OllamaClient, Priority

log = structlog.get_logger()

CHAT_PROMPT_PATH = Path("config/prompts/chat_llm.md")


def _load_system_prompt() -> tuple[str, str]:
    body = CHAT_PROMPT_PATH.read_bytes()
    return body.decode(), hashlib.sha256(body).hexdigest()


@dataclass
class ChatLLM:
    client: OllamaClient
    tool_executor: ToolExecutor
    message_repo: MessageRepo
    tool_call_repo: ToolCallRepo
    prompt_version: str = ""
    _system: str = ""
    max_tool_rounds: int = 5

    def __post_init__(self) -> None:
        self._system, self.prompt_version = _load_system_prompt()

    async def explain(self, proposal_dict: dict[str, Any]) -> str:
        """Generate a rationale for a trade proposal (non-streaming)."""
        prompt = (
            f"{self._system}\n\n"
            f"Explain this trade proposal in under 100 words. "
            f"Focus on key feature signals and risk factors.\n\n"
            f"Proposal:\n{json.dumps(proposal_dict, default=str)[:4000]}"
        )
        response = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            priority=Priority.SCHEDULED_MACRO,
        )
        return response.message.content or ""

    async def converse(
        self,
        conversation_id: str,
        user_message: str,
    ) -> str:
        """Handle a user message with tool call loop. Returns final text."""
        self.message_repo.append(conversation_id, "user", user_message)

        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system}]
        history = self.message_repo.history(conversation_id, limit=20)
        messages.extend(history)

        for _ in range(self.max_tool_rounds):
            response = await self.client.chat(
                messages=messages, priority=Priority.CHAT, tools=TOOL_SCHEMAS,
            )
            msg = response.message

            if not msg.tool_calls:
                content = msg.content or ""
                mid = self.message_repo.append(conversation_id, "assistant", content)
                return content

            # Process tool calls
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
            for tc in msg.tool_calls:
                name = tc.function.name
                args = tc.function.arguments if isinstance(tc.function.arguments, dict) else {}
                result = await self.tool_executor.execute(name, args)
                mid = self.message_repo.append(conversation_id, "assistant", f"[tool:{name}]")
                self.tool_call_repo.insert(mid, name, args, result)
                messages.append({"role": "tool", "content": result})

        # Exhausted tool rounds — return whatever we have
        final = messages[-1].get("content", "") if messages else ""
        self.message_repo.append(conversation_id, "assistant", str(final))
        return str(final)
