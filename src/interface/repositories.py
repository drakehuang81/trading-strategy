"""Conversation persistence — spec §8.1 tables: conversations, messages, tool_calls."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa


class ConversationRepo:
    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    def create(self, telegram_chat_id: str) -> str:
        cid = str(uuid.uuid4())
        with self._engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO conversations (conversation_id, started_at, telegram_chat_id) "
                "VALUES (:cid, :ts, :chat_id)"
            ), {"cid": cid, "ts": datetime.now(tz=timezone.utc), "chat_id": telegram_chat_id})
        return cid

    def get_by_chat_id(self, telegram_chat_id: str) -> str | None:
        with self._engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT conversation_id FROM conversations "
                "WHERE telegram_chat_id = :chat_id ORDER BY started_at DESC LIMIT 1"
            ), {"chat_id": telegram_chat_id}).first()
        return row[0] if row else None


class MessageRepo:
    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    def append(self, conversation_id: str, role: str, content: str) -> str:
        mid = str(uuid.uuid4())
        with self._engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO messages (message_id, conversation_id, ts, role, content) "
                "VALUES (:mid, :cid, :ts, :role, :content)"
            ), {
                "mid": mid, "cid": conversation_id,
                "ts": datetime.now(tz=timezone.utc),
                "role": role, "content": content,
            })
        return mid

    def history(self, conversation_id: str, limit: int = 20) -> list[dict[str, str]]:
        with self._engine.connect() as conn:
            rows = conn.execute(sa.text(
                "SELECT role, content FROM messages "
                "WHERE conversation_id = :cid ORDER BY ts DESC LIMIT :lim"
            ), {"cid": conversation_id, "lim": limit}).all()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


class ToolCallRepo:
    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    def insert(self, message_id: str, name: str, args: dict[str, Any], result: str) -> str:
        tcid = str(uuid.uuid4())
        with self._engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO tool_calls (tool_call_id, message_id, name, args_json, result_json) "
                "VALUES (:tcid, :mid, :name, :args, :result)"
            ), {
                "tcid": tcid, "mid": message_id,
                "name": name, "args": json.dumps(args),
                "result": result,
            })
        return tcid
