"""ChatLLM unit tests with mock Ollama."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from interface.chat_llm import CHAT_PROMPT_PATH, ChatLLM
from interface.repositories import MessageRepo, ToolCallRepo
from interface.tools import ToolExecutor
from models.llm.ollama_client import Priority


def _make_engine(tmp_path: Path) -> sa.Engine:
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{db}")


def _mock_chat_response(content: str, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(message=msg)


@pytest.fixture
def chat_deps(tmp_path):
    engine = _make_engine(tmp_path)
    msg_repo = MessageRepo(engine)
    tc_repo = ToolCallRepo(engine)
    broker_mock = AsyncMock()
    broker_mock.positions = AsyncMock(return_value=[])
    tool_exec = ToolExecutor(engine, broker_mock)
    client = AsyncMock()
    return engine, client, tool_exec, msg_repo, tc_repo


@pytest.mark.asyncio
async def test_explain_returns_text(chat_deps):
    _, client, tool_exec, msg_repo, tc_repo = chat_deps
    client.chat = AsyncMock(return_value=_mock_chat_response("ETH looks bullish due to SMC structure."))
    llm = ChatLLM(client=client, tool_executor=tool_exec,
                   message_repo=msg_repo, tool_call_repo=tc_repo)
    result = await llm.explain({"symbol": "ETHUSDT", "direction": "long"})
    assert "ETH" in result
    client.chat.assert_called_once()
    # Priority assertion: explain uses SCHEDULED_MACRO
    _, kwargs = client.chat.call_args_list[0]
    assert kwargs["priority"] == Priority.SCHEDULED_MACRO


@pytest.mark.asyncio
async def test_converse_simple_reply(chat_deps):
    engine, client, tool_exec, msg_repo, tc_repo = chat_deps
    client.chat = AsyncMock(return_value=_mock_chat_response("No open positions currently."))

    # Create conversation
    from interface.repositories import ConversationRepo
    conv_repo = ConversationRepo(engine)
    cid = conv_repo.create("test_chat_123")

    llm = ChatLLM(client=client, tool_executor=tool_exec,
                   message_repo=msg_repo, tool_call_repo=tc_repo)
    result = await llm.converse(cid, "What are my positions?")
    assert "positions" in result.lower()

    # Verify message persisted
    history = msg_repo.history(cid)
    assert any(m["role"] == "user" for m in history)
    assert any(m["role"] == "assistant" for m in history)

    # Priority assertion: converse uses CHAT
    _, kwargs = client.chat.call_args_list[0]
    assert kwargs["priority"] == Priority.CHAT


@pytest.mark.asyncio
async def test_converse_with_tool_call(chat_deps):
    engine, client, tool_exec, msg_repo, tc_repo = chat_deps

    tc = SimpleNamespace(
        function=SimpleNamespace(name="get_positions", arguments={})
    )
    call1 = _mock_chat_response("", tool_calls=[tc])
    call2 = _mock_chat_response("You have no open positions.")
    client.chat = AsyncMock(side_effect=[call1, call2])

    from interface.repositories import ConversationRepo
    cid = ConversationRepo(engine).create("test_chat_456")

    llm = ChatLLM(client=client, tool_executor=tool_exec,
                   message_repo=msg_repo, tool_call_repo=tc_repo)
    result = await llm.converse(cid, "Show my positions")
    assert "no open positions" in result.lower()
    assert client.chat.call_count == 2


@pytest.mark.asyncio
async def test_converse_exhausts_tool_rounds(chat_deps):
    engine, client, tool_exec, msg_repo, tc_repo = chat_deps

    # Every round comes back as a tool_call — the model never lands on plain text.
    tc = SimpleNamespace(
        function=SimpleNamespace(name="get_positions", arguments={})
    )
    tool_call_resp = _mock_chat_response("", tool_calls=[tc])
    client.chat = AsyncMock(return_value=tool_call_resp)

    from interface.repositories import ConversationRepo
    cid = ConversationRepo(engine).create("test_chat_exhaust")

    llm = ChatLLM(client=client, tool_executor=tool_exec,
                   message_repo=msg_repo, tool_call_repo=tc_repo)
    result = await llm.converse(cid, "Loop forever please")

    assert result == "（抱歉，無法在 5 步內完成這個請求，請重新描述。）"
    assert client.chat.call_count == llm.max_tool_rounds == 5


def test_chat_llm_prompt_version_matches_file_hash():
    expected_hex = hashlib.sha256(CHAT_PROMPT_PATH.read_bytes()).hexdigest()
    llm = ChatLLM(
        client=None,
        tool_executor=None,
        message_repo=None,
        tool_call_repo=None,
    )
    assert llm.prompt_version == expected_hex
