"""Telegram bot unit tests (no live bot — mock python-telegram-bot)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from interface.telegram_bot import TelegramBot, parse_analyze_command


def test_parse_analyze_with_symbol():
    assert parse_analyze_command("/analyze ETHUSDT") == "ETHUSDT"


def test_parse_analyze_default():
    assert parse_analyze_command("/analyze") == "ETHUSDT"


def test_parse_analyze_lowercase():
    assert parse_analyze_command("/analyze btcusdt") == "BTCUSDT"


def _engine(tmp_path: Path) -> sa.Engine:
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{db}")


@pytest.mark.asyncio
async def test_cmd_status_reports_heartbeat_halt_and_positions(tmp_path: Path):
    engine = _engine(tmp_path)
    now = datetime.now(tz=timezone.utc)
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO heartbeat (ts, trace_id) VALUES (:ts, :tid)"
        ), {"ts": now.isoformat(), "tid": "t"})

    halt = MagicMock()
    halt.is_halted.return_value = False
    broker = MagicMock()
    broker.positions = AsyncMock(return_value=[])

    bot = TelegramBot(token="", halt_manager=halt, broker=broker, engine=engine)
    update = MagicMock()
    update.effective_chat.id = 123
    update.effective_message.reply_text = AsyncMock()
    ctx = MagicMock()

    await bot._cmd_status(update, ctx)

    update.effective_message.reply_text.assert_called_once()
    text = update.effective_message.reply_text.call_args[0][0]
    assert "heartbeat" in text.lower()
    assert "clear" in text.lower()  # halt clear
    assert "open positions: 0" in text.lower()


@pytest.mark.asyncio
async def test_cmd_status_no_heartbeat_row(tmp_path: Path):
    """Fresh DB with no heartbeat should report 'none'."""
    engine = _engine(tmp_path)
    halt = MagicMock()
    halt.is_halted.return_value = True

    bot = TelegramBot(token="", halt_manager=halt, engine=engine)
    update = MagicMock()
    update.effective_message.reply_text = AsyncMock()

    await bot._cmd_status(update, MagicMock())

    text = update.effective_message.reply_text.call_args[0][0]
    assert "halt: active" in text.lower()
    assert "last heartbeat: none" in text.lower()
