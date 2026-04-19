"""Telegram bot unit tests (no live bot — mock python-telegram-bot)."""
from __future__ import annotations

import pytest

from interface.telegram_bot import parse_analyze_command


def test_parse_analyze_with_symbol():
    assert parse_analyze_command("/analyze ETHUSDT") == "ETHUSDT"


def test_parse_analyze_default():
    assert parse_analyze_command("/analyze") == "ETHUSDT"


def test_parse_analyze_lowercase():
    assert parse_analyze_command("/analyze btcusdt") == "BTCUSDT"
