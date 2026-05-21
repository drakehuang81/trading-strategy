"""Quick verification that TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID work.

Sends one "hello" message to the configured chat. If you see the message
land in your Telegram, your `.env` is set up correctly and the full bot
will work when you run `python -m src.cli`.

Usage:
    PYTHONPATH=src python scripts/telegram_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from interface.telegram_bot import TelegramBot


async def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID not set in .env")
        print("Copy .env.example → .env and fill in both values, then re-run.")
        sys.exit(1)

    bot = TelegramBot(token=token)
    bot.build()
    await bot._app.initialize()
    await bot._app.start()
    try:
        await bot.send_message(chat_id, "✅ trading-bot smoke test — wiring is correct")
        print(f"sent hello to chat_id={chat_id}")
    finally:
        await bot._app.stop()
        await bot._app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
