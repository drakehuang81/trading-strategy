"""CLI entry — ``python -m src.cli``.

Ops-only entry point. Loads .env for secrets.
"""
from __future__ import annotations

import argparse
import asyncio
import os

import structlog
from dotenv import load_dotenv

from orchestrator import Orchestrator, OrchestratorConfig


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def main() -> None:
    load_dotenv()
    _configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default=os.getenv("SQLITE_PATH", "data/state.db"))
    ap.add_argument("--telegram-token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    ap.add_argument("--notify-chat-id", default=os.getenv("TELEGRAM_CHAT_ID", ""))
    args = ap.parse_args()
    orch = Orchestrator(OrchestratorConfig(
        sqlite_path=args.sqlite,
        telegram_token=args.telegram_token,
        notify_chat_id=args.notify_chat_id,
    ))
    asyncio.run(orch.run())


if __name__ == "__main__":
    main()
