"""CLI entry — ``python -m src.cli``.

Not a user UI. Ops-only.
"""
from __future__ import annotations

import argparse
import asyncio

import structlog

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
    _configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default="data/state.db")
    args = ap.parse_args()
    orch = Orchestrator(OrchestratorConfig(sqlite_path=args.sqlite))
    asyncio.run(orch.run())


if __name__ == "__main__":
    main()
