"""Reconciliation — spec §7.4.

PaperAutoRepair: trust broker, overwrite local, log, continue.
LiveConfirmViaTelegram: Plan 4 scope.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol

import sqlalchemy as sa
import structlog

from execution.base import Broker

log = structlog.get_logger()

DUST_THRESHOLD = 0.01  # ignore differences smaller than this


class ReconciliationPolicy(Protocol):
    async def reconcile(
        self, broker: Broker, local_positions: dict[str, float]
    ) -> list[dict[str, Any]]: ...


class PaperAutoRepair:
    """Paper mode: trust broker, overwrite local, log diff.

    Returns list of diff dicts for the caller to apply to local state.
    """

    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    async def reconcile(
        self, broker: Broker, local_positions: dict[str, float]
    ) -> list[dict[str, Any]]:
        broker_positions = await broker.positions()
        broker_map = {p.symbol: p.qty for p in broker_positions}

        all_symbols = set(broker_map.keys()) | set(local_positions.keys())
        diffs: list[dict[str, Any]] = []

        for symbol in all_symbols:
            broker_qty = broker_map.get(symbol, 0.0)
            local_qty = local_positions.get(symbol, 0.0)
            if abs(broker_qty - local_qty) > DUST_THRESHOLD:
                diff: dict[str, Any] = {
                    "symbol": symbol,
                    "broker_qty": broker_qty,
                    "local_qty": local_qty,
                    "action": "trust_broker",
                }
                diffs.append(diff)
                self._persist_diff(diff)
                log.warning("position_mismatch_repaired", **diff)

        return diffs

    def _persist_diff(self, diff: dict[str, Any]) -> None:
        with self._engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO reconciliation_diffs (ts, kind, diff_json, resolution) "
                "VALUES (:ts, :kind, :diff, :res)"
            ), {
                "ts": datetime.now(tz=timezone.utc),
                "kind": "position",
                "diff": json.dumps(diff),
                "res": "auto_repaired",
            })
