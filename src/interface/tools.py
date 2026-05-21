"""ChatLLM tool registry — spec §4.6 READ_ONLY_TOOLS only.

Cannot submit orders. Enforced by boundary contract test.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import sqlalchemy as sa

from execution.base import Broker


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_positions",
            "description": "Get all currently open positions with symbol, qty, and entry price",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_proposals",
            "description": "Get recent trade proposals with direction, entry, stop loss, acceptance status",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Number of proposals (default 5)"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pnl_summary",
            "description": "Get today's P&L summary: day_pnl_r, consecutive_wins",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature_snapshot",
            "description": "Get the latest computed feature values for a symbol",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "Trading pair, e.g. ETHUSDT"}},
                "required": ["symbol"],
            },
        },
    },
]

TOOL_NAMES: set[str] = {t["function"]["name"] for t in TOOL_SCHEMAS}


class ToolExecutor:
    """Executes READ_ONLY_TOOLS against live state."""

    def __init__(self, engine: sa.Engine, broker: Broker) -> None:
        self._engine = engine
        self._broker = broker

    async def execute(self, name: str, args: dict[str, Any]) -> str:
        if name == "get_positions":
            positions = await self._broker.positions()
            return json.dumps([p.model_dump(mode="json") for p in positions])
        elif name == "get_recent_proposals":
            limit = args.get("limit", 5)
            return self._query_proposals(limit)
        elif name == "get_pnl_summary":
            return self._query_pnl()
        elif name == "get_feature_snapshot":
            symbol = args.get("symbol", "")
            return self._query_feature_snapshot(symbol)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    def _query_proposals(self, limit: int) -> str:
        with self._engine.connect() as conn:
            rows = conn.execute(sa.text(
                "SELECT proposal_id, symbol, direction, entry, stop_loss, accepted, rationale "
                "FROM proposals ORDER BY ts DESC LIMIT :lim"
            ), {"lim": limit}).all()
        proposals = [
            {"proposal_id": r[0], "symbol": r[1], "direction": r[2],
             "entry": r[3], "stop_loss": r[4], "accepted": bool(r[5]),
             "rationale": r[6]}
            for r in rows
        ]
        return json.dumps(proposals)

    def _query_pnl(self) -> str:
        with self._engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT consecutive_wins, day_pnl_r FROM session_state WHERE date = :d"
            ), {"d": date.today().isoformat()}).first()
        if row:
            return json.dumps({"consecutive_wins": row[0], "day_pnl_r": row[1]})
        return json.dumps({"consecutive_wins": 0, "day_pnl_r": 0.0})

    def _query_feature_snapshot(self, symbol: str) -> str:
        with self._engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT symbol, feature_snapshot_json, ts "
                "FROM proposals WHERE symbol = :s "
                "ORDER BY ts DESC LIMIT 1"
            ), {"s": symbol}).first()
        if row is None:
            return json.dumps({"error": "no_proposal_found", "symbol": symbol})
        return json.dumps({
            "symbol": row[0],
            "features": json.loads(row[1]),
            "ts": str(row[2]),
        })
