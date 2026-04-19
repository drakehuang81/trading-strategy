"""SQLite repositories — thin SQLAlchemy Core wrappers around §8.1 tables.

All writes are append-only (spec §7.3); inserts on unique columns use
INSERT OR IGNORE to satisfy the §8.3 idempotency contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import sqlalchemy as sa

from decision.proposal import TradeProposal
from execution.base import BrokerEvent


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class BrokerEventRepo:
    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    def insert(self, event: BrokerEvent) -> bool:
        """INSERT OR IGNORE on event_id. Returns True if inserted, False if duplicate."""
        stmt = sa.text(
            "INSERT OR IGNORE INTO broker_events "
            "(event_id, kind, order_id, ts, fill_price, fill_qty, fee, reason, ml_model_version, llm_prompt_version) "
            "VALUES (:event_id, :kind, :order_id, :ts, :fill_price, :fill_qty, :fee, :reason, :mv, :pv)"
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt, {
                "event_id": event.event_id,
                "kind": event.kind,
                "order_id": event.order_id,
                "ts": datetime.fromtimestamp(event.ts_epoch_ms / 1000, tz=timezone.utc),
                "fill_price": event.fill_price,
                "fill_qty": event.fill_qty,
                "fee": event.fee,
                "reason": event.reason,
                "mv": event.ml_model_version,
                "pv": event.llm_prompt_version,
            })
            return result.rowcount == 1

    def all(self) -> list[BrokerEvent]:
        with self._engine.connect() as conn:
            rows = conn.execute(sa.text(
                "SELECT event_id, kind, order_id, ts, fill_price, fill_qty, fee, reason, "
                "ml_model_version, llm_prompt_version FROM broker_events ORDER BY ts"
            )).all()
        return [
            BrokerEvent(
                event_id=r[0], kind=r[1], order_id=r[2],
                ts_epoch_ms=int(r[3].timestamp() * 1000) if isinstance(r[3], datetime) else int(r[3]),
                fill_price=r[4], fill_qty=r[5], fee=r[6], reason=r[7],
                ml_model_version=r[8], llm_prompt_version=r[9],
            )
            for r in rows
        ]


class ProposalRepo:
    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    def insert(self, proposal: TradeProposal, accepted: bool) -> None:
        import json
        stmt = sa.text(
            "INSERT INTO proposals "
            "(proposal_id, trace_id, ts, symbol, direction, entry, stop_loss, take_profit_json, "
            " size, confidence, feature_snapshot_json, bundle_json, risk_checks_json, "
            " accepted, rationale, feature_registry_version, ml_model_version, llm_prompt_version) "
            "VALUES (:pid, :tid, :ts, :sym, :dir, :entry, :sl, :tp, :size, :conf, "
            " :feat, :bundle, :rc, :acc, :rat, :fv, :mv, :pv)"
        )
        with self._engine.begin() as conn:
            conn.execute(stmt, {
                "pid": proposal.proposal_id, "tid": proposal.trace_id, "ts": proposal.ts,
                "sym": proposal.symbol, "dir": proposal.direction,
                "entry": proposal.entry, "sl": proposal.stop_loss,
                "tp": json.dumps(proposal.take_profit),
                "size": proposal.size, "conf": proposal.confidence,
                "feat": json.dumps(proposal.feature_snapshot, default=str),
                "bundle": proposal.bundle_json,
                "rc": json.dumps([r.model_dump() for r in proposal.risk_checks]),
                "acc": accepted, "rat": proposal.rationale,
                "fv": proposal.feature_registry_version,
                "mv": proposal.ml_model_version,
                "pv": proposal.llm_prompt_version,
            })


class SessionStateRepo:
    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    def get(self, d) -> tuple[int, float]:
        with self._engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT consecutive_wins, day_pnl_r FROM session_state WHERE date=:d"
            ), {"d": d}).first()
        return (row[0], row[1]) if row else (0, 0.0)

    def upsert(self, d, consecutive_wins: int, day_pnl_r: float) -> None:
        with self._engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO session_state (date, consecutive_wins, day_pnl_r, last_update_ts) "
                "VALUES (:d, :cw, :p, :ts) "
                "ON CONFLICT(date) DO UPDATE SET "
                "consecutive_wins=:cw, day_pnl_r=:p, last_update_ts=:ts"
            ), {"d": d, "cw": consecutive_wins, "p": day_pnl_r, "ts": _now()})
