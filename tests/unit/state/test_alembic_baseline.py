"""Alembic baseline smoke test — spec §8.1 + §8.3.

Verifies: (a) fresh upgrade from base creates all expected tables;
(b) broker_events.event_id has a UNIQUE constraint (idempotency, §8.3);
(c) round-trip downgrade → upgrade is clean."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

EXPECTED_TABLES = {
    "backtest_runs",
    "broker_events",
    "conversations",
    "dead_letter",
    "fills",
    "feature_cache_manifest",
    "halt_events",
    "heartbeat",
    "log",
    "messages",
    "model_versions",
    "positions",
    "prediction_disagreements",
    "proposals",
    "reconciliation_diffs",
    "session_state",
    "tool_calls",
}


@pytest.fixture
def alembic_config(tmp_path: Path) -> Config:
    db_path = tmp_path / "smoke.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _tables(db_url: str) -> set[str]:
    path = db_url.removeprefix("sqlite:///")
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'alembic_%'"
        ).fetchall()
    return {r[0] for r in rows}


def test_baseline_creates_all_expected_tables(alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")
    db_url = alembic_config.get_main_option("sqlalchemy.url")
    assert _tables(db_url) == EXPECTED_TABLES


def test_broker_events_event_id_is_primary_key(alembic_config: Config) -> None:
    """Spec §8.3 idempotency contract."""
    command.upgrade(alembic_config, "head")
    db_url = alembic_config.get_main_option("sqlalchemy.url")
    path = db_url.removeprefix("sqlite:///")
    with sqlite3.connect(path) as conn:
        info = conn.execute("PRAGMA table_info(broker_events)").fetchall()
    pk_cols = [row[1] for row in info if row[5] > 0]
    assert pk_cols == ["event_id"], f"event_id must be sole PK; got {pk_cols}"


def test_round_trip_downgrade_upgrade(alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")
    db_url = alembic_config.get_main_option("sqlalchemy.url")
    assert _tables(db_url) == set()
    command.upgrade(alembic_config, "head")
    assert _tables(db_url) == EXPECTED_TABLES
