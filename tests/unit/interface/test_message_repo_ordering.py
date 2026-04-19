"""Verify stable ordering under microsecond timestamp collisions."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from interface.repositories import ConversationRepo, MessageRepo


def _engine(tmp_path: Path) -> sa.Engine:
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{db}")


def test_history_stable_on_identical_timestamps(tmp_path: Path):
    engine = _engine(tmp_path)
    conv_repo = ConversationRepo(engine)
    msg_repo = MessageRepo(engine)

    cid = conv_repo.create("chat-1")
    fixed_ts = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)
    # Three rows sharing the exact same timestamp; insert in order A, B, C.
    for i, role, content in [(1, "user", "A"), (2, "assistant", "B"), (3, "user", "C")]:
        with engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO messages (message_id, conversation_id, ts, role, content, seq) "
                "VALUES (:mid, :cid, :ts, :role, :content, :seq)"
            ), {
                "mid": f"m{i}", "cid": cid, "ts": fixed_ts,
                "role": role, "content": content, "seq": i,
            })

    hist = msg_repo.history(cid, limit=10)
    assert [m["content"] for m in hist] == ["A", "B", "C"]


def test_append_assigns_monotonic_seq(tmp_path: Path):
    engine = _engine(tmp_path)
    conv_repo = ConversationRepo(engine)
    msg_repo = MessageRepo(engine)
    cid = conv_repo.create("chat-2")

    msg_repo.append(cid, "user", "first")
    msg_repo.append(cid, "assistant", "second")
    msg_repo.append(cid, "user", "third")

    with engine.connect() as conn:
        rows = conn.execute(sa.text(
            "SELECT content, seq FROM messages WHERE conversation_id = :cid ORDER BY seq"
        ), {"cid": cid}).all()
    assert [r[0] for r in rows] == ["first", "second", "third"]
    assert [r[1] for r in rows] == [1, 2, 3]
