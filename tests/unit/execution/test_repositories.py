from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from execution.base import BrokerEvent
from execution.repositories import BrokerEventRepo


@pytest.fixture
def migrated_db(tmp_path: Path) -> sa.Engine:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{tmp_path/'state.db'}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{tmp_path/'state.db'}")


def test_insert_or_ignore_is_idempotent(migrated_db):
    repo = BrokerEventRepo(migrated_db)
    e = BrokerEvent(event_id="e1", kind="filled", order_id="o1", symbol="ETHUSDT",
                    ts_epoch_ms=1, fill_price=2000.0, fill_qty=0.1, fee=0.05)
    assert repo.insert(e) is True
    assert repo.insert(e) is False
    with migrated_db.connect() as conn:
        [count] = conn.execute(sa.text("SELECT COUNT(*) FROM broker_events")).one()
    assert count == 1
