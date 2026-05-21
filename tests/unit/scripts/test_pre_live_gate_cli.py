"""Pre-Live Gate CLI — Plan 5B-4 Task 5."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest


def _bootstrap_db(sqlite_path: str):
    import alembic.command, alembic.config, sqlalchemy as sa
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{sqlite_path}")
    alembic.command.upgrade(ac, "head")
    return sa.create_engine(f"sqlite:///{sqlite_path}")


@pytest.mark.asyncio
async def test_cli_returns_nonzero_when_any_gate_fails(tmp_path, capsys):
    from scripts.pre_live_gate import main_async
    sqlite = tmp_path / "state.db"
    _bootstrap_db(str(sqlite))   # empty DB → most gates will fail

    args = argparse.Namespace(
        sqlite_path=str(sqlite),
        model_dir=str(tmp_path / "models"),
        watchdog_log=str(tmp_path / "watchdog.log"),
        brier_threshold=0.24,
    )
    with patch("execution.pre_live_gate._run_pytest_repainting", return_value=0):
        rc = await main_async(args)
    assert rc == 1
    captured = capsys.readouterr()
    # Should print every gate name in stdout.
    for name in ["no_repainting", "backtest_dsr", "calibration_brier",
                 "paper_runtime", "reconciliation", "drift_stability",
                 "watchdog_uptime", "halt_diversity"]:
        assert name in captured.out
