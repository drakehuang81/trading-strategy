"""baseline_schema

Revision ID: 15fdbaffd2bf
Revises: 
Create Date: 2026-04-18 13:02:09.467133+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = '15fdbaffd2bf'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Core event log ──────────────────────────────────────────────
    op.create_table(
        "proposals",
        sa.Column("proposal_id", sa.String, primary_key=True),
        sa.Column("trace_id", sa.String, nullable=False, index=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("symbol", sa.String, nullable=False, index=True),
        sa.Column("direction", sa.String, nullable=False),
        sa.Column("entry", sa.Float, nullable=False),
        sa.Column("stop_loss", sa.Float, nullable=False),
        sa.Column("take_profit_json", sa.Text, nullable=False),
        sa.Column("size", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("feature_snapshot_json", sa.Text, nullable=False),
        sa.Column("bundle_json", sa.Text, nullable=False),
        sa.Column("risk_checks_json", sa.Text, nullable=False),
        sa.Column("accepted", sa.Boolean, nullable=False),
        sa.Column("rationale", sa.Text),
        sa.Column("feature_registry_version", sa.String, nullable=False),
        sa.Column("ml_model_version", sa.String, nullable=False),
        sa.Column("llm_prompt_version", sa.String, nullable=False),
    )

    op.create_table(
        "broker_events",
        sa.Column("event_id", sa.String, primary_key=True),   # idempotency key §8.3
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("order_id", sa.String, nullable=False, index=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("fill_price", sa.Float),
        sa.Column("fill_qty", sa.Float),
        sa.Column("fee", sa.Float),
        sa.Column("reason", sa.Text),
        sa.Column("ml_model_version", sa.String),
        sa.Column("llm_prompt_version", sa.String),
    )

    op.create_table(
        "positions",
        sa.Column("symbol", sa.String, primary_key=True),
        sa.Column("qty", sa.Float, nullable=False),
        sa.Column("avg_entry", sa.Float, nullable=False),
        sa.Column("opened_at", sa.DateTime, nullable=False),
        sa.Column("last_update_ts", sa.DateTime, nullable=False),
    )

    op.create_table(
        "fills",
        sa.Column("fill_id", sa.String, primary_key=True),
        sa.Column("event_id", sa.String, nullable=False, index=True),
        sa.Column("order_id", sa.String, nullable=False, index=True),
        sa.Column("symbol", sa.String, nullable=False),
        sa.Column("ts", sa.DateTime, nullable=False),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("qty", sa.Float, nullable=False),
        sa.Column("fee", sa.Float, nullable=False),
    )

    # ── Prediction / reconciliation / conversations ─────────────────
    op.create_table(
        "prediction_disagreements",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("trace_id", sa.String, nullable=False, index=True),
        sa.Column("ts", sa.DateTime, nullable=False),
        sa.Column("ml_direction", sa.String, nullable=False),
        sa.Column("llm_veto", sa.Boolean, nullable=False),
        sa.Column("detail_json", sa.Text, nullable=False),
    )

    op.create_table(
        "reconciliation_diffs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("kind", sa.String, nullable=False),  # position | balance
        sa.Column("diff_json", sa.Text, nullable=False),
        sa.Column("resolution", sa.String, nullable=False),  # auto_repaired | user_accepted | halted
    )

    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.String, primary_key=True),
        sa.Column("started_at", sa.DateTime, nullable=False),
        sa.Column("telegram_chat_id", sa.String, nullable=False, index=True),
    )
    op.create_table(
        "messages",
        sa.Column("message_id", sa.String, primary_key=True),
        sa.Column("conversation_id", sa.String, nullable=False, index=True),
        sa.Column("ts", sa.DateTime, nullable=False),
        sa.Column("role", sa.String, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
    )
    op.create_table(
        "tool_calls",
        sa.Column("tool_call_id", sa.String, primary_key=True),
        sa.Column("message_id", sa.String, nullable=False, index=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("args_json", sa.Text, nullable=False),
        sa.Column("result_json", sa.Text, nullable=False),
    )

    # ── Ops / observability ─────────────────────────────────────────
    op.create_table(
        "heartbeat",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("trace_id", sa.String, nullable=False),
    )

    op.create_table(
        "halt_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("activated_at", sa.DateTime, nullable=False, index=True),
        sa.Column("trigger_source", sa.String, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("resumed_at", sa.DateTime),
    )

    op.create_table(
        "log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("level", sa.String, nullable=False),
        sa.Column("trace_id", sa.String, nullable=False, index=True),
        sa.Column("event", sa.String, nullable=False),
        sa.Column("payload_json", sa.Text, nullable=False),
    )

    op.create_table(
        "backtest_runs",
        sa.Column("run_id", sa.String, primary_key=True),
        sa.Column("started_at", sa.DateTime, nullable=False),
        sa.Column("deflated_sharpe", sa.Float),
        sa.Column("cost_model_version", sa.String, nullable=False),
        sa.Column("summary_json", sa.Text, nullable=False),
    )

    op.create_table(
        "session_state",
        sa.Column("date", sa.Date, primary_key=True),
        sa.Column("consecutive_wins", sa.Integer, nullable=False),
        sa.Column("day_pnl_r", sa.Float, nullable=False),
        sa.Column("last_update_ts", sa.DateTime, nullable=False),
    )

    op.create_table(
        "dead_letter",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("source", sa.String, nullable=False),
        sa.Column("payload_json", sa.Text, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
    )

    op.create_table(
        "model_versions",
        sa.Column("ml_model_version", sa.String, primary_key=True),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("training_window_start", sa.DateTime, nullable=False),
        sa.Column("training_window_end", sa.DateTime, nullable=False),
        sa.Column("calibration_method", sa.String, nullable=False),
        sa.Column("deployed_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "feature_cache_manifest",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String, nullable=False, index=True),
        sa.Column("timeframe", sa.String, nullable=False),
        sa.Column("feature_name", sa.String, nullable=False, index=True),
        sa.Column("feature_version", sa.String, nullable=False),
        sa.Column("as_of_start", sa.DateTime, nullable=False),
        sa.Column("as_of_end", sa.DateTime, nullable=False),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("row_count", sa.Integer, nullable=False),
    )


def downgrade() -> None:
    for table in [
        "feature_cache_manifest",
        "model_versions",
        "dead_letter",
        "session_state",
        "backtest_runs",
        "log",
        "halt_events",
        "heartbeat",
        "tool_calls",
        "messages",
        "conversations",
        "reconciliation_diffs",
        "prediction_disagreements",
        "fills",
        "positions",
        "broker_events",
        "proposals",
    ]:
        op.drop_table(table)
