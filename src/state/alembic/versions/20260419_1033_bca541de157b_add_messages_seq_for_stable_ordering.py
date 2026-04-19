"""add messages.seq for stable ordering

Revision ID: bca541de157b
Revises: 08ae5a4d451d
Create Date: 2026-04-19 10:33:35.116835+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = 'bca541de157b'
down_revision = '08ae5a4d451d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("messages") as batch_op:
        batch_op.add_column(sa.Column("seq", sa.Integer, nullable=True))
    op.execute("UPDATE messages SET seq = rowid WHERE seq IS NULL")
    with op.batch_alter_table("messages") as batch_op:
        batch_op.alter_column("seq", nullable=False)
        batch_op.create_index("ix_messages_cid_seq", ["conversation_id", "seq"])


def downgrade() -> None:
    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_index("ix_messages_cid_seq")
        batch_op.drop_column("seq")
