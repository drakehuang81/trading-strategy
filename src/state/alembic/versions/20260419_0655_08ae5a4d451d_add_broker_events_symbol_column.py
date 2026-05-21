"""add broker_events symbol column

Revision ID: 08ae5a4d451d
Revises: 15fdbaffd2bf
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers — keep the generated values
revision = "08ae5a4d451d"
down_revision = "15fdbaffd2bf"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("broker_events") as batch_op:
        batch_op.add_column(sa.Column("symbol", sa.String(20), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("broker_events") as batch_op:
        batch_op.drop_column("symbol")
