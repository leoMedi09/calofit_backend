"""Add app_cache_platos table

Revision ID: 002_add_app_cache_platos
Revises: 001_add_app_cache_alimentos
Create Date: 2026-05-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002_add_app_cache_platos"
down_revision: Union[str, Sequence[str], None] = "001_add_app_cache_alimentos"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "app_cache_platos",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("plato_normalized", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("clients.id", ondelete="SET NULL"), nullable=True),
        sa.Column("plato_id", sa.Integer(), sa.ForeignKey("platos.id", ondelete="CASCADE"), nullable=True),
        sa.Column("source", sa.String(64), nullable=True),
        sa.Column("hit_count", sa.Integer(), server_default="1", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_cache_plato_norm", "app_cache_platos", ["plato_normalized"])
    op.create_index("idx_cache_plato_user", "app_cache_platos", ["plato_normalized", "user_id"])
    op.create_index("idx_cache_plato_expires", "app_cache_platos", ["expires_at"])


def downgrade() -> None:
    op.drop_index("idx_cache_plato_expires", table_name="app_cache_platos")
    op.drop_index("idx_cache_plato_user", table_name="app_cache_platos")
    op.drop_index("idx_cache_plato_norm", table_name="app_cache_platos")
    op.drop_table("app_cache_platos")
