"""Add app_cache_alimentos table

Revision ID: 001_add_app_cache_alimentos
Revises: b31c6b8d9a10
Create Date: 2026-05-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001_add_app_cache_alimentos"
down_revision: Union[str, Sequence[str], None] = "b31c6b8d9a10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "app_cache_alimentos",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("food_normalized", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("clients.id", ondelete="SET NULL"), nullable=True),
        sa.Column("alimento_id", sa.Integer(), sa.ForeignKey("alimentos.id", ondelete="CASCADE"), nullable=True),
        sa.Column("source", sa.String(64), nullable=True),
        sa.Column("raw_response", sa.Text(), nullable=True),
        sa.Column("hit_count", sa.Integer(), server_default="1", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_cache_food", "app_cache_alimentos", ["food_normalized"])
    op.create_index("idx_cache_food_user", "app_cache_alimentos", ["food_normalized", "user_id"])
    op.create_index("idx_cache_expires", "app_cache_alimentos", ["expires_at"])


def downgrade() -> None:
    op.drop_index("idx_cache_expires", table_name="app_cache_alimentos")
    op.drop_index("idx_cache_food_user", table_name="app_cache_alimentos")
    op.drop_index("idx_cache_food", table_name="app_cache_alimentos")
    op.drop_table("app_cache_alimentos")
