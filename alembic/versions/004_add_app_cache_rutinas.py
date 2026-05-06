"""Add app_cache_rutinas table

Revision ID: 004_add_app_cache_rutinas
Revises: 003_add_alimentos_sin_resolver
Create Date: 2026-05-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004_add_app_cache_rutinas"
down_revision: Union[str, Sequence[str], None] = "003_add_alimentos_sin_resolver"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "app_cache_rutinas",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cache_key", sa.String(512), nullable=False, unique=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("clients.id", ondelete="CASCADE"), nullable=True),
        sa.Column("perfil_tipo", sa.String(16), nullable=True),
        sa.Column("zonas_objetivo", sa.Text(), nullable=True),
        sa.Column("tiempo_min", sa.Integer(), nullable=True),
        sa.Column("rutina_json", sa.Text(), nullable=False),
        sa.Column("hit_count", sa.Integer(), server_default="1", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_cache_rutinas_key", "app_cache_rutinas", ["cache_key"])
    op.create_index("idx_cache_rutinas_user", "app_cache_rutinas", ["user_id"])
    op.create_index("idx_cache_rutinas_expires", "app_cache_rutinas", ["expires_at"])


def downgrade() -> None:
    op.drop_index("idx_cache_rutinas_expires", table_name="app_cache_rutinas")
    op.drop_index("idx_cache_rutinas_user", table_name="app_cache_rutinas")
    op.drop_index("idx_cache_rutinas_key", table_name="app_cache_rutinas")
    op.drop_table("app_cache_rutinas")
