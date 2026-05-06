"""Add rutinas table

Revision ID: 005_add_rutinas
Revises: 004_add_app_cache_rutinas
Create Date: 2026-05-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "005_add_rutinas"
down_revision: Union[str, Sequence[str], None] = "004_add_app_cache_rutinas"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rutinas",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("nombre", sa.String(255), nullable=False),
        sa.Column("descripcion", sa.Text(), nullable=True),
        sa.Column("perfil_tipo", sa.String(16), nullable=True),
        sa.Column("nivel", sa.String(32), nullable=True),
        sa.Column("grupo_muscular", sa.String(128), nullable=True),
        sa.Column("tiempo_min", sa.Integer(), nullable=True),
        sa.Column("series_config", sa.Text(), nullable=True),
        sa.Column("origen", sa.String(32), server_default="llm", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_rutinas_perfil", "rutinas", ["perfil_tipo"])
    op.create_index("idx_rutinas_grupo", "rutinas", ["grupo_muscular"])


def downgrade() -> None:
    op.drop_index("idx_rutinas_grupo", table_name="rutinas")
    op.drop_index("idx_rutinas_perfil", table_name="rutinas")
    op.drop_table("rutinas")
