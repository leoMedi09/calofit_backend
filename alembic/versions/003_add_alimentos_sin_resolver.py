"""Add alimentos_sin_resolver table

Revision ID: 003_add_alimentos_sin_resolver
Revises: 002_add_app_cache_platos
Create Date: 2026-05-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003_add_alimentos_sin_resolver"
down_revision: Union[str, Sequence[str], None] = "002_add_app_cache_platos"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "alimentos_sin_resolver",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("nombre_original", sa.String(512), nullable=False),
        sa.Column("nombre_normalizado", sa.String(512), nullable=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("clients.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reporter_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("mensaje_contexto", sa.Text(), nullable=True),
        sa.Column("intentos", sa.Integer(), server_default="1", nullable=False),
        sa.Column("estado", sa.String(32), server_default="pendiente", nullable=False),
        sa.Column("notas", sa.Text(), nullable=True),
        sa.Column("fecha_reporte", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("fecha_resolucion", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_sin_resolver_nombre", "alimentos_sin_resolver", ["nombre_normalizado"])
    op.create_index("idx_sin_resolver_estado", "alimentos_sin_resolver", ["estado"])
    op.create_index("idx_sin_resolver_user", "alimentos_sin_resolver", ["user_id"])


def downgrade() -> None:
    op.drop_index("idx_sin_resolver_user", table_name="alimentos_sin_resolver")
    op.drop_index("idx_sin_resolver_estado", table_name="alimentos_sin_resolver")
    op.drop_index("idx_sin_resolver_nombre", table_name="alimentos_sin_resolver")
    op.drop_table("alimentos_sin_resolver")
