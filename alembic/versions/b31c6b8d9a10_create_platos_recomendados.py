"""create platos_recomendados

Revision ID: b31c6b8d9a10
Revises: 410141671954
Create Date: 2026-04-27

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b31c6b8d9a10"
down_revision: Union[str, Sequence[str], None] = "410141671954"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "platos_recomendados",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("nombre", sa.String(length=255), nullable=False),
        sa.Column("nombre_normalizado", sa.String(length=255), nullable=False),
        sa.Column("calorias", sa.Float(), nullable=False, server_default="0"),
        sa.Column("proteinas_g", sa.Float(), nullable=False, server_default="0"),
        sa.Column("carbohidratos_g", sa.Float(), nullable=False, server_default="0"),
        sa.Column("grasas_g", sa.Float(), nullable=False, server_default="0"),
        sa.Column("ingredientes", sa.JSON(), nullable=True),
        sa.Column("preparacion", sa.JSON(), nullable=True),
        sa.Column("nota", sa.Text(), nullable=True),
        sa.Column("origen", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_platos_recomendados_id", "platos_recomendados", ["id"])
    op.create_index(
        "ix_platos_recomendados_nombre_normalizado",
        "platos_recomendados",
        ["nombre_normalizado"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_platos_recomendados_nombre_normalizado", table_name="platos_recomendados")
    op.drop_index("ix_platos_recomendados_id", table_name="platos_recomendados")
    op.drop_table("platos_recomendados")

