"""Add rutinas_ejercicios table

Revision ID: 006_add_rutinas_ejercicios
Revises: 005_add_rutinas
Create Date: 2026-05-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006_add_rutinas_ejercicios"
down_revision: Union[str, Sequence[str], None] = "005_add_rutinas"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rutinas_ejercicios",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("rutina_id", sa.Integer(), sa.ForeignKey("rutinas.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ejercicio_id", sa.String(100), sa.ForeignKey("ejercicios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("orden", sa.Integer(), server_default="1", nullable=False),
        sa.Column("series", sa.Integer(), server_default="3", nullable=False),
        sa.Column("reps", sa.Integer(), server_default="12", nullable=False),
        sa.Column("descanso_s", sa.Integer(), server_default="60", nullable=False),
        sa.Column("peso_sugerido_kg", sa.Float(), nullable=True),
        sa.Column("notas", sa.Text(), nullable=True),
    )
    op.create_index("idx_rut_ej_rutina", "rutinas_ejercicios", ["rutina_id"])
    op.create_index("idx_rut_ej_ejercicio", "rutinas_ejercicios", ["ejercicio_id"])
    op.create_index("idx_rut_ej_orden", "rutinas_ejercicios", ["rutina_id", "orden"])


def downgrade() -> None:
    op.drop_index("idx_rut_ej_orden", table_name="rutinas_ejercicios")
    op.drop_index("idx_rut_ej_ejercicio", table_name="rutinas_ejercicios")
    op.drop_index("idx_rut_ej_rutina", table_name="rutinas_ejercicios")
    op.drop_table("rutinas_ejercicios")
