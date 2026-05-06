"""Add workout_session_ejercicios table

Revision ID: 008_add_workout_session_ejercicios
Revises: 007_add_workout_sessions
Create Date: 2026-05-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "008_workout_session_ej"
down_revision: Union[str, Sequence[str], None] = "007_add_workout_sessions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "workout_session_ejercicios",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("workout_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ejercicio_id", sa.String(100), sa.ForeignKey("ejercicios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("orden", sa.Integer(), server_default="1", nullable=False),
        sa.Column("series_completadas", sa.Integer(), nullable=True),
        sa.Column("reps_completadas", sa.Integer(), nullable=True),
        sa.Column("peso_kg", sa.Float(), nullable=True),
        sa.Column("duracion_s", sa.Integer(), nullable=True),
        sa.Column("calorias_quemadas", sa.Float(), nullable=True),
        sa.Column("notas", sa.Text(), nullable=True),
    )
    op.create_index("idx_wse_session", "workout_session_ejercicios", ["session_id"])
    op.create_index("idx_wse_ejercicio", "workout_session_ejercicios", ["ejercicio_id"])
    op.create_index("idx_wse_orden", "workout_session_ejercicios", ["session_id", "orden"])


def downgrade() -> None:
    op.drop_index("idx_wse_orden", table_name="workout_session_ejercicios")
    op.drop_index("idx_wse_ejercicio", table_name="workout_session_ejercicios")
    op.drop_index("idx_wse_session", table_name="workout_session_ejercicios")
    op.drop_table("workout_session_ejercicios")
