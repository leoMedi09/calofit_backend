"""Add workout_sessions table

Revision ID: 007_add_workout_sessions
Revises: 006_add_rutinas_ejercicios
Create Date: 2026-05-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "007_add_workout_sessions"
down_revision: Union[str, Sequence[str], None] = "006_add_rutinas_ejercicios"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "workout_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("client_id", sa.Integer(), sa.ForeignKey("clients.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rutina_id", sa.Integer(), sa.ForeignKey("rutinas.id", ondelete="SET NULL"), nullable=True),
        sa.Column("nombre_rutina", sa.String(255), nullable=True),
        sa.Column("fecha", sa.Date(), nullable=False),
        sa.Column("duracion_min", sa.Integer(), nullable=True),
        sa.Column("calorias_quemadas", sa.Float(), nullable=True),
        sa.Column("intensity", sa.String(16), nullable=True),
        sa.Column("notas", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_ws_client", "workout_sessions", ["client_id"])
    op.create_index("idx_ws_fecha", "workout_sessions", ["client_id", "fecha"])
    op.create_index("idx_ws_rutina", "workout_sessions", ["rutina_id"])


def downgrade() -> None:
    op.drop_index("idx_ws_rutina", table_name="workout_sessions")
    op.drop_index("idx_ws_fecha", table_name="workout_sessions")
    op.drop_index("idx_ws_client", table_name="workout_sessions")
    op.drop_table("workout_sessions")
