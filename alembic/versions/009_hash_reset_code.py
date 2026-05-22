"""hash reset_code column — store HMAC-SHA256 instead of plain text

Revision ID: 009_hash_reset_code
Revises: f9k0l1m2n3o4
Create Date: 2026-05-22
"""
from alembic import op
import sqlalchemy as sa

revision = '009_hash_reset_code'
down_revision = 'f9k0l1m2n3o4'
branch_labels = None
depends_on = None


def upgrade():
    # Ampliar columna y limpiar códigos plain-text existentes (ya expirados o inválidos)
    op.alter_column(
        'password_resets', 'reset_code',
        existing_type=sa.String(6),
        type_=sa.String(64),
        existing_nullable=False
    )
    # Marcar todos los registros previos como usados — sus códigos plain-text
    # ya no son verificables con el nuevo esquema de hashing
    op.execute("UPDATE password_resets SET is_used = TRUE WHERE LENGTH(reset_code) <= 6")


def downgrade():
    op.execute("DELETE FROM password_resets")
    op.alter_column(
        'password_resets', 'reset_code',
        existing_type=sa.String(64),
        type_=sa.String(6),
        existing_nullable=False
    )
