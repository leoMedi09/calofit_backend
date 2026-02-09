"""Create password_resets table with new schema

Revision ID: f9k0l1m2n3o4
Revises: None (o la última migración)
Create Date: 2026-01-15 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f9k0l1m2n3o4'
down_revision: Union[str, Sequence[str], None] = None  # Cambiar a la última migración si existe
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Create password_resets table."""
    op.create_table(
        'password_resets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('reset_code', sa.String(length=6), nullable=False),
        sa.Column('is_used', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('used_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    # Crear índices para búsquedas rápidas
    op.create_index(op.f('ix_password_resets_email'), 'password_resets', ['email'], unique=False)
    op.create_index(op.f('ix_password_resets_reset_code'), 'password_resets', ['reset_code'], unique=False)


def downgrade() -> None:
    """Downgrade schema - Drop password_resets table."""
    op.drop_index(op.f('ix_password_resets_reset_code'), table_name='password_resets')
    op.drop_index(op.f('ix_password_resets_email'), table_name='password_resets')
    op.drop_table('password_resets')
