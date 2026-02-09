"""add password_reset table

Revision ID: a1b2c3d4e5f6
Revises: d07b43e35f9a
Create Date: 2026-01-15 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'd07b43e35f9a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Create password_resets table."""
    op.create_table(
        'password_resets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('reset_code', sa.String(length=6), nullable=False),
        sa.Column('is_used', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('used_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('reset_code')
    )
    # Crear índices para búsquedas rápidas
    op.create_index(op.f('ix_password_resets_email'), 'password_resets', ['email'], unique=False)
    op.create_index(op.f('ix_password_resets_reset_code'), 'password_resets', ['reset_code'], unique=False)


def downgrade() -> None:
    """Downgrade schema - Drop password_resets table."""
    op.drop_index(op.f('ix_password_resets_reset_code'), table_name='password_resets')
    op.drop_index(op.f('ix_password_resets_email'), table_name='password_resets')
    op.drop_table('password_resets')
