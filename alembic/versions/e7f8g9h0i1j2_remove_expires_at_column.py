"""Remove expires_at column from password_resets table

Revision ID: e7f8g9h0i1j2
Revises: a1b2c3d4e5f6
Create Date: 2026-01-15 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7f8g9h0i1j2'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Remove expires_at column (now calculated property)."""
    # Remover la columna expires_at (ahora es una propiedad @property)
    op.drop_column('password_resets', 'expires_at')


def downgrade() -> None:
    """Downgrade schema - Add expires_at column back."""
    # Restaurar columna si se necesita rollback
    op.add_column(
        'password_resets',
        sa.Column('expires_at', sa.DateTime(), nullable=False)
    )
