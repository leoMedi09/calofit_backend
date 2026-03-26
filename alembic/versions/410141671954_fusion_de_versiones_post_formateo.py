"""fusion de versiones post-formateo

Revision ID: 410141671954
Revises: 004_nutricion, f9k0l1m2n3o4
Create Date: 2026-03-26 05:43:57.755047

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '410141671954'
down_revision: Union[str, Sequence[str], None] = ('004_nutricion', 'f9k0l1m2n3o4')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
