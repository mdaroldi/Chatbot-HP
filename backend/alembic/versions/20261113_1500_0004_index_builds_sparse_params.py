"""index_builds.sparse_params: the BM25 parameters of a build's sparse vectors (ADR 0009)

NULL on rows built before this revision: fastembed's defaults (k 1.2, b 0.75,
avg_len 256), which those builds used.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-13 15:00:00.000000+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("index_builds", sa.Column("sparse_params", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("index_builds", "sparse_params")
