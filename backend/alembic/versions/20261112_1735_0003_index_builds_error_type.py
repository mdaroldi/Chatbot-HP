"""index_builds.error_type: the exception type of a failed build (ADR 0006)

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-12 17:35:00.000000+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("index_builds", sa.Column("error_type", sa.String(length=255), nullable=True))
    op.create_check_constraint(
        op.f("ck_index_builds_error_type_status"),
        "index_builds",
        "(status = 'failed') = (error_type IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_index_builds_error_type_status"), "index_builds", type_="check")
    op.drop_column("index_builds", "error_type")
