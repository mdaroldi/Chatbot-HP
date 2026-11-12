"""index_builds: one row per index build that started (ADR 0006)

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-12 16:55:20.930297+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "index_builds",
        sa.Column("build_id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("collection", sa.String(length=255), nullable=False),
        sa.Column("document_hashes", sa.JSON(), nullable=False),
        sa.Column("chunking_version", sa.String(length=32), nullable=False),
        sa.Column("dense_model", sa.String(length=255), nullable=False),
        sa.Column("sparse_model", sa.String(length=255), nullable=False),
        sa.Column("models", sa.JSON(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("chunk_counts", sa.JSON(), nullable=False),
        sa.Column("drop_stats", sa.JSON(), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "status IN ('succeeded', 'failed')", name=op.f("ck_index_builds_status")
        ),
        sa.CheckConstraint(
            "chunk_count >= 0", name=op.f("ck_index_builds_chunk_count_non_negative")
        ),
        sa.PrimaryKeyConstraint("build_id", name=op.f("pk_index_builds")),
        sa.UniqueConstraint("collection", name=op.f("uq_index_builds_collection")),
    )


def downgrade() -> None:
    op.drop_table("index_builds")
