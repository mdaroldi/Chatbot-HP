"""conversations and messages: chat history scoped to a client ID (Phase 3)

The check constraints repeat the rules of app.domain.chat.Message, so the database
rejects a row that contradicts them even when it bypasses the domain types. The
composite foreign key (conversation_id, client_id) means a message can only belong to
a conversation of the same client; deleting a conversation deletes its messages.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-13 19:00:00.000000+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen copies of the rule text in app/infra/orm.py: a migration must not change when
# the models do later. The shared violation cases test both (SQLite and Postgres).
EMPTY_ANSWER = "citations = '[]' AND sources = '[]'"
NO_ANSWER_TRACE = (
    "chunk_ids = '[]' AND flags = '{}' AND error_code IS NULL AND model IS NULL "
    "AND input_tokens IS NULL AND output_tokens IS NULL AND dense_embed_ms IS NULL "
    "AND sparse_embed_ms IS NULL AND search_ms IS NULL AND rerank_ms IS NULL "
    "AND retrieval_ms IS NULL AND rewrite_ms IS NULL AND ttft_ms IS NULL "
    "AND generation_ms IS NULL AND total_ms IS NULL AND relevance_signal IS NULL "
    "AND relevance_threshold IS NULL AND index_build_id IS NULL AND rewritten_query IS NULL"
)


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("client_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=80), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "title IS NULL OR length(title) >= 1", name=op.f("ck_conversations_title_not_empty")
        ),
        sa.CheckConstraint(
            "updated_at >= created_at",
            name=op.f("ck_conversations_updated_not_before_created"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
        sa.UniqueConstraint("id", "client_id", name="uq_conversations_id_client_id"),
    )
    op.create_index(
        "ix_conversations_client_id_updated_at_id",
        "conversations",
        ["client_id", "updated_at", "id"],
    )
    op.create_index("ix_conversations_updated_at", "conversations", ["updated_at"])

    jsonb = postgresql.JSONB(astext_type=sa.Text())
    timing: sa.Double[float] = sa.Double()
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("conversation_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("client_id", sa.String(length=64), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=True),
        sa.Column("refusal", sa.Boolean(), nullable=False),
        sa.Column("refusal_reason", sa.String(length=32), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("flags", jsonb, nullable=False),
        sa.Column("citations", jsonb, nullable=False),
        sa.Column("sources", jsonb, nullable=False),
        sa.Column("chunk_ids", jsonb, nullable=False),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("dense_embed_ms", timing, nullable=True),
        sa.Column("sparse_embed_ms", timing, nullable=True),
        sa.Column("search_ms", timing, nullable=True),
        sa.Column("rerank_ms", timing, nullable=True),
        sa.Column("retrieval_ms", timing, nullable=True),
        sa.Column("rewrite_ms", timing, nullable=True),
        sa.Column("ttft_ms", timing, nullable=True),
        sa.Column("generation_ms", timing, nullable=True),
        sa.Column("total_ms", timing, nullable=True),
        sa.Column("relevance_signal", timing, nullable=True),
        sa.Column("relevance_threshold", timing, nullable=True),
        sa.Column("index_build_id", sa.String(length=32), nullable=True),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.CheckConstraint("seq >= 1", name=op.f("ck_messages_seq_positive")),
        sa.CheckConstraint("role IN ('user', 'assistant')", name=op.f("ck_messages_role")),
        sa.CheckConstraint(
            "outcome IN ('pending', 'completed', 'refused', 'failed', 'cancelled')",
            name=op.f("ck_messages_outcome"),
        ),
        sa.CheckConstraint(
            "refusal_reason IN ('below_threshold', 'model_refused', 'provider_refused')",
            name=op.f("ck_messages_refusal_reason"),
        ),
        sa.CheckConstraint(
            "(role = 'assistant') = (outcome IS NOT NULL)", name=op.f("ck_messages_outcome_role")
        ),
        sa.CheckConstraint(
            "(refusal_reason IS NOT NULL) = (outcome IS NOT NULL AND outcome = 'refused')",
            name=op.f("ck_messages_refusal_reason_outcome"),
        ),
        sa.CheckConstraint(
            "refusal = (refusal_reason IS NOT NULL)", name=op.f("ck_messages_refusal_flag")
        ),
        sa.CheckConstraint(
            f"role = 'assistant' OR ({EMPTY_ANSWER})",
            name=op.f("ck_messages_user_has_no_citations"),
        ),
        sa.CheckConstraint(
            f"outcome IS NULL OR outcome <> 'refused' OR ({EMPTY_ANSWER})",
            name=op.f("ck_messages_refusal_has_no_citations"),
        ),
        sa.CheckConstraint(
            f"role = 'assistant' OR ({NO_ANSWER_TRACE})",
            name=op.f("ck_messages_user_has_no_answer_trace"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id", "client_id"],
            ["conversations.id", "conversations.client_id"],
            name=op.f("fk_messages_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint("conversation_id", "seq", name="uq_messages_conversation_id_seq"),
    )
    op.create_index(
        "ix_messages_one_pending",
        "messages",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text("outcome = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index("ix_messages_one_pending", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_conversations_updated_at", table_name="conversations")
    op.drop_index("ix_conversations_client_id_updated_at_id", table_name="conversations")
    op.drop_table("conversations")
