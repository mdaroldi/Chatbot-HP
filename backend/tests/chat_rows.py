"""Synthetic chat rows that bypass the domain, so only database constraints can stop them.

Shared by the SQLite unit tests (tables from the ORM metadata) and the Postgres
integration tests (tables from migration 0005), which check the same rules.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from app.infra.orm import ConversationRow, MessageRow

T0 = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)
CLIENT = "client-a"
OTHER_CLIENT = "client-b"
CONVERSATION_ID = "00000000-0000-4000-8000-000000000001"

Row = dict[str, object]

SOURCE_JSON: Row = {
    "tag": "S1",
    "doc_id": "synthetic-doc",
    "document_title": "Synthetic manual",
    "section_path": ["Chapter", "Section"],
    "pages": {"pdf_start": 3, "pdf_end": 4, "printed_start": 1, "printed_end": 2},
    "has_warning": False,
    "snippet": "A short synthetic snippet.",
}

# One value per assistant-only column; a user row must leave every one of them empty.
ANSWER_TRACE: Row = {
    "error_code": "abandoned",
    "flags": {"uncited": True},
    "chunk_ids": ["chunk-1"],
    "model": "stub",
    "input_tokens": 10,
    "output_tokens": 5,
    "dense_embed_ms": 1.0,
    "sparse_embed_ms": 1.0,
    "search_ms": 1.0,
    "rerank_ms": 1.0,
    "retrieval_ms": 4.0,
    "rewrite_ms": 1.0,
    "ttft_ms": 1.0,
    "generation_ms": 1.0,
    "total_ms": 9.0,
    "relevance_signal": -1.5,
    "relevance_threshold": -6.54,
    "index_build_id": "20261113t173108z",
    "rewritten_query": "a synthetic standalone query",
}


def conversation_row(**overrides: object) -> Row:
    return {
        "id": CONVERSATION_ID,
        "client_id": CLIENT,
        "title": None,
        "created_at": T0,
        "updated_at": T0,
    } | overrides


def message_row(seq: int, role: str = "user", **overrides: object) -> Row:
    row: Row = {
        "id": f"00000000-0000-4000-8000-{seq:012d}",
        "conversation_id": CONVERSATION_ID,
        "client_id": CLIENT,
        "seq": seq,
        "role": role,
        "content": "synthetic text",
        "created_at": T0,
        "outcome": "completed" if role == "assistant" else None,
        "refusal": False,
        "citations": [],
        "sources": [],
        "chunk_ids": [],
        "flags": {},
    }
    return row | overrides


REFUSED: Row = {"outcome": "refused", "refusal": True, "refusal_reason": "model_refused"}

VALID_MESSAGES = [
    message_row(1),
    message_row(2, "assistant", citations=["S1"], sources=[SOURCE_JSON], **ANSWER_TRACE),
    message_row(3),
    message_row(4, "assistant", **REFUSED),
    message_row(5),
    message_row(6, "assistant", outcome="pending"),
]

# (row, the one constraint it breaks). Each row breaks exactly one rule, so a check
# that went missing or got merged into another shows up as a wrong or absent name.
MESSAGE_VIOLATIONS = [
    pytest.param(message_row(1) | {"seq": 0}, "ck_messages_seq_positive", id="seq-zero"),
    pytest.param(message_row(1, role="system"), "ck_messages_role", id="unknown-role"),
    pytest.param(
        message_row(1, "assistant", outcome="done"), "ck_messages_outcome", id="unknown-outcome"
    ),
    pytest.param(
        message_row(1, "assistant", **(REFUSED | {"refusal_reason": "bored"})),
        "ck_messages_refusal_reason",
        id="unknown-refusal-reason",
    ),
    pytest.param(
        message_row(1, outcome="completed"), "ck_messages_outcome_role", id="user-with-outcome"
    ),
    pytest.param(
        message_row(1, "assistant", outcome=None),
        "ck_messages_outcome_role",
        id="assistant-without-outcome",
    ),
    pytest.param(
        message_row(1, "assistant", outcome="refused"),
        "ck_messages_refusal_reason_outcome",
        id="refusal-without-reason",
    ),
    pytest.param(
        message_row(1, "assistant", refusal=True, refusal_reason="model_refused"),
        "ck_messages_refusal_reason_outcome",
        id="reason-without-refusal",
    ),
    pytest.param(
        message_row(1, refusal=True, refusal_reason="model_refused"),
        "ck_messages_refusal_reason_outcome",
        id="user-with-reason",
    ),
    pytest.param(
        message_row(1, citations=["S1"]), "ck_messages_user_has_no_citations", id="user-citations"
    ),
    pytest.param(
        message_row(1, sources=[SOURCE_JSON]),
        "ck_messages_user_has_no_citations",
        id="user-sources",
    ),
    pytest.param(
        message_row(1, "assistant", **(REFUSED | {"citations": ["S1"]})),
        "ck_messages_refusal_has_no_citations",
        id="refusal-citations",
    ),
    pytest.param(
        message_row(1, "assistant", **(REFUSED | {"sources": [SOURCE_JSON]})),
        "ck_messages_refusal_has_no_citations",
        id="refusal-sources",
    ),
    pytest.param(
        message_row(1, "assistant", refusal=True),
        "ck_messages_refusal_flag",
        id="flag-without-reason",
    ),
    pytest.param(
        message_row(1, "assistant", **(REFUSED | {"refusal": False})),
        "ck_messages_refusal_flag",
        id="reason-without-flag",
    ),
    *(
        pytest.param(
            message_row(1) | {column: value},
            "ck_messages_user_has_no_answer_trace",
            id=f"user-{column}",
        )
        for column, value in ANSWER_TRACE.items()
    ),
]

CONVERSATION_VIOLATIONS = [
    pytest.param(conversation_row(title=""), "ck_conversations_title_not_empty", id="empty-title"),
    pytest.param(
        conversation_row(updated_at=T0 - timedelta(seconds=1)),
        "ck_conversations_updated_not_before_created",
        id="updated-before-created",
    ),
]


async def insert_conversation(connection: AsyncConnection, row: Row) -> None:
    await connection.execute(insert(ConversationRow), [row])


async def insert_message(connection: AsyncConnection, row: Row) -> None:
    await connection.execute(insert(MessageRow), [row])


# Integrity errors a turn can meet, and whether each one means "an answer is pending".
# Setup for every case: conversation_row() plus PENDING_ANSWER. Only the second pending
# answer is a busy conversation; every other violation is a bug and must surface.
PENDING_ANSWER = message_row(1, "assistant", outcome="pending")
CONFLICT_CASES = [
    pytest.param(message_row(2, "assistant", outcome="pending"), True, id="second-pending"),
    pytest.param(message_row(3) | {"seq": 1}, False, id="duplicate-seq"),
    pytest.param(message_row(4, client_id=OTHER_CLIENT), False, id="foreign-key"),
    pytest.param(message_row(5) | {"seq": 0}, False, id="check-constraint"),
]
