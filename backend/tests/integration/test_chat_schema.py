"""Migration 0005 on the real Postgres: the chat tables enforce the domain rules themselves.

Each row in MESSAGE_VIOLATIONS breaks exactly one Message rule from app.domain.chat;
the database must reject it with that rule's constraint, even though the row never
went through the domain types.
"""

import pathlib
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.infra.db import create_engine
from app.infra.orm import ConversationRow, MessageRow
from tests.chat_rows import (
    CONVERSATION_ID,
    CONVERSATION_VIOLATIONS,
    MESSAGE_VIOLATIONS,
    OTHER_CLIENT,
    VALID_MESSAGES,
    Row,
    conversation_row,
    insert_conversation,
    insert_message,
    message_row,
)
from tests.integration.conftest import SettingsFactory, run_alembic

pytestmark = pytest.mark.integration

OTHER_CONVERSATION_ID = "00000000-0000-4000-8000-000000000002"


@pytest.fixture
async def engine(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> AsyncIterator[AsyncEngine]:
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    engine = create_engine(make_settings(postgres_db=scratch_db))
    yield engine
    await engine.dispose()


async def rejection(engine: AsyncEngine, table: str, row: Row) -> str | None:
    """The error text when the database refuses the row, None when it accepts it."""
    insert = insert_message if table == "messages" else insert_conversation
    try:
        async with engine.begin() as connection:
            await insert(connection, row)
    except IntegrityError as error:
        return str(error.orig)
    return None


def unmet(cases: list[tuple[str, str | None, str]]) -> list[str]:
    return [
        f"{case_id}: expected {constraint}, got {'acceptance' if got is None else got}"
        for case_id, got, constraint in cases
        if got is None or constraint not in got
    ]


async def test_valid_chat_rows_are_accepted(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await insert_conversation(connection, conversation_row(title="Synthetic"))
        for row in VALID_MESSAGES:
            await insert_message(connection, row)

    async with engine.connect() as connection:
        count = (await connection.execute(select(func.count()).select_from(MessageRow))).scalar()
    assert count == len(VALID_MESSAGES)


async def test_every_message_rule_is_enforced_by_the_database(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await insert_conversation(connection, conversation_row())

    cases = []
    for case in MESSAGE_VIOLATIONS:
        row, constraint = case.values
        assert isinstance(row, dict)
        assert isinstance(constraint, str)
        cases.append((str(case.id), await rejection(engine, "messages", row), constraint))

    assert unmet(cases) == []
    assert len(cases) == len(MESSAGE_VIOLATIONS) > 30


async def test_every_conversation_rule_is_enforced_by_the_database(engine: AsyncEngine) -> None:
    cases = []
    for case in CONVERSATION_VIOLATIONS:
        row, constraint = case.values
        assert isinstance(row, dict)
        assert isinstance(constraint, str)
        cases.append((str(case.id), await rejection(engine, "conversations", row), constraint))

    assert unmet(cases) == []


async def test_a_message_must_belong_to_a_conversation_of_the_same_client(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as connection:
        await insert_conversation(connection, conversation_row())

    got = await rejection(engine, "messages", message_row(1, client_id=OTHER_CLIENT))

    assert got is not None
    assert "fk_messages_conversation_id_conversations" in got


async def test_sequence_numbers_are_unique_per_conversation(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await insert_conversation(connection, conversation_row())
        await insert_conversation(connection, conversation_row(id=OTHER_CONVERSATION_ID))
        await insert_message(connection, message_row(1))
    # The same seq in another conversation is fine.
    other = message_row(2, conversation_id=OTHER_CONVERSATION_ID) | {"seq": 1}
    assert await rejection(engine, "messages", other) is None

    got = await rejection(engine, "messages", message_row(3) | {"seq": 1})

    assert got is not None
    assert "uq_messages_conversation_id_seq" in got


async def test_a_conversation_has_at_most_one_pending_answer(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await insert_conversation(connection, conversation_row())
        await insert_message(connection, message_row(1, "assistant", outcome="pending"))
        await insert_message(connection, message_row(2, "assistant", outcome="failed"))
    assert (
        await rejection(engine, "messages", message_row(3, "assistant", outcome="failed")) is None
    )

    got = await rejection(engine, "messages", message_row(4, "assistant", outcome="pending"))

    assert got is not None
    assert "ix_messages_one_pending" in got


async def test_deleting_a_conversation_deletes_its_messages(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await insert_conversation(connection, conversation_row())
        await insert_message(connection, message_row(1))
        await insert_message(connection, message_row(2, "assistant"))

    async with engine.begin() as connection:
        await connection.execute(
            delete(ConversationRow).where(ConversationRow.id == CONVERSATION_ID)
        )

    async with engine.connect() as connection:
        count = (await connection.execute(select(func.count()).select_from(MessageRow))).scalar()
    assert count == 0


async def test_answer_json_columns_are_jsonb(engine: AsyncEngine) -> None:
    query = text(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'messages' AND column_name IN "
        "('citations', 'sources', 'chunk_ids', 'flags')"
    )
    async with engine.connect() as connection:
        found: dict[str, str] = dict((await connection.execute(query)).all())

    assert found == dict.fromkeys(("citations", "sources", "chunk_ids", "flags"), "jsonb")
