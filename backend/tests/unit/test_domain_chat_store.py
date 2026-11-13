"""What a finished answer records and the types the chat store port exchanges."""

from datetime import UTC, datetime

import pytest

from app.domain.chat import (
    AnswerFlags,
    AnswerRecord,
    Message,
    Outcome,
    RefusalReason,
    Role,
    Source,
)
from app.domain.chat_store import (
    ConversationBusyError,
    ConversationNotFoundError,
    Turn,
)
from app.domain.ingestion import PageRange

T0 = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)
SOURCE = Source(
    "S1", "synthetic-doc", "Synthetic manual", ("A", "B"), PageRange(1, 1, None, None), False, "x"
)


def record(**overrides: object) -> AnswerRecord:
    fields: dict[str, object] = {"outcome": Outcome.COMPLETED, "content": "An answer [S1]."}
    fields.update(overrides)
    return AnswerRecord(**fields)  # type: ignore[arg-type]  # fields typed per test


def test_a_completed_record_defaults_to_an_empty_trace() -> None:
    answer = record()

    assert (answer.citations, answer.sources, answer.chunk_ids) == ((), (), ())
    assert answer.flags == AnswerFlags()
    assert answer.refusal_reason is None
    assert answer.model is None


def test_a_record_cannot_be_pending() -> None:
    with pytest.raises(ValueError, match="a finished answer cannot be pending"):
        record(outcome=Outcome.PENDING)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"outcome": Outcome.REFUSED}, "a refusal needs a reason"),
        ({"refusal_reason": RefusalReason.MODEL_REFUSED}, "only a refusal has a reason"),
        (
            {
                "outcome": Outcome.REFUSED,
                "refusal_reason": RefusalReason.BELOW_THRESHOLD,
                "citations": ("S1",),
            },
            "a refusal has no citations or sources",
        ),
        (
            {
                "outcome": Outcome.REFUSED,
                "refusal_reason": RefusalReason.BELOW_THRESHOLD,
                "sources": (SOURCE,),
            },
            "a refusal has no citations or sources",
        ),
    ],
)
def test_a_record_follows_the_message_refusal_rules(
    overrides: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        record(**overrides)


def test_a_refusal_record_with_its_reason_is_valid() -> None:
    refused = record(outcome=Outcome.REFUSED, refusal_reason=RefusalReason.PROVIDER_REFUSED)

    assert refused.refusal_reason is RefusalReason.PROVIDER_REFUSED


def test_the_rewrite_fallback_flag_holds_a_code_or_nothing() -> None:
    assert AnswerFlags().rewrite_fallback is None
    assert AnswerFlags(rewrite_fallback="timeout").rewrite_fallback == "timeout"
    for bad in ("", "Timed out!", "two words"):
        with pytest.raises(ValueError, match="rewrite_fallback must be a code"):
            AnswerFlags(rewrite_fallback=bad)


def test_invalid_citation_count_is_never_negative() -> None:
    assert AnswerFlags(invalid_citations=2).invalid_citations == 2
    with pytest.raises(ValueError, match="invalid_citations must not be negative"):
        AnswerFlags(invalid_citations=-1)


def message(seq: int, role: Role, **overrides: object) -> Message:
    fields: dict[str, object] = {
        "id": f"m-{seq}",
        "conversation_id": "c-1",
        "seq": seq,
        "role": role,
        "content": "text" if role is Role.USER else "",
        "created_at": T0,
        "outcome": Outcome.PENDING if role is Role.ASSISTANT else None,
    }
    fields.update(overrides)
    return Message(**fields)  # type: ignore[arg-type]  # fields typed per test


def test_a_turn_is_a_user_message_followed_by_its_pending_answer() -> None:
    turn = Turn(message(3, Role.USER), message(4, Role.ASSISTANT))

    assert (turn.user.seq, turn.assistant.seq) == (3, 4)


@pytest.mark.parametrize(
    ("user", "assistant"),
    [
        (message(1, Role.ASSISTANT), message(2, Role.ASSISTANT)),
        (message(1, Role.USER), message(2, Role.USER)),
        (message(1, Role.USER), message(2, Role.ASSISTANT, outcome=Outcome.COMPLETED)),
        (message(1, Role.USER), message(3, Role.ASSISTANT)),
        (message(1, Role.USER), message(2, Role.ASSISTANT, conversation_id="c-2")),
    ],
)
def test_a_turn_rejects_anything_else(user: Message, assistant: Message) -> None:
    with pytest.raises(ValueError, match="a turn is a user message and its pending answer"):
        Turn(user, assistant)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ConversationNotFoundError, "conversation_not_found"),
        (ConversationBusyError, "conversation_busy"),
    ],
)
def test_store_errors_carry_a_fixed_code_and_message(
    error: type[ConversationNotFoundError | ConversationBusyError], code: str
) -> None:
    raised = error()

    assert raised.code == code
    assert str(raised) == raised.message
    assert raised.message
