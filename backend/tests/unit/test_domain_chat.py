"""Chat domain types: which messages are valid and how an answer ended."""

from datetime import UTC, datetime, timedelta

import pytest

from app.domain.chat import (
    REFUSAL_MARKER,
    Conversation,
    Message,
    Outcome,
    RefusalReason,
    Role,
    Source,
)
from app.domain.ingestion import PageRange

T0 = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)
PAGES = PageRange(3, 4, 1, 2)


def source(tag: str = "S1") -> Source:
    return Source(
        tag=tag,
        doc_id="synthetic-doc",
        document_title="Synthetic manual",
        section_path=("Chapter", "Section"),
        pages=PAGES,
        has_warning=False,
        snippet="A short synthetic snippet.",
    )


def message(role: Role = Role.ASSISTANT, **overrides: object) -> Message:
    fields: dict[str, object] = {
        "id": "m-1",
        "conversation_id": "c-1",
        "seq": 2,
        "role": role,
        "content": "An answer [S1].",
        "created_at": T0,
        "outcome": Outcome.COMPLETED if role is Role.ASSISTANT else None,
    }
    fields.update(overrides)
    return Message(**fields)  # type: ignore[arg-type]  # fields typed per test


def test_the_refusal_marker_is_a_fixed_token_the_prompt_can_name() -> None:
    assert REFUSAL_MARKER == "[[NOT_IN_DOCUMENTS]]"


def test_a_completed_answer_with_citations_is_not_a_refusal() -> None:
    answer = message(citations=("S1",), sources=(source(),))

    assert not answer.refusal
    assert answer.counts_as_turn


@pytest.mark.parametrize("reason", list(RefusalReason))
def test_a_refused_answer_names_its_reason_and_counts_as_a_turn(reason: RefusalReason) -> None:
    refused = message(outcome=Outcome.REFUSED, refusal_reason=reason, content="Not covered.")

    assert refused.refusal
    assert refused.counts_as_turn


@pytest.mark.parametrize("outcome", [Outcome.PENDING, Outcome.FAILED, Outcome.CANCELLED])
def test_unfinished_or_broken_answers_do_not_count_as_turns(outcome: Outcome) -> None:
    assert not message(outcome=outcome).counts_as_turn


def test_a_user_message_counts_as_a_turn_and_is_never_a_refusal() -> None:
    question = message(Role.USER, content="How do I do this?")

    assert question.counts_as_turn
    assert not question.refusal


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"outcome": Outcome.COMPLETED}, "user message has no outcome"),
        ({"refusal_reason": RefusalReason.MODEL_REFUSED}, "user message has no outcome"),
        ({"citations": ("S1",)}, "user message has no citations or sources"),
        ({"sources": (source(),)}, "user message has no citations or sources"),
    ],
)
def test_a_user_message_carries_no_answer_fields(overrides: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        message(Role.USER, **overrides)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"outcome": None}, "assistant message needs an outcome"),
        ({"outcome": Outcome.REFUSED}, "a refusal needs a reason"),
        ({"refusal_reason": RefusalReason.BELOW_THRESHOLD}, "only a refusal has a reason"),
        (
            {
                "outcome": Outcome.REFUSED,
                "refusal_reason": RefusalReason.MODEL_REFUSED,
                "citations": ("S1",),
            },
            "a refusal has no citations or sources",
        ),
        (
            {
                "outcome": Outcome.REFUSED,
                "refusal_reason": RefusalReason.MODEL_REFUSED,
                "sources": (source(),),
            },
            "a refusal has no citations or sources",
        ),
    ],
)
def test_assistant_message_rules(overrides: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        message(Role.ASSISTANT, **overrides)


def test_message_sequence_numbers_start_at_one() -> None:
    with pytest.raises(ValueError, match="seq must be at least 1"):
        message(seq=0)


@pytest.mark.parametrize("tag", ["S0", "s1", "X1", "S", "S1 "])
def test_a_source_tag_is_s_and_a_positive_number(tag: str) -> None:
    with pytest.raises(ValueError, match="source tag must look like S1"):
        source(tag)


def test_a_conversation_may_have_no_title_yet() -> None:
    assert Conversation("c-1", None, T0, T0).title is None


@pytest.mark.parametrize("title", ["", "x" * 81])
def test_a_conversation_title_is_one_to_eighty_characters(title: str) -> None:
    with pytest.raises(ValueError, match="title must be 1 to 80 characters"):
        Conversation("c-1", title, T0, T0)


def test_a_conversation_is_never_updated_before_it_was_created() -> None:
    with pytest.raises(ValueError, match="updated_at must not precede created_at"):
        Conversation("c-1", "Title", T0, T0 - timedelta(seconds=1))
