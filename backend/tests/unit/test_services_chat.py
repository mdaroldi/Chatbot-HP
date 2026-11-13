"""ChatService: one question from the stored turn to the stored answer, per outcome.

Real domain code and the real RetrievalService and QueryRewriter run over fakes of the
ports (store, search, models, language model, index identity), so each test checks
what the user is shown and what is stored, not how the service is put together.
"""

import itertools
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.core.settings import Settings
from app.domain.chat import (
    REFUSAL_MARKER,
    REFUSAL_TEXT,
    AnswerFlags,
    Outcome,
    RefusalReason,
    Role,
    UnknownDocumentError,
    conversation_title,
)
from app.domain.chat_store import ConversationNotFoundError
from app.domain.citations import snippet
from app.domain.llm import LLMUnavailableError, Purpose
from app.domain.prompt import ANSWER_SYSTEM_PROMPT, HistoryBudget
from app.domain.retrieval import (
    IndexBuildRef,
    RetrievalConfig,
    SearchHit,
    SearchMode,
)
from app.services.chat import (
    AnswerDone,
    AnswerEvent,
    AnswerToken,
    ChatConfig,
    ChatService,
    chat_config_from_settings,
)
from app.services.retrieval import RetrievalService
from app.services.rewrite import QueryRewriter, RewriteConfig, rewrite_config_from_settings
from tests.unit.chat_fakes import InMemoryChatStore
from tests.unit.conftest import LogReader, SettingsFactory
from tests.unit.llm_fakes import FakeProvider
from tests.unit.retrieval_fakes import FakeQueryEmbedder, FakeReranker, FakeSearchStore, make_hit

CLIENT = "client-a"
CANARY = "CANARY-5e1d-chat"
OMEN, ENVY = "omen-17-service-guide", "envy-6000-user-guide"
TITLES = {OMEN: "Synthetic OMEN guide", ENVY: "Synthetic ENVY guide"}
BUILD = IndexBuildRef("hpdocs_chunks_20261113t173108z", "20261113t173108z")


def retrieval_config(**overrides: Any) -> RetrievalConfig:  # Any: per-test field overrides
    fields: dict[str, Any] = {
        "mode": SearchMode.HYBRID,
        "rerank": True,
        "top_k": 3,
        "rerank_candidates": 4,
        "prefetch_dense": 4,
        "prefetch_sparse": 4,
        "rrf_k": 60,
        "product_filter": False,
        "relevance_threshold": 0.0,
        "dense_query_prefix": "",
    }
    fields.update(overrides)
    return RetrievalConfig(**fields)


CONFIG = ChatConfig(
    model="configured-answer-model",
    max_tokens=900,
    effort="low",
    history=HistoryBudget(max_turns=4, max_tokens=4_000),
    snippet_chars=40,
)
REWRITE = RewriteConfig(
    model="configured-rewrite-model",
    max_tokens=60,
    effort="low",
    timeout_seconds=5.0,
    budget=HistoryBudget(max_turns=3, max_tokens=1_000),
)

# The fake reranker scores a chunk by its text length, so the rank order is c2, c3, c1
# (c4 is shortest and falls outside top_k = 3): tags S1, S2, S3 are c2, c3, c1.
HITS = [
    make_hit("c1", text="Breadcrumb > c1\n\nTen chars."),
    make_hit("c2", text="Breadcrumb > c2\n\nThe longest synthetic body of all four chunks here."),
    make_hit("c3", text="Breadcrumb > c3\n\nA middle-sized synthetic body.", doc_id=ENVY),
    make_hit("c4", text="c4"),
]
RANKED = ("c2", "c3", "c1")
ANSWER_PIECES = ("Remove the cover [S1]. ", "Then the fan [S2][S9]. ")


@dataclass
class FakeIndexIdentity:
    ref: IndexBuildRef = BUILD
    calls: int = 0

    async def current(self) -> IndexBuildRef:
        self.calls += 1
        return self.ref


def ticking() -> Callable[[], float]:
    """A clock that moves 10 ms per reading, so every duration is positive."""
    readings = itertools.count()
    return lambda: next(readings) / 100


@dataclass
class Rig:
    answer: FakeProvider = field(default_factory=lambda: FakeProvider(ANSWER_PIECES))
    rewrite: FakeProvider = field(default_factory=lambda: FakeProvider(("cover fan removal",)))
    hits: list[SearchHit] = field(default_factory=lambda: list(HITS))
    store: InMemoryChatStore = field(default_factory=InMemoryChatStore)
    index: FakeIndexIdentity = field(default_factory=FakeIndexIdentity)
    reranker: FakeReranker = field(default_factory=FakeReranker)
    search: FakeSearchStore = field(init=False)

    def __post_init__(self) -> None:
        self.search = FakeSearchStore(self.hits)

    def service(self, config: RetrievalConfig | None = None) -> ChatService:
        retrieval = RetrievalService(FakeQueryEmbedder(), self.search, self.reranker)
        return ChatService(
            store=self.store,
            retrieval=retrieval,
            retrieval_config=config or retrieval_config(),
            rewriter=QueryRewriter(self.rewrite, REWRITE),
            provider=self.answer,
            index=self.index,
            titles=TITLES,
            config=CONFIG,
            clock=ticking(),
        )

    async def conversation(self) -> str:
        return (await self.store.create_conversation(CLIENT)).id


async def collect(events: AsyncIterator[AnswerEvent]) -> list[AnswerEvent]:
    return [event async for event in events]


def tokens(events: list[AnswerEvent]) -> list[str]:
    return [event.text for event in events if isinstance(event, AnswerToken)]


def done(events: list[AnswerEvent]) -> AnswerDone:
    assert isinstance(events[-1], AnswerDone)
    assert sum(isinstance(event, AnswerDone) for event in events) == 1
    return events[-1]


async def ask(
    rig: Rig, question: str = "How do I open it?", config: RetrievalConfig | None = None
) -> tuple[list[AnswerEvent], AnswerDone]:
    conversation_id = await rig.conversation()
    events = await collect(rig.service(config).answer(CLIENT, conversation_id, question))
    return events, done(events)


# --- Completed answers ----------------------------------------------------------------


async def test_an_answer_streams_its_text_and_stores_the_validated_answer() -> None:
    rig = Rig()

    events, finished = await ask(rig)

    assert tokens(events) == list(ANSWER_PIECES)
    answer = finished.answer
    assert answer.outcome is Outcome.COMPLETED
    assert answer.content == "Remove the cover [S1]. Then the fan [S2]."
    assert answer.citations == ("S1", "S2")
    assert answer.flags == AnswerFlags(invalid_citations=1)
    assert answer.refusal_reason is None
    assert rig.store.answers[finished.turn.assistant.id] == answer


async def test_the_done_event_names_the_stored_turn() -> None:
    rig = Rig()
    conversation_id = await rig.conversation()

    events = await collect(rig.service().answer(CLIENT, conversation_id, "How do I open it?"))

    turn = done(events).turn
    thread = await rig.store.get_thread(CLIENT, conversation_id)
    assert thread is not None
    assert [m.id for m in thread.messages] == [turn.user.id, turn.assistant.id]
    assert thread.messages[0].content == "How do I open it?"
    assert thread.messages[1].outcome is Outcome.COMPLETED


async def test_stored_chunk_ids_follow_the_tag_order_and_each_source_is_its_chunk() -> None:
    rig = Rig()

    _, finished = await ask(rig)

    answer = finished.answer
    assert answer.chunk_ids == RANKED
    texts = {hit.chunk_id: hit for hit in HITS}
    for source in answer.sources:
        hit = texts[answer.chunk_ids[int(source.tag[1:]) - 1]]
        assert (source.doc_id, source.section_path, source.pages) == (
            hit.doc_id,
            hit.section_path,
            hit.pages,
        )
        assert source.document_title == TITLES[hit.doc_id]
        assert source.has_warning is hit.has_warning
        assert source.snippet == snippet(hit.text, CONFIG.snippet_chars)
    assert [source.tag for source in answer.sources] == ["S1", "S2"]
    assert [source.doc_id for source in answer.sources] == [OMEN, ENVY]


async def test_the_prompt_tags_the_excerpts_in_rank_order() -> None:
    rig = Rig()

    await ask(rig)

    last = rig.answer.requests[0].messages[-1].text
    positions = [last.index(f"Breadcrumb &gt; {chunk_id}") for chunk_id in RANKED]
    assert positions == sorted(positions)
    assert last.index('tag="S1"') < positions[0] < last.index('tag="S2"') < positions[1]


async def test_the_answer_request_uses_the_answer_settings_and_system_prompt() -> None:
    rig = Rig()

    await ask(rig, "Where is <the> fan?")

    (request,) = rig.answer.requests
    assert request.purpose is Purpose.ANSWER
    assert (request.model, request.max_tokens, request.effort) == (
        "configured-answer-model",
        900,
        "low",
    )
    assert request.system == ANSWER_SYSTEM_PROMPT
    assert request.messages[-1].role is Role.USER
    assert request.messages[-1].text.endswith("<question>Where is &lt;the&gt; fan?</question>")


async def test_a_late_marker_is_removed_and_the_answer_trimmed_before_it_is_stored() -> None:
    rig = Rig(answer=FakeProvider(("Do this [S1]. ", REFUSAL_MARKER, " \n")))

    events, finished = await ask(rig)

    assert REFUSAL_MARKER not in "".join(tokens(events))
    assert finished.answer.content == "Do this [S1]."
    assert finished.answer.flags.late_refusal_marker is True
    assert finished.answer.outcome is Outcome.COMPLETED


async def test_text_held_back_as_a_possible_marker_is_shown_when_the_stream_ends() -> None:
    rig = Rig(answer=FakeProvider(("Use the left slot [S1] [[NOT",)))

    events, finished = await ask(rig)

    assert tokens(events) == ["Use the left slot [S1] ", "[[NOT"]
    assert finished.answer.content == "Use the left slot [S1] [[NOT"
    assert finished.answer.flags.late_refusal_marker is False


async def test_an_answer_held_back_whole_still_has_a_first_token_time() -> None:
    rig = Rig(answer=FakeProvider(("[[NOT",)))

    events, finished = await ask(rig)

    assert tokens(events) == ["[[NOT"]
    assert finished.answer.outcome is Outcome.COMPLETED
    assert finished.answer.ttft_ms is not None


async def test_an_answer_without_valid_citations_is_kept_and_flagged_uncited() -> None:
    rig = Rig(answer=FakeProvider(("An answer that cites ", "nothing [S7].")))

    _, finished = await ask(rig)

    answer = finished.answer
    assert answer.content == "An answer that cites nothing."
    assert (answer.citations, answer.sources) == ((), ())
    assert answer.flags == AnswerFlags(uncited=True, invalid_citations=1)


async def test_the_answer_trace_is_stored_with_the_index_build() -> None:
    rig = Rig()

    _, finished = await ask(rig)

    answer = finished.answer
    assert rig.index.calls == 1
    assert answer.index_build_id == BUILD.build_id
    assert answer.model == "fake-served"
    assert (answer.input_tokens, answer.output_tokens) == (1, 1)
    assert answer.timings is not None
    assert answer.relevance is not None
    assert answer.relevance.above_threshold is True
    assert (answer.rewritten_query, answer.rewrite_ms) == (None, None)
    assert answer.ttft_ms is not None
    assert answer.generation_ms is not None
    assert answer.total_ms is not None
    assert 0 < answer.ttft_ms < answer.total_ms
    assert 0 < answer.generation_ms < answer.total_ms


async def test_the_first_question_titles_the_conversation() -> None:
    rig = Rig()
    question = "How do I remove the bottom cover and the fan of this synthetic laptop model?"
    conversation_id = await rig.conversation()

    await collect(rig.service().answer(CLIENT, conversation_id, question))

    thread = await rig.store.get_thread(CLIENT, conversation_id)
    assert thread is not None
    assert thread.conversation.title == conversation_title(question)


async def test_an_unknown_conversation_is_not_found_and_nothing_runs() -> None:
    rig = Rig()

    with pytest.raises(ConversationNotFoundError):
        await collect(rig.service().answer(CLIENT, "no-such-conversation", "How?"))

    assert (rig.search.requests, rig.answer.requests, rig.index.calls) == ([], [], 0)


async def test_another_clients_conversation_is_not_found() -> None:
    rig = Rig()
    conversation_id = await rig.conversation()

    with pytest.raises(ConversationNotFoundError):
        await collect(rig.service().answer("client-b", conversation_id, "How?"))

    assert rig.answer.requests == []


async def test_an_answer_whose_conversation_was_deleted_meanwhile_still_finishes(
    json_logs: LogReader,
) -> None:
    rig = Rig()
    conversation_id = await rig.conversation()
    events: list[AnswerEvent] = []

    async for event in rig.service().answer(CLIENT, conversation_id, "How?"):
        events.append(event)
        if len(events) == 1:
            assert await rig.store.delete_conversation(CLIENT, conversation_id)

    assert done(events).answer.outcome is Outcome.COMPLETED
    assert rig.store.answers == {}
    warnings = [r for r in json_logs() if r["message"] == "answer not stored"]
    assert [r["level"] for r in warnings] == ["WARNING"]


# --- Follow-ups and rewriting ---------------------------------------------------------


async def follow_up(rig: Rig, question: str) -> AnswerDone:
    conversation_id = await rig.conversation()
    await collect(rig.service().answer(CLIENT, conversation_id, "How do I open it?"))
    rig.reranker.calls.clear()
    return done(await collect(rig.service().answer(CLIENT, conversation_id, question)))


async def test_a_follow_up_retrieves_with_the_rewrite_and_answers_the_original() -> None:
    rig = Rig()

    finished = await follow_up(rig, "And the fan?")

    assert rig.reranker.calls[0][0] == "cover fan removal"
    assert finished.answer.rewritten_query == "cover fan removal"
    assert finished.answer.rewrite_ms is not None
    assert finished.answer.flags.rewrite_fallback is None
    request = rig.answer.requests[-1]
    assert request.messages[-1].text.endswith("<question>And the fan?</question>")
    assert "cover fan removal" not in request.messages[-1].text


async def test_a_follow_up_carries_the_earlier_turn_as_history() -> None:
    rig = Rig()

    await follow_up(rig, "And the fan?")

    messages = rig.answer.requests[-1].messages
    assert [m.role for m in messages] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert messages[0].text == "<question>How do I open it?</question>"
    assert messages[1].text == "Remove the cover. Then the fan."


async def test_a_failed_rewrite_falls_back_to_the_message_and_flags_its_code(
    json_logs: LogReader,
) -> None:
    rig = Rig(rewrite=FakeProvider(("unused",), error=LLMUnavailableError()))

    finished = await follow_up(rig, "And the fan?")

    assert rig.reranker.calls[0][0] == "And the fan?"
    assert finished.answer.rewritten_query is None
    assert finished.answer.flags.rewrite_fallback == "llm_error"
    lines = [r for r in json_logs() if r["message"] == "chat answer"]
    assert lines[-1]["rewrite_fallback"] == "llm_error"


# --- Layer 1: the threshold gate ------------------------------------------------------


async def test_below_the_threshold_the_fixed_refusal_is_given_without_the_model() -> None:
    rig = Rig()

    events, finished = await ask(rig, config=retrieval_config(relevance_threshold=1_000.0))

    assert rig.answer.requests == []
    assert tokens(events) == [REFUSAL_TEXT]
    answer = finished.answer
    assert answer.outcome is Outcome.REFUSED
    assert answer.refusal_reason is RefusalReason.BELOW_THRESHOLD
    assert answer.content == REFUSAL_TEXT
    assert (answer.citations, answer.sources) == ((), ())
    assert rig.store.answers[finished.turn.assistant.id] == answer


async def test_a_below_threshold_refusal_stores_retrieval_but_no_model_data() -> None:
    rig = Rig()

    _, finished = await ask(rig, config=retrieval_config(relevance_threshold=1_000.0))

    answer = finished.answer
    assert answer.chunk_ids == RANKED
    assert answer.relevance is not None
    assert answer.relevance.above_threshold is False
    assert answer.relevance.threshold == 1_000.0
    assert answer.timings is not None
    assert answer.index_build_id == BUILD.build_id
    assert answer.total_ms is not None
    assert (answer.model, answer.input_tokens, answer.output_tokens) == (None, None, None)
    assert (answer.ttft_ms, answer.generation_ms) == (None, None)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(retrieval_config(rerank=False, relevance_threshold=1_000.0), id="no-signal"),
        pytest.param(retrieval_config(relevance_threshold=None), id="no-threshold"),
    ],
)
async def test_without_a_relevance_verdict_the_gate_does_not_apply(
    config: RetrievalConfig,
) -> None:
    rig = Rig()

    _, finished = await ask(rig, config=config)

    assert len(rig.answer.requests) == 1
    assert finished.answer.outcome is Outcome.COMPLETED
    assert finished.answer.relevance is not None
    assert finished.answer.relevance.above_threshold is None


# --- Layer 2: the model's marker ------------------------------------------------------


@pytest.mark.parametrize(
    "pieces",
    [
        pytest.param((REFUSAL_MARKER,), id="whole"),
        pytest.param((" \n[[NOT_", "IN_DOCU", "MENTS]]"), id="split-after-whitespace"),
        pytest.param((REFUSAL_MARKER, "never read"), id="followed-by-text"),
    ],
)
async def test_a_leading_marker_is_a_model_refusal_shown_as_the_fixed_text(
    pieces: tuple[str, ...],
) -> None:
    rig = Rig(answer=FakeProvider(pieces))

    events, finished = await ask(rig)

    assert tokens(events) == [REFUSAL_TEXT]
    answer = finished.answer
    assert answer.outcome is Outcome.REFUSED
    assert answer.refusal_reason is RefusalReason.MODEL_REFUSED
    assert answer.content == REFUSAL_TEXT
    assert (answer.citations, answer.sources) == ((), ())


async def test_the_stream_is_closed_before_the_refusal_is_shown() -> None:
    rig = Rig(answer=FakeProvider((REFUSAL_MARKER, "never read")))
    conversation_id = await rig.conversation()

    shown = []
    async for event in rig.service().answer(CLIENT, conversation_id, "How?"):
        if isinstance(event, AnswerToken):
            shown.append((event.text, rig.answer.closed))

    assert shown == [(REFUSAL_TEXT, 1)]


async def test_a_model_refusal_stores_the_model_and_retrieval_trace() -> None:
    rig = Rig(answer=FakeProvider((REFUSAL_MARKER,)))

    _, finished = await ask(rig)

    answer = finished.answer
    assert answer.model == "fake-served"
    assert (answer.input_tokens, answer.output_tokens) == (1, 1)
    assert answer.chunk_ids == RANKED
    assert answer.index_build_id == BUILD.build_id
    assert answer.ttft_ms is not None
    assert answer.generation_ms is not None
    assert rig.store.answers[finished.turn.assistant.id] == answer


# --- An excerpt the service cannot title ----------------------------------------------


async def test_an_excerpt_from_an_unknown_document_fails_with_a_fixed_message() -> None:
    stray = make_hit("c9", text=f"Breadcrumb\n\n{CANARY} stray text.", doc_id="stray-doc")
    rig = Rig(hits=[stray])
    conversation_id = await rig.conversation()

    with pytest.raises(UnknownDocumentError) as raised:
        await collect(rig.service().answer(CLIENT, conversation_id, "How?"))

    assert str(raised.value) == UnknownDocumentError.message
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert rig.answer.requests == []
    thread = await rig.store.get_thread(CLIENT, conversation_id)
    assert thread is not None
    assert thread.messages == ()


# --- Logs -----------------------------------------------------------------------------


async def test_one_log_line_per_answer_with_ids_counts_and_codes_only(
    json_logs: LogReader,
) -> None:
    rig = Rig(answer=FakeProvider((f"{CANARY} answer [S1].",)))

    _, finished = await ask(rig, f"{CANARY} question?")

    lines = [r for r in json_logs() if r["message"] == "chat answer"]
    assert len(lines) == 1
    line = lines[0]
    assert line["conversation_id"] == finished.turn.user.conversation_id
    assert line["message_id"] == finished.turn.assistant.id
    assert (line["outcome"], line["refusal_reason"]) == ("completed", None)
    assert line["chunk_ids"] == list(RANKED)
    assert line["citations"] == 1
    assert line["index_build_id"] == BUILD.build_id
    assert line["rewrite_fallback"] is None
    assert CANARY not in str(json_logs())


async def test_a_refusal_is_logged_with_its_reason(json_logs: LogReader) -> None:
    rig = Rig()

    await ask(rig, config=retrieval_config(relevance_threshold=1_000.0))

    (line,) = [r for r in json_logs() if r["message"] == "chat answer"]
    assert (line["outcome"], line["refusal_reason"]) == ("refused", "below_threshold")


# --- Configuration --------------------------------------------------------------------


def test_the_chat_config_comes_from_settings(make_settings: SettingsFactory) -> None:
    settings: Settings = make_settings(
        llm_answer_model="answer-model",
        llm_answer_max_tokens=1_500,
        llm_answer_effort=None,
        chat_history_max_turns=2,
        chat_history_max_tokens=500,
        chat_snippet_chars=90,
    )

    assert chat_config_from_settings(settings) == ChatConfig(
        model="answer-model",
        max_tokens=1_500,
        effort=None,
        history=HistoryBudget(max_turns=2, max_tokens=500),
        snippet_chars=90,
    )


def test_the_rewrite_config_comes_from_settings(make_settings: SettingsFactory) -> None:
    settings: Settings = make_settings(
        llm_rewrite_model="rewrite-model",
        llm_rewrite_max_tokens=80,
        llm_rewrite_effort="low",
        llm_rewrite_timeout_seconds=4.0,
        chat_rewrite_history_turns=0,
        chat_rewrite_history_tokens=700,
    )

    assert rewrite_config_from_settings(settings) == RewriteConfig(
        model="rewrite-model",
        max_tokens=80,
        effort="low",
        timeout_seconds=4.0,
        budget=HistoryBudget(max_turns=0, max_tokens=700),
    )


def test_the_service_exposes_its_provider_and_index_identity() -> None:
    rig = Rig()
    service = rig.service()

    assert service.provider is rig.answer
    assert service.index_identity is rig.index
