"""The chat service: one question, from the stored turn to the stored answer.

The question and a pending answer row are committed first. Then a follow-up is
rewritten for retrieval, the excerpts are retrieved and the relevance gate decides
(layer 1). Below the threshold the fixed refusal is given and the model is never
called; otherwise the model answers from the excerpts, its text goes through the
marker detector (layer 2) and the finished answer through citation validation. Every
outcome is stored with its trace and logged once, as ids, counts, codes and timings,
never as text.

`answer` is an async generator of events: the API reads the first one before it
starts the response, so failures before the first token stay HTTP errors. Until the
first token nothing is kept: a failure or a cancellation removes the turn as if it was
never sent. After it, what the user saw is stored: a model error as `failed` (with an
error event), a cancellation or a closed stream as `cancelled`, a provider refusal as a
withdrawn answer. The final write runs in its own shielded task, so a second
cancellation cannot leave the answer pending.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from app.core.settings import Settings
from app.core.tasks import shielded
from app.domain.chat import (
    REFUSAL_TEXT,
    AnswerFlags,
    AnswerRecord,
    ChatError,
    Message,
    Outcome,
    RefusalReason,
    Source,
    UnknownDocumentError,
    conversation_title,
)
from app.domain.chat_store import ChatStore, ConversationNotFoundError, Turn
from app.domain.citations import snippet, validate_citations
from app.domain.llm import (
    GenerationRequest,
    LLMEmptyAnswerError,
    LLMError,
    LLMProvider,
    PromptMessage,
    Purpose,
    StopReason,
    Usage,
)
from app.domain.marker import MarkerDetector
from app.domain.prompt import ANSWER_SYSTEM_PROMPT, HistoryBudget, answer_messages
from app.domain.retrieval import (
    IndexBuildRef,
    IndexIdentity,
    ModelsNotReadyError,
    RetrievalConfig,
    RetrievalResult,
    RetrievalUnavailableError,
)
from app.services.retrieval import RetrievalService
from app.services.rewrite import QueryRewriter, RewriteResult

logger = logging.getLogger(__name__)

# The error event after a provider safety refusal: the text already shown is withdrawn.
ANSWER_WITHDRAWN = ("answer_withdrawn", "The answer was withdrawn.")
INTERNAL_ERROR = "internal_error"
# Pre-first-token cancellation comes from the client going away (commit 10's watcher).
CLIENT_DISCONNECTED = "client_disconnected"


@dataclass(frozen=True)
class ChatConfig:
    model: str
    max_tokens: int
    effort: str | None
    history: HistoryBudget
    snippet_chars: int


@dataclass(frozen=True)
class AnswerToken:
    """Text to show now; the final content (in AnswerDone) replaces what was streamed,
    since citations are validated on the finished answer.
    """

    text: str


@dataclass(frozen=True)
class AnswerDone:
    turn: Turn
    answer: AnswerRecord


@dataclass(frozen=True)
class AnswerFailed:
    """The answer stopped after text was shown; code and message are fixed and safe."""

    turn: Turn
    code: str
    message: str


AnswerEvent = AnswerToken | AnswerDone | AnswerFailed


@dataclass(frozen=True)
class _Trace:
    """What one turn knows before generation, for its stored record."""

    client_id: str
    turn: Turn
    start: float
    build: IndexBuildRef
    rewrite: RewriteResult
    retrieved: RetrievalResult

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Rank order, which is also tag order: tag S<i> is chunk i."""
        return tuple(chunk.hit.chunk_id for chunk in self.retrieved.chunks)

    @property
    def rewrite_fallback(self) -> str | None:
        fallback = self.rewrite.fallback
        return None if fallback is None else fallback.value


@dataclass
class _Generation:
    """What the model produced so far: shown text (or a refusal), usage and timing."""

    parts: list[str] = field(default_factory=list)
    refused: bool = False  # the leading marker (layer 2)
    provider_refused: bool = False  # the provider's own safety stop
    late_marker: bool = False
    stop_reason: StopReason | None = None
    model: str | None = None
    usage: Usage = field(default_factory=lambda: Usage(None, None))
    ttft: float | None = None  # clock reading of the first visible text
    generation_ms: float | None = None

    @property
    def shown_text(self) -> str:
        return "".join(self.parts).strip()


@dataclass
class _State:
    """How far one turn got, for the outcome to store when it stops early."""

    client_id: str
    turn: Turn
    start: float
    trace: _Trace | None = None
    generation: _Generation = field(default_factory=_Generation)
    shown: bool = False  # an event reached the user
    finishing: bool = False  # the final write has started


class ChatService:
    def __init__(  # noqa: PLR0913 - one port or setting per argument, all keyword-only
        self,
        *,
        store: ChatStore,
        retrieval: RetrievalService,
        retrieval_config: RetrievalConfig,
        rewriter: QueryRewriter,
        provider: LLMProvider,
        index: IndexIdentity,
        titles: Mapping[str, str],
        config: ChatConfig,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._store, self._retrieval, self._retrieval_config = store, retrieval, retrieval_config
        self._rewriter, self._provider, self._index = rewriter, provider, index
        self._titles, self._config, self._clock = titles, config, clock

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def index_identity(self) -> IndexIdentity:
        return self._index

    async def answer(
        self, client_id: str, conversation_id: str, question: str
    ) -> AsyncGenerator[AnswerEvent, None]:
        """Raises ConversationNotFoundError or ConversationBusyError before anything
        is stored; any other failure before the first event removes the turn first.
        """
        start = self._clock()
        thread = await self._store.get_thread(client_id, conversation_id)
        if thread is None:
            raise ConversationNotFoundError
        turn = await self._store.start_turn(
            client_id, conversation_id, question, title=conversation_title(question)
        )
        state = _State(client_id, turn, start)
        try:
            async with contextlib.aclosing(self._run(state, question, thread.messages)) as run:
                async for event in run:
                    state.shown = True
                    yield event
        except (asyncio.CancelledError, GeneratorExit):
            await self._stop(state, CLIENT_DISCONNECTED, Outcome.CANCELLED, None)
            raise
        except Exception as error:
            code = _error_code(error)
            await self._stop(state, code, Outcome.FAILED, code)
            if not state.shown or not isinstance(error, LLMError):
                raise
            yield AnswerFailed(turn, error.code, error.message)

    async def _run(
        self, state: _State, question: str, history: Sequence[Message]
    ) -> AsyncGenerator[AnswerEvent, None]:
        build = await self._index.current()
        rewrite = await self._rewriter.rewrite(question, history)
        retrieved = await self._retrieval.retrieve(rewrite.query, self._retrieval_config)
        trace = state.trace = _Trace(
            state.client_id, state.turn, state.start, build, rewrite, retrieved
        )
        if retrieved.relevance.above_threshold is False:
            yield AnswerToken(REFUSAL_TEXT)
            yield await self._finish(state, self._below_threshold(trace))
            return
        messages = self._prompt(trace, question, history)
        generation = state.generation
        # Closed explicitly, so leaving early closes the provider stream at once.
        async with contextlib.aclosing(self._generate(messages, generation)) as tokens:
            async for token in tokens:
                yield token
        refused = generation.refused or generation.provider_refused
        if not refused and not generation.parts:
            raise LLMEmptyAnswerError
        if generation.provider_refused and state.shown:
            yield await self._withdraw(state, trace)
            return
        if refused:
            generation.ttft = self._clock()  # the fixed text is the first one shown
            yield AnswerToken(REFUSAL_TEXT)
        yield await self._finish(state, self._answered(trace, generation))

    async def _stop(
        self, state: _State, discard_code: str, outcome: Outcome, error_code: str | None
    ) -> None:
        """Before the first event, remove the turn; after it, store what was shown
        (unless the final write already started).
        """
        trace = state.trace
        if trace is None or not state.shown:
            await self._discard(state, discard_code)
        elif not state.finishing:
            record = self._record(
                trace,
                outcome,
                state.generation.shown_text,
                generation=state.generation,
                error_code=error_code,
                flags=AnswerFlags(rewrite_fallback=trace.rewrite_fallback),
            )
            await self._write(state, record)

    async def _discard(self, state: _State, code: str) -> None:
        turn = state.turn
        logger.warning(
            "turn discarded",
            extra={
                "conversation_id": turn.assistant.conversation_id,
                "message_id": turn.assistant.id,
                "error_code": code,
            },
        )
        await shielded(self._store.discard_turn(state.client_id, turn))

    async def _withdraw(self, state: _State, trace: _Trace) -> AnswerFailed:
        """A provider safety stop after text was shown: the text is not kept."""
        code, message = ANSWER_WITHDRAWN
        record = self._record(
            trace,
            Outcome.REFUSED,
            REFUSAL_TEXT,
            reason=RefusalReason.PROVIDER_REFUSED,
            generation=state.generation,
            error_code=code,
            flags=AnswerFlags(rewrite_fallback=trace.rewrite_fallback),
        )
        state.finishing = True
        await self._write(state, record)
        return AnswerFailed(state.turn, code, message)

    def _prompt(
        self, trace: _Trace, question: str, history: Sequence[Message]
    ) -> tuple[PromptMessage, ...]:
        try:
            return answer_messages(
                question, trace.retrieved.chunks, self._titles, history, self._config.history
            )
        except ValueError:
            # Only an untitled document raises here; its message names the document,
            # so the fixed-message error replaces it (and its context) entirely.
            raise UnknownDocumentError from None

    async def _generate(
        self, messages: tuple[PromptMessage, ...], generation: _Generation
    ) -> AsyncGenerator[AnswerToken, None]:
        """Stream the answer through the marker detector, filling `generation`.

        A leading marker stops reading at once: leaving the context closes the stream.
        """
        request = GenerationRequest(
            purpose=Purpose.ANSWER,
            model=self._config.model,
            system=ANSWER_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=self._config.max_tokens,
            effort=self._config.effort,
        )
        detector = MarkerDetector()
        begin = self._clock()
        async with self._provider.stream(request) as stream:
            try:
                async for piece in stream:
                    visible = detector.feed(piece)
                    if detector.refused:
                        break
                    if visible:
                        yield self._shown(generation, visible)
                generation.stop_reason = stream.stop_reason
                generation.provider_refused = stream.stop_reason is StopReason.REFUSAL
                held = detector.refused or generation.provider_refused
                rest = "" if held else detector.finish()
                if rest:
                    yield self._shown(generation, rest)
            finally:
                # Usage and model as last reported, even when the stream failed.
                generation.model, generation.usage = stream.served_model, stream.usage
                generation.generation_ms = (self._clock() - begin) * 1000
        generation.refused, generation.late_marker = detector.refused, detector.late_marker

    def _shown(self, generation: _Generation, text: str) -> AnswerToken:
        if generation.ttft is None:
            generation.ttft = self._clock()
        generation.parts.append(text)
        return AnswerToken(text)

    def _below_threshold(self, trace: _Trace) -> AnswerRecord:
        return self._record(
            trace,
            Outcome.REFUSED,
            REFUSAL_TEXT,
            reason=RefusalReason.BELOW_THRESHOLD,
            flags=AnswerFlags(rewrite_fallback=trace.rewrite_fallback),
        )

    def _answered(self, trace: _Trace, generation: _Generation) -> AnswerRecord:
        if generation.refused or generation.provider_refused:
            reason = (
                RefusalReason.MODEL_REFUSED
                if generation.refused
                else RefusalReason.PROVIDER_REFUSED
            )
            return self._record(
                trace,
                Outcome.REFUSED,
                REFUSAL_TEXT,
                reason=reason,
                generation=generation,
                flags=AnswerFlags(rewrite_fallback=trace.rewrite_fallback),
            )
        # Removing a late marker can leave whitespace at the end.
        checked = validate_citations(generation.shown_text, self._tags(trace))
        return self._record(
            trace,
            Outcome.COMPLETED,
            checked.text,
            citations=checked.tags,
            sources=tuple(self._source(trace, tag) for tag in checked.tags),
            generation=generation,
            flags=AnswerFlags(
                late_refusal_marker=generation.late_marker,
                uncited=not checked.tags,
                invalid_citations=checked.invalid,
                rewrite_fallback=trace.rewrite_fallback,
                truncated=generation.stop_reason is not StopReason.END_TURN,
            ),
        )

    def _record(  # noqa: PLR0913 - the parts of a record that vary by outcome
        self,
        trace: _Trace,
        outcome: Outcome,
        content: str,
        *,
        flags: AnswerFlags,
        reason: RefusalReason | None = None,
        citations: tuple[str, ...] = (),
        sources: tuple[Source, ...] = (),
        generation: _Generation | None = None,
        error_code: str | None = None,
    ) -> AnswerRecord:
        """The stored record; model fields stay None when the model was not called."""
        made = generation or _Generation()
        return AnswerRecord(
            outcome=outcome,
            content=content,
            refusal_reason=reason,
            citations=citations,
            sources=sources,
            chunk_ids=trace.chunk_ids,
            model=made.model,
            input_tokens=made.usage.input_tokens,
            output_tokens=made.usage.output_tokens,
            timings=trace.retrieved.timings,
            rewrite_ms=trace.rewrite.rewrite_ms,
            ttft_ms=None if made.ttft is None else (made.ttft - trace.start) * 1000,
            generation_ms=made.generation_ms,
            total_ms=(self._clock() - trace.start) * 1000,
            relevance=trace.retrieved.relevance,
            index_build_id=trace.build.build_id,
            rewritten_query=trace.rewrite.rewritten_query,
            error_code=error_code,
            flags=flags,
        )

    @staticmethod
    def _tags(trace: _Trace) -> set[str]:
        return {f"S{n}" for n in range(1, len(trace.retrieved.chunks) + 1)}

    def _source(self, trace: _Trace, tag: str) -> Source:
        hit = trace.retrieved.chunks[int(tag[1:]) - 1].hit
        return Source(
            tag=tag,
            doc_id=hit.doc_id,
            document_title=self._titles[hit.doc_id],
            section_path=hit.section_path,
            pages=hit.pages,
            has_warning=hit.has_warning,
            snippet=snippet(hit.text, self._config.snippet_chars),
        )

    async def _finish(self, state: _State, answer: AnswerRecord) -> AnswerDone:
        state.finishing = True
        await self._write(state, answer)
        return AnswerDone(state.turn, answer)

    async def _write(self, state: _State, answer: AnswerRecord) -> None:
        turn = state.turn
        stored = await shielded(
            self._store.finish_answer(state.client_id, turn.assistant.id, answer)
        )
        if stored is None:
            # The conversation was deleted while answering; the user still sees it.
            logger.warning(
                "answer not stored",
                extra={
                    "conversation_id": turn.assistant.conversation_id,
                    "message_id": turn.assistant.id,
                },
            )
        _log_answer(turn, answer)


def _error_code(error: Exception) -> str:
    if isinstance(error, LLMError | ChatError):
        return error.code
    if isinstance(error, RetrievalUnavailableError):
        return "retrieval_unavailable"
    if isinstance(error, ModelsNotReadyError):
        return "models_not_ready"
    return INTERNAL_ERROR


def _log_answer(turn: Turn, answer: AnswerRecord) -> None:
    flags = answer.flags
    reason = answer.refusal_reason
    logger.info(
        "chat answer",
        extra={
            "conversation_id": turn.assistant.conversation_id,
            "message_id": turn.assistant.id,
            "outcome": answer.outcome.value,
            "refusal_reason": None if reason is None else reason.value,
            "error_code": answer.error_code,
            "chunk_ids": list(answer.chunk_ids),
            "citations": len(answer.citations),
            "invalid_citations": flags.invalid_citations,
            "uncited": flags.uncited,
            "late_refusal_marker": flags.late_refusal_marker,
            "rewrite_fallback": flags.rewrite_fallback,
            "truncated": flags.truncated,
            "index_build_id": answer.index_build_id,
            "model": answer.model,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "ttft_ms": answer.ttft_ms,
            "generation_ms": answer.generation_ms,
            "total_ms": answer.total_ms,
        },
    )


def chat_config_from_settings(settings: Settings) -> ChatConfig:
    return ChatConfig(
        model=settings.llm_answer_model,
        max_tokens=settings.llm_answer_max_tokens,
        effort=settings.llm_answer_effort,
        history=HistoryBudget(
            max_turns=settings.chat_history_max_turns,
            max_tokens=settings.chat_history_max_tokens,
        ),
        snippet_chars=settings.chat_snippet_chars,
    )
