"""The chat service: one question, from the stored turn to the stored answer.

The question and a pending answer row are committed first. Then a follow-up is
rewritten for retrieval, the excerpts are retrieved and the relevance gate decides
(layer 1). Below the threshold the fixed refusal is given and the model is never
called; otherwise the model answers from the excerpts, its text goes through the
marker detector (layer 2) and the finished answer through citation validation. Every
outcome is stored with its trace and logged once, as ids, counts, codes and timings,
never as text.

`answer` is an async generator of events: the API reads the first one before it
starts the response, so failures before the first token stay HTTP errors.
"""

import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from app.core.settings import Settings
from app.domain.chat import (
    REFUSAL_TEXT,
    AnswerFlags,
    AnswerRecord,
    Message,
    Outcome,
    RefusalReason,
    Source,
    UnknownDocumentError,
    conversation_title,
)
from app.domain.chat_store import ChatStore, ConversationNotFoundError, Turn
from app.domain.citations import snippet, validate_citations
from app.domain.llm import GenerationRequest, LLMProvider, PromptMessage, Purpose, Usage
from app.domain.marker import MarkerDetector
from app.domain.prompt import ANSWER_SYSTEM_PROMPT, HistoryBudget, answer_messages
from app.domain.retrieval import IndexBuildRef, IndexIdentity, RetrievalConfig, RetrievalResult
from app.services.retrieval import RetrievalService
from app.services.rewrite import QueryRewriter, RewriteResult

logger = logging.getLogger(__name__)


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


AnswerEvent = AnswerToken | AnswerDone


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
    refused: bool = False
    late_marker: bool = False
    model: str | None = None
    usage: Usage = field(default_factory=lambda: Usage(None, None))
    ttft: float | None = None  # clock reading of the first visible text
    generation_ms: float | None = None


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
    ) -> AsyncIterator[AnswerEvent]:
        """Raises ConversationNotFoundError or ConversationBusyError before anything
        is stored, and UnknownDocumentError after removing the turn it started.
        """
        start = self._clock()
        thread = await self._store.get_thread(client_id, conversation_id)
        if thread is None:
            raise ConversationNotFoundError
        history = thread.messages
        turn = await self._store.start_turn(
            client_id, conversation_id, question, title=conversation_title(question)
        )
        build = await self._index.current()
        rewrite = await self._rewriter.rewrite(question, history)
        retrieved = await self._retrieval.retrieve(rewrite.query, self._retrieval_config)
        trace = _Trace(client_id, turn, start, build, rewrite, retrieved)
        if retrieved.relevance.above_threshold is False:
            yield AnswerToken(REFUSAL_TEXT)
            yield await self._finish(trace, self._below_threshold(trace))
            return
        messages = await self._prompt(trace, question, history)
        generation = _Generation()
        async for token in self._generate(messages, generation):
            yield token
        if generation.refused:
            generation.ttft = self._clock()  # the fixed text is the first one shown
            yield AnswerToken(REFUSAL_TEXT)
        yield await self._finish(trace, self._answered(trace, generation))

    async def _prompt(
        self, trace: _Trace, question: str, history: Sequence[Message]
    ) -> tuple[PromptMessage, ...]:
        try:
            return answer_messages(
                question, trace.retrieved.chunks, self._titles, history, self._config.history
            )
        except ValueError:
            # Only an untitled document raises here; its message names the document,
            # so the fixed-message error replaces it (and its context) entirely.
            await self._store.discard_turn(trace.client_id, trace.turn)
            raise UnknownDocumentError from None

    async def _generate(
        self, messages: tuple[PromptMessage, ...], generation: _Generation
    ) -> AsyncIterator[AnswerToken]:
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
            async for piece in stream:
                visible = detector.feed(piece)
                if detector.refused:
                    break
                if visible:
                    yield self._shown(generation, visible)
            rest = "" if detector.refused else detector.finish()
            if rest:
                yield self._shown(generation, rest)
            generation.model, generation.usage = stream.served_model, stream.usage
        generation.refused, generation.late_marker = detector.refused, detector.late_marker
        generation.generation_ms = (self._clock() - begin) * 1000

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
        if generation.refused:
            return self._record(
                trace,
                Outcome.REFUSED,
                REFUSAL_TEXT,
                reason=RefusalReason.MODEL_REFUSED,
                generation=generation,
                flags=AnswerFlags(rewrite_fallback=trace.rewrite_fallback),
            )
        # Removing a late marker can leave whitespace at the end.
        checked = validate_citations("".join(generation.parts).strip(), self._tags(trace))
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

    async def _finish(self, trace: _Trace, answer: AnswerRecord) -> AnswerDone:
        turn = trace.turn
        stored = await self._store.finish_answer(trace.client_id, turn.assistant.id, answer)
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
        return AnswerDone(turn, answer)


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
