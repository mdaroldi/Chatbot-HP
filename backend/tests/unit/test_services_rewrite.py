"""QueryRewriter: when it calls the model, what it sends, and every fallback."""

import json
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from app.domain.chat import Message, Outcome, Role
from app.domain.llm import (
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    PromptMessage,
    Purpose,
    StopReason,
)
from app.domain.prompt import HistoryBudget, PastTurn
from app.domain.rewrite import rewrite_system_prompt
from app.infra.llm_stub import StubProvider
from app.services.rewrite import (
    QueryRewriter,
    RewriteConfig,
    RewriteFallback,
    RewriteResult,
)
from tests.unit.conftest import LogReader
from tests.unit.llm_fakes import FakeProvider

CANARY = "CANARY-91c4-follow-up"
NOW = datetime(2026, 11, 13, tzinfo=UTC)
BUDGET = HistoryBudget(max_turns=4, max_tokens=1_000)
CONFIG = RewriteConfig(
    model="configured-rewrite-model",
    max_tokens=60,
    effort="low",
    timeout_seconds=5.0,
    budget=BUDGET,
)
HISTORY = [
    Message("m1", "c1", 1, Role.USER, "How do I remove the bottom cover?", NOW),
    Message("m2", "c1", 2, Role.ASSISTANT, "Remove the screws [S1].", NOW, Outcome.COMPLETED),
]


def clock(*readings: float) -> Callable[[], float]:
    values = iter(readings)
    return lambda: next(values)


def rewriter(
    provider: FakeProvider | StubProvider,
    config: RewriteConfig = CONFIG,
    readings: tuple[float, ...] = (10.0, 10.25),
) -> QueryRewriter:
    return QueryRewriter(provider, config, clock=clock(*readings))


# --- When the model is called ---------------------------------------------------------


async def test_a_first_question_is_used_as_it_is_without_a_model_call() -> None:
    provider = FakeProvider(("unused",))

    result = await rewriter(provider).rewrite("And the fan?", [])

    assert result == RewriteResult("And the fan?", None, None, None)
    assert provider.requests == []


async def test_history_without_a_finished_turn_is_no_reason_to_call_the_model() -> None:
    provider = FakeProvider(("unused",))
    failed = [HISTORY[0], Message("m2", "c1", 2, Role.ASSISTANT, "Part", NOW, Outcome.FAILED)]

    result = await rewriter(provider).rewrite("And the fan?", failed)

    assert result == RewriteResult("And the fan?", None, None, None)
    assert provider.requests == []


async def test_a_zero_turn_budget_turns_rewriting_off() -> None:
    provider = FakeProvider(("unused",))
    config = RewriteConfig("m", 60, None, 5.0, HistoryBudget(max_turns=0, max_tokens=1_000))

    result = await rewriter(provider, config).rewrite("And the fan?", HISTORY)

    assert result.rewritten_query is None
    assert provider.requests == []
    one_turn = RewriteConfig("m", 60, None, 5.0, HistoryBudget(max_turns=1, max_tokens=1_000))
    assert (await rewriter(provider, one_turn).rewrite("And the fan?", HISTORY)).rewrite_ms
    assert len(provider.requests) == 1


# --- What is sent and what comes back -------------------------------------------------


async def test_a_follow_up_is_rewritten_from_the_conversation() -> None:
    provider = FakeProvider(("remove the fan ", "on the laptop"))

    result = await rewriter(provider).rewrite("And the fan?", HISTORY)

    assert result == RewriteResult(
        "remove the fan on the laptop", "remove the fan on the laptop", 250.0, None
    )
    (request,) = provider.requests
    assert request.purpose is Purpose.REWRITE
    assert (request.model, request.max_tokens, request.effort) == (
        "configured-rewrite-model",
        60,
        "low",
    )
    assert request.system == rewrite_system_prompt(
        [PastTurn("How do I remove the bottom cover?", "Remove the screws.")]
    )
    assert request.messages == (PromptMessage(Role.USER, "And the fan?"),)
    assert provider.closed == 1


async def test_the_latest_message_is_sent_as_the_user_wrote_it() -> None:
    provider = FakeProvider(("query",))

    await rewriter(provider).rewrite('What about <this> & "that"?', HISTORY)

    assert provider.requests[0].messages[-1].text == 'What about <this> & "that"?'


async def test_the_stub_echoes_the_message_on_one_line() -> None:
    result = await rewriter(StubProvider(honour_directives=False)).rewrite(
        "And   the\nfan?", HISTORY
    )

    assert result.rewritten_query == "And the fan?"
    assert result.query == "And the fan?"
    assert result.fallback is None


# --- Fallbacks: the raw message is the query ------------------------------------------


@pytest.mark.parametrize("error", [LLMUnavailableError(), LLMRequestError(), LLMTimeoutError()])
@pytest.mark.parametrize("fail_at", [0, 1])
async def test_a_provider_failure_falls_back_to_the_raw_message(
    error: Exception, fail_at: int
) -> None:
    provider = FakeProvider(("remove ", "the fan"), error=error, fail_at=fail_at)

    result = await rewriter(provider).rewrite("And the fan?", HISTORY)

    assert result == RewriteResult("And the fan?", None, 250.0, RewriteFallback.LLM_ERROR)
    assert provider.closed == 1


async def test_a_stalled_rewrite_times_out_and_closes_the_stream() -> None:
    provider = FakeProvider(("never",), stall=True)
    config = RewriteConfig("m", 60, None, 0.01, BUDGET)

    result = await rewriter(provider, config).rewrite("And the fan?", HISTORY)

    assert result == RewriteResult("And the fan?", None, 250.0, RewriteFallback.TIMEOUT)
    assert provider.closed == 1


@pytest.mark.parametrize("stop", [StopReason.MAX_TOKENS, StopReason.REFUSAL, StopReason.OTHER])
async def test_an_output_that_did_not_end_its_turn_falls_back(stop: StopReason) -> None:
    provider = FakeProvider(("remove the fan",), stop=stop)

    result = await rewriter(provider).rewrite("And the fan?", HISTORY)

    assert result == RewriteResult("And the fan?", None, 250.0, RewriteFallback.INCOMPLETE)


@pytest.mark.parametrize("pieces", [(), ("  ", "\n"), ("one\n", "two"), ("a" * 301,)])
async def test_an_empty_multi_line_or_too_long_output_falls_back(
    pieces: tuple[str, ...],
) -> None:
    provider = FakeProvider(pieces)

    result = await rewriter(provider).rewrite("And the fan?", HISTORY)

    assert result == RewriteResult("And the fan?", None, 250.0, RewriteFallback.INVALID_OUTPUT)


async def test_an_unexpected_error_is_not_swallowed() -> None:
    provider = FakeProvider(("x",), error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await rewriter(provider).rewrite("And the fan?", HISTORY)


def test_a_rewrite_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        RewriteConfig("m", 60, None, 0, BUDGET)


# --- Logs -----------------------------------------------------------------------------


async def test_a_rewrite_logs_its_outcome_and_time_but_never_the_text(
    json_logs: LogReader,
) -> None:
    provider = FakeProvider((f"query {CANARY}",))
    history = [
        Message("m1", "c1", 1, Role.USER, f"first {CANARY}", NOW),
        Message("m2", "c1", 2, Role.ASSISTANT, f"answer {CANARY}", NOW, Outcome.COMPLETED),
    ]

    await rewriter(provider).rewrite(f"follow-up {CANARY}", history)

    lines = json_logs()
    assert CANARY not in json.dumps(lines)
    (done,) = [line for line in lines if line["message"] == "query rewrite"]
    assert (done["outcome"], done["rewrite_ms"], done["history_turns"]) == (
        "rewritten",
        250.0,
        1,
    )


async def test_a_fallback_logs_its_code_but_never_the_text(json_logs: LogReader) -> None:
    provider = FakeProvider((f"bad\n{CANARY}",))

    await rewriter(provider).rewrite(f"follow-up {CANARY}", HISTORY)

    lines = json_logs()
    assert CANARY not in json.dumps(lines)
    (done,) = [line for line in lines if line["message"] == "query rewrite"]
    assert done["outcome"] == "invalid_output"
