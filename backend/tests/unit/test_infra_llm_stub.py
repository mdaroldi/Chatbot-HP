"""The stub model: deterministic streamed answers, the refusal marker and failures."""

import math

import pytest

from app.domain.chat import REFUSAL_MARKER, Role
from app.domain.llm import (
    GenerationRequest,
    LLMProvider,
    LLMUnavailableError,
    PromptMessage,
    Purpose,
    StopReason,
    Usage,
)
from app.infra.llm_stub import StubProvider, StubStream

ONE_EXCERPT = '<excerpts><excerpt tag="S1">Synthetic text.</excerpt></excerpts>'
TWO_EXCERPTS = (
    '<excerpts><excerpt tag="S1">One.</excerpt><excerpt tag="S2">Two.</excerpt></excerpts>'
)


def answer_request(last: str, *, history: tuple[PromptMessage, ...] = ()) -> GenerationRequest:
    return GenerationRequest(
        purpose=Purpose.ANSWER,
        model="configured-model",
        system="System rules.",
        messages=(*history, PromptMessage(Role.USER, last)),
        max_tokens=100,
    )


def rewrite_request(last: str) -> GenerationRequest:
    return GenerationRequest(
        purpose=Purpose.REWRITE,
        model="configured-rewrite-model",
        system="Rewrite rules and the conversation as data.",
        messages=(PromptMessage(Role.USER, last),),
        max_tokens=50,
    )


async def collect(provider: LLMProvider, req: GenerationRequest) -> list[str]:
    async with provider.stream(req) as stream:
        return [piece async for piece in stream]


def stub(*, honour: bool = True) -> StubProvider:
    return StubProvider(honour_directives=honour)


def test_the_stub_names_itself() -> None:
    assert stub().name == "stub"


async def test_an_answer_cites_the_top_excerpt_and_streams_in_several_pieces() -> None:
    pieces = await collect(stub(), answer_request(ONE_EXCERPT + "<question>Q?</question>"))

    assert len(pieces) > 1
    text = "".join(pieces)
    assert "[S1]" in text
    assert "[S2]" not in text
    assert REFUSAL_MARKER not in text


async def test_an_answer_cites_the_second_excerpt_when_there_is_one() -> None:
    text = "".join(await collect(stub(), answer_request(TWO_EXCERPTS + "<question>Q?</question>")))

    assert "[S1]" in text
    assert "[S2]" in text


async def test_the_answer_is_deterministic() -> None:
    req = answer_request(TWO_EXCERPTS + "<question>Q?</question>")

    assert await collect(stub(), req) == await collect(stub(), req)


async def test_without_excerpts_the_stub_emits_only_the_refusal_marker() -> None:
    pieces = await collect(stub(), answer_request("<question>Q?</question>"))

    assert "".join(pieces) == REFUSAL_MARKER
    assert len(pieces) > 1  # split, so the detector sees a marker across deltas


async def test_the_refuse_directive_emits_the_marker() -> None:
    text = "".join(await collect(stub(), answer_request(ONE_EXCERPT + "Q? [stub:refuse]")))

    assert text == REFUSAL_MARKER


async def drain(req: GenerationRequest, received: list[str]) -> None:
    async with stub().stream(req) as stream:
        async for piece in stream:
            received.append(piece)


async def test_fail_before_raises_before_any_text() -> None:
    received: list[str] = []
    with pytest.raises(LLMUnavailableError):
        await drain(answer_request(ONE_EXCERPT + "[stub:fail-before]"), received)

    assert received == []


async def test_fail_after_raises_after_the_first_piece() -> None:
    received: list[str] = []
    with pytest.raises(LLMUnavailableError):
        await drain(answer_request(ONE_EXCERPT + "[stub:fail-after]"), received)

    assert len(received) == 1


@pytest.mark.parametrize("directive", ["[stub:refuse]", "[stub:fail-before]", "[stub:fail-after]"])
async def test_directives_are_plain_text_when_not_honoured(directive: str) -> None:
    text = "".join(await collect(stub(honour=False), answer_request(ONE_EXCERPT + directive)))

    assert "[S1]" in text
    assert REFUSAL_MARKER not in text


async def test_a_rewrite_echoes_the_last_message_on_one_line() -> None:
    text = "".join(await collect(stub(), rewrite_request("  And   what\nabout  the cover?  ")))

    assert text == "And what about the cover?"


async def test_a_rewrite_is_capped_at_three_hundred_characters() -> None:
    text = "".join(await collect(stub(), rewrite_request("word " * 200)))

    assert len(text) <= 300
    assert not text.endswith(" ")


async def test_usage_model_and_stop_reason_are_reported() -> None:
    req = answer_request(ONE_EXCERPT + "<question>Q?</question>")
    prompt_chars = len(req.system) + sum(len(m.text) for m in req.messages)
    async with stub().stream(req) as stream:
        before = (stream.stop_reason, stream.usage)
        text = "".join([piece async for piece in stream])
        after = (stream.stop_reason, stream.usage)
        served = stream.served_model

    assert served == "stub"
    assert before == (None, Usage(math.ceil(prompt_chars / 4), 0))
    assert after == (
        StopReason.END_TURN,
        Usage(math.ceil(prompt_chars / 4), math.ceil(len(text) / 4)),
    )


async def test_the_stream_is_closed_on_exit_even_when_left_early() -> None:
    async with stub().stream(answer_request(TWO_EXCERPTS)) as stream:
        assert isinstance(stream, StubStream)
        open_inside = not stream.closed
        async for _ in stream:
            break
    assert open_inside
    assert stream.closed
    assert stream.stop_reason is None


async def test_each_piece_waits_for_the_configured_delay() -> None:
    delays: list[float] = []

    async def record(seconds: float) -> None:
        delays.append(seconds)

    provider = StubProvider(honour_directives=True, token_delay_ms=20, sleep=record)
    pieces = await collect(provider, answer_request(ONE_EXCERPT))

    assert delays == [0.02] * len(pieces)


async def test_no_delay_means_no_sleep() -> None:
    async def forbidden(_seconds: float) -> None:
        raise AssertionError("sleep called")

    provider = StubProvider(honour_directives=True, sleep=forbidden)
    assert await collect(provider, answer_request(ONE_EXCERPT))


@pytest.mark.parametrize("honour", [True, False])
def test_the_stub_reports_how_it_was_configured(honour: bool) -> None:
    provider = StubProvider(honour_directives=honour, token_delay_ms=12)

    assert (provider.honours_directives, provider.token_delay_ms) == (honour, 12)
