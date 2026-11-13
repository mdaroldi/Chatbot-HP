"""make smoke-llm: a few answers and one follow-up rewrite through the configured
models, reported as numbers and codes only (time to first token, stop reasons, served
models, error codes, the rewrite's outcome).
"""

import io
import runpy
from collections import Counter
from typing import Any

import pytest
from pydantic import SecretStr

from app import smoke_llm
from app.domain.chat import Role
from app.domain.llm import LLMRequestError, LLMTimeoutError, Purpose, StopReason
from app.domain.prompt import ANSWER_SYSTEM_PROMPT
from app.infra.llm_anthropic import AnthropicProvider
from app.services.rewrite import RewriteFallback, RewriteResult
from tests.unit.conftest import SettingsFactory
from tests.unit.llm_fakes import FakeProvider

CANARY = "CANARY-5e1d-smoke"


def clock_from(readings: list[float]) -> Any:  # Any: a zero-argument clock
    values = iter(readings)
    return lambda: next(values)


async def test_each_run_records_time_to_first_token_stop_reason_and_model() -> None:
    fake = FakeProvider(("First ", CANARY), stop=StopReason.MAX_TOKENS)
    # Per run: start, first piece.
    clock = clock_from([0.0, 0.5, 10.0, 11.5, 20.0, 21.0])

    report = await smoke_llm.measure(fake, smoke_llm.smoke_request("model-x", 300, "low"), 3, clock)

    assert report.ttft_ms == [500.0, 1500.0, 1000.0]
    assert report.p50_ttft_ms == 1000.0
    assert report.stop_reasons == Counter({"max_tokens": 3})
    assert report.served_models == Counter({"fake-served": 3})
    assert report.errors == Counter()


async def test_failures_are_counted_by_code() -> None:
    fake = FakeProvider(("x",), error=LLMTimeoutError())

    report = await smoke_llm.measure(
        fake, smoke_llm.smoke_request("m", 300, None), 2, clock_from([0.0] * 4)
    )

    assert report.errors == Counter({"llm_timeout": 2})
    assert report.ttft_ms == []
    assert report.p50_ttft_ms is None


async def test_a_stream_without_a_stop_reason_is_counted_as_none() -> None:
    report = await smoke_llm.measure(
        FakeProvider(("x",), stop=None),
        smoke_llm.smoke_request("m", 300, None),
        1,
        clock_from([0.0, 1.0]),
    )

    assert report.stop_reasons == Counter({"none": 1})


def test_the_smoke_request_is_a_synthetic_answer_prompt() -> None:
    request = smoke_llm.smoke_request("model-x", 300, "low")

    assert (request.purpose, request.model, request.max_tokens, request.effort) == (
        Purpose.ANSWER,
        "model-x",
        300,
        "low",
    )
    assert request.system == ANSWER_SYSTEM_PROMPT
    (message,) = request.messages
    assert message.role is Role.USER
    assert 'tag="S1"' in message.text
    assert "<question>" in message.text


def test_the_report_has_numbers_and_codes_only() -> None:
    report = smoke_llm.SmokeReport(
        runs=3,
        ttft_ms=[500.0, 1500.0],
        stop_reasons=Counter({"end_turn": 2}),
        served_models=Counter({"claude-x": 2}),
        errors=Counter({"llm_timeout": 1}),
    )

    assert report.render() == (
        "runs: 3\n"
        "ttft_p50_ms: 1000\n"
        "ttft_ms: 500, 1500\n"
        "stop_reasons: end_turn=2\n"
        "served_models: claude-x=2\n"
        "errors: llm_timeout=1\n"
    )


def test_an_empty_report_says_none() -> None:
    report = smoke_llm.SmokeReport(1, [], Counter(), Counter(), Counter({"llm_error": 1}))

    assert "ttft_p50_ms: none\n" in report.render()
    assert "stop_reasons: none\n" in report.render()


async def test_main_runs_the_configured_answer_model_and_prints_the_report(
    make_settings: SettingsFactory,
) -> None:
    fake = FakeProvider(("Answer ", CANARY))
    settings = make_settings(llm_answer_model="answer-model", llm_answer_max_tokens=512)
    out = io.StringIO()

    code = await smoke_llm.run(["--runs", "2"], settings, lambda _: fake, out)

    assert code == 0
    answers = [r.model for r in fake.requests if r.purpose is Purpose.ANSWER]
    assert answers == ["answer-model", "answer-model"]
    assert "runs: 2\n" in out.getvalue()
    assert CANARY not in out.getvalue()


async def test_main_fails_when_any_run_failed(make_settings: SettingsFactory) -> None:
    out = io.StringIO()

    code = await smoke_llm.run(
        ["--runs", "1"],
        make_settings(),
        lambda _: FakeProvider(("x",), error=LLMTimeoutError()),
        out,
    )

    assert code == 1
    assert "errors: llm_timeout=1" in out.getvalue()


@pytest.mark.parametrize("runs", ["0", "51", "many"])
async def test_main_rejects_a_bad_run_count(make_settings: SettingsFactory, runs: str) -> None:
    with pytest.raises(SystemExit):
        await smoke_llm.run(
            ["--runs", runs], make_settings(), lambda _: FakeProvider(), io.StringIO()
        )


def test_the_default_provider_is_the_anthropic_adapter(make_settings: SettingsFactory) -> None:
    settings = make_settings(
        llm_provider="anthropic", anthropic_api_key=SecretStr("placeholder-not-a-key")
    )

    assert isinstance(smoke_llm.anthropic_provider(settings), AnthropicProvider)


def test_main_refuses_to_run_without_the_anthropic_provider(
    make_settings: SettingsFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    assert smoke_llm.main(["--runs", "1"], settings=make_settings()) == 2
    assert "LLM_PROVIDER=anthropic" in capsys.readouterr().err


def test_main_reports_on_stdout(
    make_settings: SettingsFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(
        llm_provider="anthropic", anthropic_api_key=SecretStr("placeholder-not-a-key")
    )

    code = smoke_llm.main(
        ["--runs", "1"], settings=settings, provider_factory=lambda _: FakeProvider(("x",))
    )

    assert code == 0
    assert "runs: 1\n" in capsys.readouterr().out


def test_the_module_runs_main_and_exits_with_its_code() -> None:
    """As `make smoke-llm` runs it; the unit-test environment selects the stub."""
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("app.smoke_llm", run_name="__main__")

    assert exited.value.code == 2


async def test_main_makes_one_rewrite_call_with_the_rewrite_model(
    make_settings: SettingsFactory,
) -> None:
    fake = FakeProvider(("standalone query about ", CANARY))
    out = io.StringIO()

    code = await smoke_llm.run(
        ["--runs", "1"], make_settings(llm_rewrite_model="rewrite-model"), lambda _: fake, out
    )

    assert code == 0
    rewrites = [r for r in fake.requests if r.purpose is Purpose.REWRITE]
    assert [r.model for r in rewrites] == ["rewrite-model"]
    assert "rewrite: rewritten" in out.getvalue()
    assert CANARY not in out.getvalue()


async def test_a_rewrite_that_fell_back_fails_the_smoke_test(
    make_settings: SettingsFactory,
) -> None:
    """In the app a fallback is silent (the raw message is searched); here it means
    the rewrite model or its options are wrong, which is what the smoke test is for.
    """
    out = io.StringIO()

    code = await smoke_llm.run(
        ["--runs", "1"],
        make_settings(),
        lambda _: FakeProvider(("x",), error=LLMRequestError(), fail_at=0),
        out,
    )

    assert code == 1
    assert "rewrite: fallback=llm_error" in out.getvalue()


async def test_the_rewrite_call_has_a_finished_synthetic_turn_to_resolve(
    make_settings: SettingsFactory,
) -> None:
    fake = FakeProvider(("a query",))

    result = await smoke_llm.rewrite_once(fake, make_settings())

    (request,) = fake.requests
    assert request.purpose is Purpose.REWRITE
    assert smoke_llm.QUESTION in request.system  # the earlier turn, given as data
    assert [m.text for m in request.messages] == [smoke_llm.FOLLOW_UP]
    assert result.rewritten_query == "a query"


@pytest.mark.parametrize(
    ("result", "line"),
    [
        (RewriteResult("q", "q", 812.4, None), "rewrite: rewritten (812 ms)\n"),
        (
            RewriteResult("raw", None, 5000.0, RewriteFallback.TIMEOUT),
            "rewrite: fallback=timeout (5000 ms)\n",
        ),
        (RewriteResult("raw", None, None, None), "rewrite: skipped (no history turns)\n"),
    ],
)
def test_the_rewrite_line_has_the_outcome_and_time_only(result: RewriteResult, line: str) -> None:
    assert smoke_llm.render_rewrite(result) == line


async def test_rewriting_disabled_by_settings_is_reported_and_does_not_fail(
    make_settings: SettingsFactory,
) -> None:
    fake = FakeProvider(("x",))
    out = io.StringIO()

    code = await smoke_llm.run(
        ["--runs", "1"], make_settings(chat_rewrite_history_turns=0), lambda _: fake, out
    )

    assert code == 0
    assert [r.purpose for r in fake.requests] == [Purpose.ANSWER]
    assert "rewrite: skipped (no history turns)\n" in out.getvalue()
