"""`make smoke-llm`: a few answers and one rewrite from the real models, to see their
latency, stops and whether the rewrite options are accepted.

Manual and billed (it needs ANTHROPIC_API_KEY): never part of `make check` or CI.
Each run sends the real answer system prompt with one synthetic excerpt (no HP text)
and measures the time from the request to the first answer text. Then one follow-up
goes through the real query rewriter with a synthetic earlier turn. The report holds
numbers and codes only: time to first token, stop reasons, the models that served,
error codes and the rewrite's outcome, never the model's text.
"""

import argparse
import asyncio
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TextIO

from app.core.settings import LLMProviderName, Settings, get_settings
from app.domain.chat import Message, Outcome, Role
from app.domain.ingestion import ContentType, PageRange
from app.domain.llm import GenerationRequest, LLMError, LLMProvider, Purpose
from app.domain.prompt import ANSWER_SYSTEM_PROMPT, answer_turn
from app.domain.retrieval import RetrievedChunk, SearchHit
from app.infra.llm_anthropic import AnthropicProvider, anthropic_config_from_settings
from app.services.rewrite import QueryRewriter, RewriteResult, rewrite_config_from_settings

MAX_RUNS = 50
_DOC = "synthetic-manual"
_EXCERPT = SearchHit(
    chunk_id="smoke-1",
    doc_id=_DOC,
    section_path=("Maintenance", "Cleaning the fan"),
    pages=PageRange(10, 10, 8, 8),
    content_type=ContentType.PROCEDURE,
    has_warning=False,
    text="Synthetic manual > Maintenance > Cleaning the fan\n\n"
    "Turn the device off. Remove the two screws of the fan cover. Lift the cover and "
    "blow the dust out of the fan with compressed air.",
    score=1.0,
)
QUESTION = "How do I clean the fan?"
FOLLOW_UP = "And how often should I do it?"
_ANSWER = "Turn the device off, remove the fan cover and blow the dust out [S1]."

ProviderFactory = Callable[[Settings], LLMProvider]


@dataclass(frozen=True)
class SmokeReport:
    runs: int
    ttft_ms: list[float]
    stop_reasons: Counter[str]
    served_models: Counter[str]
    errors: Counter[str]

    @property
    def p50_ttft_ms(self) -> float | None:
        return statistics.median(self.ttft_ms) if self.ttft_ms else None

    def render(self) -> str:
        p50 = self.p50_ttft_ms
        lines = [
            f"runs: {self.runs}",
            f"ttft_p50_ms: {'none' if p50 is None else round(p50)}",
            f"ttft_ms: {', '.join(str(round(v)) for v in self.ttft_ms) or 'none'}",
            f"stop_reasons: {_counts(self.stop_reasons)}",
            f"served_models: {_counts(self.served_models)}",
            f"errors: {_counts(self.errors)}",
        ]
        return "".join(f"{line}\n" for line in lines)


def _counts(counter: Counter[str]) -> str:
    return " ".join(f"{key}={n}" for key, n in sorted(counter.items())) or "none"


def smoke_request(model: str, max_tokens: int, effort: str | None) -> GenerationRequest:
    chunk = RetrievedChunk(hit=_EXCERPT, rank=1, rerank_score=1.0)
    return GenerationRequest(
        purpose=Purpose.ANSWER,
        model=model,
        system=ANSWER_SYSTEM_PROMPT,
        messages=(answer_turn(QUESTION, [chunk], {_DOC: "Synthetic manual"}),),
        max_tokens=max_tokens,
        effort=effort,
    )


async def measure(
    provider: LLMProvider,
    request: GenerationRequest,
    runs: int,
    clock: Callable[[], float] = time.perf_counter,
) -> SmokeReport:
    """One request after another, so each time is the model's, not a queue's."""
    ttft: list[float] = []
    stops: Counter[str] = Counter()
    models: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for _ in range(runs):
        start = clock()
        first: float | None = None
        try:
            async with provider.stream(request) as stream:
                async for _piece in stream:
                    if first is None:
                        first = clock()
                stop = stream.stop_reason
                stops[stop.value if stop is not None else "none"] += 1
                models[stream.served_model or "none"] += 1
        except LLMError as error:
            errors[error.code] += 1
        if first is not None:
            ttft.append((first - start) * 1000)
    return SmokeReport(runs, ttft, stops, models, errors)


def _history() -> tuple[Message, ...]:
    at = datetime(2026, 1, 1, tzinfo=UTC)
    return (
        Message("smoke-q", "smoke-c", 1, Role.USER, QUESTION, at),
        Message("smoke-a", "smoke-c", 2, Role.ASSISTANT, _ANSWER, at, Outcome.COMPLETED),
    )


async def rewrite_once(provider: LLMProvider, settings: Settings) -> RewriteResult:
    """The app's own rewriter and settings, so a rejected option (a model that does not
    take `thinking: disabled`, say) shows up here as the fallback it would cause.
    """
    rewriter = QueryRewriter(provider, rewrite_config_from_settings(settings))
    return await rewriter.rewrite(FOLLOW_UP, _history())


def render_rewrite(result: RewriteResult) -> str:
    if result.rewrite_ms is None:
        return "rewrite: skipped (no history turns)\n"
    outcome = "rewritten" if result.fallback is None else f"fallback={result.fallback}"
    return f"rewrite: {outcome} ({round(result.rewrite_ms)} ms)\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.smoke_llm", description=__doc__)
    parser.add_argument("--runs", type=_run_count, default=5, help=f"1 to {MAX_RUNS}")
    return parser


def _run_count(raw: str) -> int:
    value = int(raw)
    if not 1 <= value <= MAX_RUNS:
        raise argparse.ArgumentTypeError(f"runs must be 1 to {MAX_RUNS}")
    return value


def anthropic_provider(settings: Settings) -> LLMProvider:
    return AnthropicProvider(anthropic_config_from_settings(settings))


async def run(
    argv: Sequence[str], settings: Settings, provider_factory: ProviderFactory, out: TextIO
) -> int:
    args = _parser().parse_args(argv)
    request = smoke_request(
        settings.llm_answer_model, settings.llm_answer_max_tokens, settings.llm_answer_effort
    )
    provider = provider_factory(settings)
    report = await measure(provider, request, args.runs)
    out.write(report.render())
    rewrite = await rewrite_once(provider, settings)
    out.write(render_rewrite(rewrite))
    return 1 if report.errors or rewrite.fallback is not None else 0


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    provider_factory: ProviderFactory = anthropic_provider,
) -> int:
    settings = settings or get_settings()
    if settings.llm_provider is not LLMProviderName.ANTHROPIC:
        sys.stderr.write("smoke-llm calls the real model: set LLM_PROVIDER=anthropic\n")
        return 2
    return asyncio.run(
        run(sys.argv[1:] if argv is None else argv, settings, provider_factory, sys.stdout)
    )
