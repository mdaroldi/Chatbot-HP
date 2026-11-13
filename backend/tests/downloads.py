"""Downloads for the corpus and model tests, and how their failures read.

CI runs `python -m tests.downloads` (make fetch-test-assets) as its own step before
the tests: it fetches the HP manuals live from HP and verifies them against
backend/corpus.toml, and fetches the pinned models into EMBEDDING_CACHE_DIR. The
tests then find verified files on disk. An unreachable host is retried with a
doubling backoff; a file that no longer matches its pin is not, because retrying
cannot fix it.

Neither is a test failure, and the logs must say so: each one is reported under its
own label (CORPUS UNAVAILABLE, CORPUS CHANGED UPSTREAM, MODEL UNAVAILABLE, MODEL
CHANGED UPSTREAM), never as an assertion error. Messages name documents, models and
digests only, never document content.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import NoReturn

import pytest
from pydantic import SecretStr

from app.core.settings import Settings
from app.domain.ingestion import (
    ChecksumMismatchError,
    CorpusDownloadError,
    CorpusError,
    ModelDownloadError,
    ModelError,
    ModelIntegrityError,
    VerifiedModel,
)
from app.infra.corpus_http import HttpCorpusSource, create_corpus_http_client
from app.infra.model_files import HuggingFaceHub, PinnedModelSource, load_pins
from app.ingestion.corpus import load_corpus

CORPUS_UNAVAILABLE = "CORPUS UNAVAILABLE"
CORPUS_CHANGED = "CORPUS CHANGED UPSTREAM"
MODEL_UNAVAILABLE = "MODEL UNAVAILABLE"
MODEL_CHANGED = "MODEL CHANGED UPSTREAM"

RETRYABLE = (CorpusDownloadError, ModelDownloadError)
Sleep = Callable[[float], Awaitable[None]]
Fetch = Callable[[], Awaitable[object]]

logger = logging.getLogger(__name__)


def failure_message(exc: Exception) -> str:
    """One line that says which download failed and that no test assertion did."""
    if isinstance(exc, ChecksumMismatchError):
        label, why = CORPUS_CHANGED, "the live file differs from its pin in backend/corpus.toml"
    elif isinstance(exc, CorpusError):
        label, why = CORPUS_UNAVAILABLE, "the manual could not be downloaded"
    elif isinstance(exc, ModelIntegrityError):
        label, why = MODEL_CHANGED, "the hub served files that differ from backend/models.toml"
    else:
        label, why = MODEL_UNAVAILABLE, "the pinned model could not be downloaded"
    return f"{label} ({why}; not a test failure): {exc}"


async def with_retries[T](
    fetch: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    backoff_seconds: float = 10.0,
    sleep: Sleep = asyncio.sleep,
) -> T:
    """Retry only unreachable hosts: a changed file stays changed on every attempt."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    delay = backoff_seconds
    for _ in range(attempts - 1):
        try:
            return await fetch()
        except RETRYABLE as exc:
            logger.warning("download failed (%s); retrying in %g s", type(exc).__name__, delay)
            await sleep(delay)
            delay *= 2
    return await fetch()


def download_settings() -> Settings:
    # The password is never used: downloads and corpus or model tests touch no database.
    return Settings(postgres_password=SecretStr("unused-by-corpus-and-model-tests"))


async def fetch_corpus(settings: Settings) -> dict[str, Path]:
    """Both manuals, verified; a verified file already in CORPUS_DIR is reused."""
    documents = load_corpus()
    async with create_corpus_http_client(settings) as client:
        source = HttpCorpusSource.from_settings(client, documents, settings)
        return {d.doc_id: await source.fetch(d) for d in documents}


async def fetch_model(settings: Settings, name: str) -> VerifiedModel:
    """The pinned model, verified in EMBEDDING_CACHE_DIR (downloaded on first use)."""
    source = PinnedModelSource(load_pins(), settings.embedding_cache_dir, HuggingFaceHub())
    return await source.fetch(name)


def fail(exc: Exception) -> NoReturn:
    """End a corpus or model test with the download label instead of a traceback."""
    pytest.fail(failure_message(exc), pytrace=False)


async def fetch_or_fail[T](fetch: Callable[[], Awaitable[T]]) -> T:
    try:
        return await with_retries(fetch)
    except (CorpusError, ModelError) as exc:
        fail(exc)


CORPUS = "corpus"


def planned_downloads(*, corpus: bool) -> list[str]:
    """The corpus (unless left out) and every model in models.toml, rerankers included."""
    return [*([CORPUS] if corpus else []), *load_pins()]


def _fetcher(settings: Settings, name: str) -> Fetch:
    if name == CORPUS:
        return lambda: fetch_corpus(settings)
    return lambda: fetch_model(settings, name)


def _default_fetchers(*, corpus: bool) -> list[Fetch]:
    settings = download_settings()
    return [_fetcher(settings, name) for name in planned_downloads(corpus=corpus)]


async def _run(fetchers: Sequence[Fetch], attempts: int, backoff: float, sleep: Sleep) -> None:
    for fetch in fetchers:
        await with_retries(fetch, attempts=attempts, backoff_seconds=backoff, sleep=sleep)


def main(
    argv: Sequence[str] | None = None,
    *,
    fetchers: Sequence[Fetch] | None = None,
    sleep: Sleep = asyncio.sleep,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=10.0)
    parser.add_argument("--models-only", action="store_true", help="skip the HP manuals")
    args = parser.parse_args(argv)
    chosen = _default_fetchers(corpus=not args.models_only) if fetchers is None else fetchers
    try:
        asyncio.run(_run(chosen, args.attempts, args.backoff_seconds, sleep))
    except (CorpusError, ModelError) as exc:
        sys.stderr.write(failure_message(exc) + "\n")
        return 1
    logger.info("%d downloads verified", len(chosen))
    return 0


if __name__ == "__main__":
    # Warnings from any library, but only this module's own progress lines (httpx would
    # log every request URL at INFO).
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    raise SystemExit(main())
