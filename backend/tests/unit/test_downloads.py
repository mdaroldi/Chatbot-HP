"""The CI download step: retries with backoff, and failures that never read as test failures."""

import logging
from collections.abc import Awaitable, Callable

import pytest

from app.domain.ingestion import (
    ChecksumMismatchError,
    CorpusDownloadError,
    ModelDownloadError,
    ModelIntegrityError,
)
from tests.downloads import (
    CORPUS_CHANGED,
    CORPUS_UNAVAILABLE,
    MODEL_CHANGED,
    MODEL_UNAVAILABLE,
    failure_message,
    main,
    with_retries,
)

UNREACHABLE = CorpusDownloadError("envy-6000-user-guide: ConnectTimeout")
CHANGED = ChecksumMismatchError("envy-6000-user-guide: SHA-256 abc, pinned def")
HUB_DOWN = ModelDownloadError(
    "demo/dense: cannot download demo/dense-onnx@0123", model="demo/dense"
)
HUB_CHANGED = ModelIntegrityError("demo/dense", "model.onnx", expected="def", actual="abc")


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def failing(
    errors: list[Exception], result: str = "ok"
) -> tuple[Callable[[], Awaitable[str]], list[int]]:
    """A fetch that raises each error in turn, then returns `result`; counts its calls."""
    calls: list[int] = []

    async def fetch() -> str:
        calls.append(len(calls) + 1)
        if errors:
            raise errors.pop(0)
        return result

    return fetch, calls


@pytest.mark.parametrize(
    ("error", "label"),
    [
        (UNREACHABLE, CORPUS_UNAVAILABLE),
        (CHANGED, CORPUS_CHANGED),
        (HUB_DOWN, MODEL_UNAVAILABLE),
        (HUB_CHANGED, MODEL_CHANGED),
    ],
)
def test_each_failure_has_its_own_label_and_is_not_a_test_failure(
    error: Exception, label: str
) -> None:
    message = failure_message(error)

    assert message.startswith(f"{label} (")
    assert "not a test failure" in message
    assert message.endswith(f": {error}")


def test_the_four_labels_are_distinct() -> None:
    labels = [CORPUS_UNAVAILABLE, CORPUS_CHANGED, MODEL_UNAVAILABLE, MODEL_CHANGED]

    assert len(set(labels)) == 4
    assert CORPUS_UNAVAILABLE == "CORPUS UNAVAILABLE"
    assert CORPUS_CHANGED == "CORPUS CHANGED UPSTREAM"


def test_a_changed_file_message_points_at_its_pin_file() -> None:
    assert "backend/corpus.toml" in failure_message(CHANGED)
    assert "backend/models.toml" in failure_message(HUB_CHANGED)


async def test_a_first_success_does_not_sleep() -> None:
    fetch, calls = failing([])
    sleeps = Sleeps()

    assert await with_retries(fetch, attempts=3, backoff_seconds=10, sleep=sleeps) == "ok"
    assert (calls, sleeps.delays) == ([1], [])


@pytest.mark.parametrize("error", [UNREACHABLE, HUB_DOWN])
async def test_unreachable_hosts_are_retried_with_a_doubling_backoff(error: Exception) -> None:
    fetch, calls = failing([error, error])
    sleeps = Sleeps()

    assert await with_retries(fetch, attempts=3, backoff_seconds=10, sleep=sleeps) == "ok"
    assert (calls, sleeps.delays) == ([1, 2, 3], [10, 20])


async def test_the_last_error_is_raised_after_the_final_attempt() -> None:
    last = CorpusDownloadError("omen-17-service-guide: HTTP 503")
    fetch, calls = failing([UNREACHABLE, UNREACHABLE, last])
    sleeps = Sleeps()

    with pytest.raises(CorpusDownloadError) as raised:
        await with_retries(fetch, attempts=3, backoff_seconds=5, sleep=sleeps)

    assert raised.value is last
    assert (calls, sleeps.delays) == ([1, 2, 3], [5, 10])


@pytest.mark.parametrize("error", [CHANGED, HUB_CHANGED])
async def test_changed_files_are_never_retried(error: Exception) -> None:
    fetch, calls = failing([error])
    sleeps = Sleeps()

    with pytest.raises(type(error)):
        await with_retries(fetch, attempts=3, backoff_seconds=10, sleep=sleeps)

    assert (calls, sleeps.delays) == ([1], [])


async def test_one_attempt_means_no_retry() -> None:
    fetch, calls = failing([UNREACHABLE])
    sleeps = Sleeps()

    with pytest.raises(CorpusDownloadError):
        await with_retries(fetch, attempts=1, backoff_seconds=10, sleep=sleeps)

    assert (calls, sleeps.delays) == ([1], [])


@pytest.mark.parametrize("attempts", [0, -1])
async def test_attempts_must_be_positive(attempts: int) -> None:
    fetch, calls = failing([])

    with pytest.raises(ValueError, match="attempts"):
        await with_retries(fetch, attempts=attempts, backoff_seconds=1, sleep=Sleeps())

    assert calls == []


async def test_a_retry_logs_the_error_type_and_delay_only(caplog: pytest.LogCaptureFixture) -> None:
    fetch, _ = failing([UNREACHABLE])

    with caplog.at_level(logging.WARNING, logger="tests.downloads"):
        await with_retries(fetch, attempts=2, backoff_seconds=7, sleep=Sleeps())

    [record] = caplog.records
    assert record.getMessage() == "download failed (CorpusDownloadError); retrying in 7 s"


def test_main_runs_every_fetch_with_the_given_retry_settings() -> None:
    corpus, corpus_calls = failing([UNREACHABLE])
    model, model_calls = failing([HUB_DOWN, HUB_DOWN])
    sleeps = Sleeps()

    status = main(
        ["--attempts", "3", "--backoff-seconds", "2"], fetchers=[corpus, model], sleep=sleeps
    )

    assert status == 0
    assert (corpus_calls, model_calls, sleeps.delays) == ([1, 2], [1, 2, 3], [2, 2, 4])


def test_main_logs_one_line_on_success(caplog: pytest.LogCaptureFixture) -> None:
    fetch, _ = failing([])

    with caplog.at_level(logging.INFO, logger="tests.downloads"):
        main([], fetchers=[fetch, fetch], sleep=Sleeps())

    assert [r.getMessage() for r in caplog.records] == ["2 downloads verified"]


def test_main_defaults_to_three_attempts_and_a_ten_second_backoff() -> None:
    fetch, calls = failing([UNREACHABLE, UNREACHABLE, UNREACHABLE])
    sleeps = Sleeps()

    status = main([], fetchers=[fetch], sleep=sleeps)

    assert (status, calls, sleeps.delays) == (1, [1, 2, 3], [10, 20])


def test_main_reports_the_label_and_stops_at_the_first_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus, _ = failing([CHANGED])
    model, model_calls = failing([])

    status = main([], fetchers=[corpus, model], sleep=Sleeps())

    assert status == 1
    assert model_calls == []
    assert capsys.readouterr().err == failure_message(CHANGED) + "\n"
