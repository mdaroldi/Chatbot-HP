"""Behavior of the readiness check across dependency probes."""

import json
import time

import pytest

from app.domain.health import FailureReason, ProbeResult, ProbeStatus
from app.services.readiness import check_readiness
from tests.unit.conftest import LogReader
from tests.unit.fakes import FakeProbe

TIMEOUT = 0.5


async def test_all_probes_ok_means_ready_with_results_in_probe_order() -> None:
    report = await check_readiness([FakeProbe("postgres"), FakeProbe("qdrant")], TIMEOUT)

    assert report.ready is True
    assert report.results == (
        ProbeResult("postgres", ProbeStatus.OK),
        ProbeResult("qdrant", ProbeStatus.OK),
    )


async def test_no_probes_means_ready() -> None:
    report = await check_readiness([], TIMEOUT)

    assert report.ready is True
    assert report.results == ()


async def test_each_probe_is_called_once() -> None:
    probes = [FakeProbe("postgres"), FakeProbe("qdrant")]

    await check_readiness(probes, TIMEOUT)

    assert [p.calls for p in probes] == [1, 1]


async def test_failing_probe_is_down_unreachable_and_others_still_checked() -> None:
    report = await check_readiness(
        [FakeProbe("postgres", error=ConnectionRefusedError()), FakeProbe("qdrant")], TIMEOUT
    )

    assert report.ready is False
    assert report.results == (
        ProbeResult("postgres", ProbeStatus.DOWN, FailureReason.UNREACHABLE),
        ProbeResult("qdrant", ProbeStatus.OK),
    )


async def test_hanging_probe_is_down_timeout_after_the_timeout() -> None:
    started = time.perf_counter()

    report = await check_readiness([FakeProbe("qdrant", hang=True)], timeout_seconds=0.05)

    elapsed = time.perf_counter() - started
    assert report.results == (ProbeResult("qdrant", ProbeStatus.DOWN, FailureReason.TIMEOUT),)
    assert 0.05 <= elapsed < 0.5


async def test_probe_raising_timeout_itself_is_reported_as_timeout() -> None:
    report = await check_readiness([FakeProbe("qdrant", error=TimeoutError())], TIMEOUT)

    assert report.results[0].detail is FailureReason.TIMEOUT


async def test_probes_run_concurrently() -> None:
    probes = [
        FakeProbe("a", delay_seconds=0.1),
        FakeProbe("b", delay_seconds=0.1),
        FakeProbe("c", delay_seconds=0.15),
    ]
    started = time.perf_counter()

    report = await check_readiness(probes, TIMEOUT)

    elapsed = time.perf_counter() - started
    assert report.ready is True
    # Sequential would take the sum (0.35 s); concurrent takes about the slowest (0.15 s).
    assert 0.15 <= elapsed < 0.25


async def test_unexpected_exception_is_down_unreachable_and_logs_only_its_type(
    json_logs: LogReader,
) -> None:
    error = RuntimeError("password=hunter2 for user secret@example.com")

    report = await check_readiness([FakeProbe("postgres", error=error)], TIMEOUT)

    assert report.results == (ProbeResult("postgres", ProbeStatus.DOWN, FailureReason.UNREACHABLE),)
    logs = json_logs()
    raw = json.dumps(logs)
    assert "hunter2" not in raw
    assert "secret@example.com" not in raw
    failures = [line for line in logs if line["message"] == "readiness probe failed"]
    assert len(failures) == 1
    assert failures[0]["level"] == "WARNING"
    assert failures[0]["probe"] == "postgres"
    assert failures[0]["reason"] == "unreachable"
    assert failures[0]["error_type"] == "RuntimeError"


async def test_timeout_is_logged_with_reason(json_logs: LogReader) -> None:
    await check_readiness([FakeProbe("qdrant", hang=True)], timeout_seconds=0.01)

    failures = [line for line in json_logs() if line["message"] == "readiness probe failed"]
    assert [(f["probe"], f["reason"]) for f in failures] == [("qdrant", "timeout")]


@pytest.mark.parametrize("bad_timeout", [0, -1])
async def test_non_positive_timeout_is_rejected(bad_timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        await check_readiness([FakeProbe("qdrant")], bad_timeout)
