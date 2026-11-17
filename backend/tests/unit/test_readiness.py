"""Behavior of the readiness check across dependency probes."""

import asyncio
import json

import pytest

from app.domain.health import (
    DependencyUnavailableError,
    FailureReason,
    ProbeResult,
    ProbeStatus,
)
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
    """The probe never answers: the check ends anyway (the outer guard would fail a hang)
    and not before its timeout, read on the event loop's clock that asyncio.timeout uses.
    No upper bound on elapsed time: a slow runner may wake the loop late.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()

    async with asyncio.timeout(5):
        report = await check_readiness([FakeProbe("qdrant", hang=True)], timeout_seconds=0.05)

    assert report.results == (ProbeResult("qdrant", ProbeStatus.DOWN, FailureReason.TIMEOUT),)
    assert loop.time() - started >= 0.05


async def test_probe_raising_timeout_itself_is_reported_as_timeout() -> None:
    report = await check_readiness([FakeProbe("qdrant", error=TimeoutError())], TIMEOUT)

    assert report.results[0].detail is FailureReason.TIMEOUT


class BarrierProbe:
    """A probe that finishes only once every probe of its group has started."""

    def __init__(self, name: str, barrier: asyncio.Barrier) -> None:
        self.name = name
        self._barrier = barrier
        self.started = False

    async def ping(self) -> dict[str, str] | None:
        self.started = True
        await self._barrier.wait()
        return None


async def test_probes_run_concurrently() -> None:
    """Structural, not timed: the barrier opens only when all three probes are inside
    ping() at once, so a sequential check blocks on the first probe and the outer
    timeout fails the test. A concurrent one passes at once, however slow the runner.
    """
    barrier = asyncio.Barrier(3)
    probes = [BarrierProbe(name, barrier) for name in ("a", "b", "c")]

    async with asyncio.timeout(5):
        report = await check_readiness(probes, timeout_seconds=30)

    assert report.ready is True
    assert [probe.started for probe in probes] == [True, True, True]


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


@pytest.mark.parametrize("reason", [FailureReason.LOADING, FailureReason.FAILED])
async def test_a_probe_can_say_why_it_is_down(reason: FailureReason) -> None:
    probe = FakeProbe("models", error=DependencyUnavailableError("detail", reason=reason))

    report = await check_readiness([probe], TIMEOUT)

    assert report.results == (ProbeResult("models", ProbeStatus.DOWN, reason),)


async def test_an_unavailable_dependency_without_a_reason_is_unreachable() -> None:
    probe = FakeProbe("qdrant", error=DependencyUnavailableError("503"))

    report = await check_readiness([probe], TIMEOUT)

    assert report.results == (ProbeResult("qdrant", ProbeStatus.DOWN, FailureReason.UNREACHABLE),)


async def test_a_ready_probe_can_describe_what_it_serves() -> None:
    probe = FakeProbe("models", info={"dense": "BAAI/bge-small-en-v1.5"})

    report = await check_readiness([probe], TIMEOUT)

    assert report.results == (
        ProbeResult("models", ProbeStatus.OK, info={"dense": "BAAI/bge-small-en-v1.5"}),
    )
