"""Readiness orchestration: run every dependency probe concurrently under a timeout."""

import asyncio
import logging
from collections.abc import Sequence

from app.domain.health import (
    DependencyProbe,
    DependencyUnavailableError,
    FailureReason,
    ProbeResult,
    ProbeStatus,
    ReadinessReport,
)

_logger = logging.getLogger(__name__)


async def check_readiness(
    probes: Sequence[DependencyProbe], timeout_seconds: float
) -> ReadinessReport:
    """Probe all dependencies at once, so /ready costs the slowest probe, not the sum."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    results = await asyncio.gather(*(_run_probe(probe, timeout_seconds) for probe in probes))
    return ReadinessReport(tuple(results))


async def _run_probe(probe: DependencyProbe, timeout_seconds: float) -> ProbeResult:
    try:
        async with asyncio.timeout(timeout_seconds):
            info = await probe.ping()
    except TimeoutError as exc:
        return _down(probe, FailureReason.TIMEOUT, exc)
    except DependencyUnavailableError as exc:
        # A probe may know why (models still loading); otherwise it is unreachable.
        return _down(probe, exc.reason or FailureReason.UNREACHABLE, exc)
    except Exception as exc:
        return _down(probe, FailureReason.UNREACHABLE, exc)
    return ProbeResult(probe.name, ProbeStatus.OK, info=info)


def _down(probe: DependencyProbe, reason: FailureReason, exc: Exception) -> ProbeResult:
    # Only the exception type: messages can hold hosts, credentials or user data.
    _logger.warning(
        "readiness probe failed",
        extra={"probe": probe.name, "reason": reason.value, "error_type": type(exc).__name__},
    )
    return ProbeResult(probe.name, ProbeStatus.DOWN, reason)
