"""Dependency health: the probe interface and the readiness result types (no I/O)."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class DependencyProbe(Protocol):
    """Checks that one external dependency can serve requests.

    `ping` returns on success and raises on any failure; callers own the timeout.
    """

    @property
    def name(self) -> str: ...

    async def ping(self) -> None: ...


class ProbeStatus(StrEnum):
    OK = "ok"
    DOWN = "down"


class FailureReason(StrEnum):
    """Deliberately coarse: clients learn what failed, never the internal error."""

    TIMEOUT = "timeout"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class ProbeResult:
    name: str
    status: ProbeStatus
    detail: FailureReason | None = None


@dataclass(frozen=True)
class ReadinessReport:
    results: tuple[ProbeResult, ...]

    @property
    def ready(self) -> bool:
        return all(result.status is ProbeStatus.OK for result in self.results)
