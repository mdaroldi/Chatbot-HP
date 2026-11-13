"""Dependency health: the probe interface and the readiness result types (no I/O)."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class DependencyProbe(Protocol):
    """Checks that one external dependency can serve requests.

    `ping` returns on success and raises on any failure; callers own the timeout.
    """

    @property
    def name(self) -> str: ...

    async def ping(self) -> Mapping[str, str] | None: ...


class DependencyUnavailableError(Exception):
    """A dependency answered, but not in a way that means it can serve requests."""

    def __init__(self, message: str, *, reason: "FailureReason | None" = None) -> None:
        super().__init__(message)
        self.reason = reason


class ProbeStatus(StrEnum):
    OK = "ok"
    DOWN = "down"


class FailureReason(StrEnum):
    """Deliberately coarse: clients learn what failed, never the internal error."""

    TIMEOUT = "timeout"
    UNREACHABLE = "unreachable"
    LOADING = "loading"
    FAILED = "failed"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class ProbeResult:
    name: str
    status: ProbeStatus
    detail: FailureReason | None = None
    info: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ReadinessReport:
    results: tuple[ProbeResult, ...]

    @property
    def ready(self) -> bool:
        return all(result.status is ProbeStatus.OK for result in self.results)
