"""Pure mapping from backend status results to what the page displays."""

from dataclasses import dataclass, replace
from http import HTTPStatus
from typing import Literal

from hpdocs_ui.status import BackendStatus

Level = Literal["success", "warning", "error"]
# Where the answers are written, as /ready names it; anything else is unknown.
Provider = Literal["anthropic", "stub", "unknown"]


@dataclass(frozen=True)
class StatusView:
    level: Level
    headline: str
    lines: tuple[str, ...]
    provider: Provider = "unknown"


def describe(health: BackendStatus, ready: BackendStatus) -> StatusView:
    """Fixed headlines only; backend data appears only as validated short fields.

    The provider is read from /ready whatever the health check said: the privacy
    notice must not claim more than the backend reported.
    """
    return replace(_levels(health, ready), provider=_provider(ready.body))


def _levels(health: BackendStatus, ready: BackendStatus) -> StatusView:
    if not health.reachable:
        return StatusView("error", "Backend is unreachable.", ())
    if health.http_status != HTTPStatus.OK:
        return StatusView("error", f"Backend returned HTTP {health.http_status}.", ())
    if not ready.reachable:
        return StatusView("warning", "Backend is up; readiness could not be checked.", ())
    lines = (*_dependency_lines(ready.body), *_request_id_line(ready.request_id))
    if ready.http_status == HTTPStatus.OK:
        return StatusView("success", "Backend is up and ready.", lines)
    return StatusView("warning", "Backend is up but not ready.", lines)


def _dependency_lines(body: dict[str, object] | None) -> tuple[str, ...]:
    checks = body.get("checks") if body else None
    if not isinstance(checks, dict):
        return ()
    lines = []
    for name, check in checks.items():
        if not isinstance(check, dict) or not isinstance(check.get("status"), str):
            continue
        detail = check.get("detail")
        suffix = f" ({detail})" if isinstance(detail, str) else ""
        lines.append(f"{name}: {check['status']}{suffix}")
    return tuple(lines)


def _request_id_line(request_id: str | None) -> tuple[str, ...]:
    return (f"Request ID: {request_id}",) if request_id else ()


def _provider(body: dict[str, object] | None) -> Provider:
    llm = body.get("llm") if body else None
    if llm == "anthropic":
        return "anthropic"
    if llm == "stub":
        return "stub"
    return "unknown"
