"""Behavior of describe(): what the status page says for each backend state."""

import pytest

from hpdocs_ui.status import BackendStatus
from hpdocs_ui.view import StatusView, describe

DOWN = BackendStatus(reachable=False, http_status=None, body=None, request_id=None)


def ok(body: dict[str, object], status: int = 200, rid: str = "rid-12345678") -> BackendStatus:
    return BackendStatus(reachable=True, http_status=status, body=body, request_id=rid)


READY_BODY: dict[str, object] = {
    "status": "ready",
    "checks": {"postgres": {"status": "ok"}, "qdrant": {"status": "ok"}},
}


def test_up_and_ready_is_success_with_dependency_lines() -> None:
    view = describe(ok({"status": "ok"}), ok(READY_BODY))

    assert view == StatusView(
        level="success",
        headline="Backend is up and ready.",
        lines=("postgres: ok", "qdrant: ok", "Request ID: rid-12345678"),
    )


def test_up_but_not_ready_is_warning_with_failure_details() -> None:
    body: dict[str, object] = {
        "status": "not_ready",
        "checks": {
            "postgres": {"status": "down", "detail": "unreachable"},
            "qdrant": {"status": "ok"},
        },
    }

    view = describe(ok({"status": "ok"}), ok(body, status=503, rid="rid-87654321"))

    assert view.level == "warning"
    assert view.headline == "Backend is up but not ready."
    assert view.lines == ("postgres: down (unreachable)", "qdrant: ok", "Request ID: rid-87654321")


def test_unreachable_backend_is_error_without_details() -> None:
    view = describe(DOWN, DOWN)

    assert view == StatusView(level="error", headline="Backend is unreachable.", lines=())


def test_backend_answering_health_with_an_error_status_is_error() -> None:
    view = describe(ok({}, status=500), DOWN)

    assert (view.level, view.headline) == ("error", "Backend returned HTTP 500.")


def test_readiness_that_cannot_be_checked_is_warning() -> None:
    view = describe(ok({"status": "ok"}), DOWN)

    assert (view.level, view.headline) == (
        "warning",
        "Backend is up; readiness could not be checked.",
    )


def test_malformed_readiness_body_is_ignored_not_shown() -> None:
    body: dict[str, object] = {"status": "ready", "checks": {"postgres": "weird", "qdrant": 3}}

    view = describe(ok({"status": "ok"}), ok(body, rid="rid-12345678"))

    assert view.lines == ("Request ID: rid-12345678",)


def test_missing_request_id_is_not_shown() -> None:
    body: dict[str, object] = {"status": "ready", "checks": {"qdrant": {"status": "ok"}}}

    view = describe(ok({"status": "ok"}), BackendStatus(True, 200, body, None))

    assert view.lines == ("qdrant: ok",)


def test_ready_response_without_checks_object_shows_no_dependency_lines() -> None:
    view = describe(ok({"status": "ok"}), BackendStatus(True, 200, None, None))

    assert view == StatusView(level="success", headline="Backend is up and ready.", lines=())


# --- the language model provider, for the privacy notice -------------------------


@pytest.mark.parametrize("llm", ["anthropic", "stub"])
def test_the_provider_comes_from_readiness_even_when_not_ready(llm: str) -> None:
    ready = describe(ok({"status": "ok"}), ok({**READY_BODY, "llm": llm}))
    not_ready = describe(ok({"status": "ok"}), ok({"status": "not_ready", "llm": llm}, 503))

    assert (ready.provider, not_ready.provider) == (llm, llm)


@pytest.mark.parametrize(
    "body",
    [
        READY_BODY,
        {**READY_BODY, "llm": "openai"},
        {**READY_BODY, "llm": "Anthropic"},
        {**READY_BODY, "llm": ["anthropic"]},
        {**READY_BODY, "llm": None},
    ],
)
def test_a_missing_or_unknown_provider_is_unknown(body: dict[str, object]) -> None:
    assert describe(ok({"status": "ok"}), ok(body)).provider == "unknown"


def test_the_provider_is_read_even_when_health_fails() -> None:
    body = {**READY_BODY, "llm": "anthropic"}

    assert describe(DOWN, ok(body)).provider == "anthropic"
    assert describe(ok({}, 500), ok(body)).provider == "anthropic"


def test_the_provider_is_unknown_when_readiness_is_unreachable() -> None:
    assert describe(ok({"status": "ok"}), DOWN).provider == "unknown"
    unreadable = BackendStatus(reachable=True, http_status=503, body=None, request_id=None)
    assert describe(ok({"status": "ok"}), unreadable).provider == "unknown"
