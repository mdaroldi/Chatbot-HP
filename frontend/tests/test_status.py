"""Behavior of fetch_status: one HTTP call, short timeouts, never raises."""

from collections.abc import Callable

import httpx
import pytest

from hpdocs_ui.status import STATUS_TIMEOUT, BackendStatus, create_client, fetch_status

Handler = Callable[[httpx.Request], httpx.Response]
BASE = "http://backend:8000/"


def fetch(handler: Handler, path: str = "/health", base: str = BASE) -> BackendStatus:
    return fetch_status(base, path, transport=httpx.MockTransport(handler))


def test_reachable_backend_returns_status_body_and_request_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://backend:8000/health"
        return httpx.Response(200, json={"status": "ok"}, headers={"x-request-id": "rid-12345678"})

    assert fetch(handler) == BackendStatus(
        reachable=True, http_status=200, body={"status": "ok"}, request_id="rid-12345678"
    )


def test_error_status_is_still_reachable_with_its_body() -> None:
    result = fetch(lambda _: httpx.Response(503, json={"status": "not_ready", "checks": {}}))

    assert (result.reachable, result.http_status) == (True, 503)
    assert result.body == {"status": "not_ready", "checks": {}}
    assert result.request_id is None


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(httpx.Response(200, text="<html>proxy page</html>"), id="not-json"),
        pytest.param(httpx.Response(200, json=["a", "list"]), id="json-not-object"),
    ],
)
def test_non_object_bodies_are_dropped(response: httpx.Response) -> None:
    result = fetch(lambda _: response)

    assert (result.reachable, result.http_status, result.body) == (True, 200, None)


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError],
)
def test_transport_failures_mean_unreachable_and_never_raise(
    error: type[httpx.TransportError],
) -> None:
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        raise error("backend secret detail", request=request)

    assert fetch(handler) == BackendStatus(
        reachable=False, http_status=None, body=None, request_id=None
    )
    assert attempts == ["/health"]


@pytest.mark.parametrize("base", ["backend:8000", "http://", "::not-a-url::"])
def test_malformed_base_url_means_unreachable_and_never_raises(base: str) -> None:
    result = fetch_status(base, "/health")

    assert result.reachable is False


def test_failed_request_is_attempted_once_without_retries() -> None:
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        raise httpx.ConnectError("refused", request=request)

    fetch(handler)

    assert attempts == ["/health"]


def test_redirects_are_not_followed() -> None:
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        return httpx.Response(307, headers={"location": "http://elsewhere/health"})

    result = fetch(handler)

    assert (result.reachable, result.http_status) == (True, 307)
    assert attempts == ["/health"]


def test_base_url_path_is_preserved() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={})

    fetch(handler, path="/ready", base="https://api.example.com/hp/")

    assert seen == ["https://api.example.com/hp/ready"]


def test_client_has_short_explicit_timeouts_no_redirects_and_no_retries() -> None:
    with create_client() as client:
        assert client.timeout == STATUS_TIMEOUT
        assert client.follow_redirects is False

    assert STATUS_TIMEOUT.connect is not None
    assert STATUS_TIMEOUT.connect <= 2
    assert STATUS_TIMEOUT.read is not None
    assert STATUS_TIMEOUT.read <= 3
