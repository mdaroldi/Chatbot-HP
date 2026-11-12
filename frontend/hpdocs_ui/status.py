"""Backend status lookup for the status page."""

import logging
from dataclasses import dataclass

import httpx

# Short and explicit: a status page must answer fast even when the backend hangs.
STATUS_TIMEOUT = httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=1.0)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendStatus:
    reachable: bool
    http_status: int | None
    body: dict[str, object] | None
    request_id: str | None


UNREACHABLE = BackendStatus(reachable=False, http_status=None, body=None, request_id=None)


def create_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """One attempt per check (no retries, no redirects), so the page reflects reality."""
    return httpx.Client(
        timeout=STATUS_TIMEOUT,
        follow_redirects=False,
        transport=transport or httpx.HTTPTransport(retries=0),
    )


def fetch_status(
    base_url: str, path: str, *, transport: httpx.BaseTransport | None = None
) -> BackendStatus:
    """GET base_url + path. Never raises: any failure is reported as unreachable."""
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    try:
        with create_client(transport) as client:
            response = client.get(url)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        # Type only: exception messages can carry hosts or internal details.
        _logger.warning("backend status check failed: %s", type(exc).__name__)
        return UNREACHABLE
    return BackendStatus(
        reachable=True,
        http_status=response.status_code,
        body=_json_object(response),
        request_id=response.headers.get("x-request-id"),
    )


def _json_object(response: httpx.Response) -> dict[str, object] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None
