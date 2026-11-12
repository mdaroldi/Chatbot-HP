"""HTTP client used to reach Qdrant's REST API."""

import httpx

from app.core.settings import Settings


def create_qdrant_http_client(settings: Settings) -> httpx.AsyncClient:
    """Explicit timeouts on every phase and no redirects: a probe must answer for the
    configured host only, and never hang longer than the readiness budget.
    """
    seconds = settings.readiness_timeout_seconds
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=seconds, read=seconds, write=seconds, pool=seconds),
        follow_redirects=False,
    )
