"""Rate limits per route group, as FastAPI dependencies (decision 9, ADR 0010).

Chat routes are keyed by the client ID they already require. /v1/search keys by the
client ID when one is valid and falls back to the peer IP; behind the frontend every
browser shares that IP, which is why the chat UI always sends its client ID.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Annotated

from fastapi import Depends, Request

from app.api.dependencies import get_client_id, require_client_id
from app.core.errors import AppError
from app.core.rate_limit import RateLimit, TokenBucketLimiter
from app.core.settings import Settings


class RateLimitedError(AppError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    code = "rate_limited"

    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__("Too many requests; retry after the time in Retry-After.")
        # Whole seconds, rounded up: retrying at the hint always finds a token.
        self.headers = {"Retry-After": str(max(1, math.ceil(retry_after_seconds)))}


@dataclass(frozen=True)
class RateLimiters:
    """One limiter per route group, so search traffic never spends chat tokens."""

    conversations: TokenBucketLimiter
    messages: TokenBucketLimiter
    search: TokenBucketLimiter

    @classmethod
    def from_settings(
        cls, settings: Settings, *, clock: Callable[[], float] = time.monotonic
    ) -> "RateLimiters":
        def limiter(per_minute: int, burst: int) -> TokenBucketLimiter:
            return TokenBucketLimiter(
                RateLimit(per_minute, burst), max_keys=settings.rate_limit_max_keys, clock=clock
            )

        return cls(
            conversations=limiter(
                settings.rate_limit_conversations_per_minute,
                settings.rate_limit_conversations_burst,
            ),
            messages=limiter(
                settings.rate_limit_messages_per_minute, settings.rate_limit_messages_burst
            ),
            search=limiter(settings.rate_limit_search_per_minute, settings.rate_limit_search_burst),
        )


def _limiters(request: Request) -> RateLimiters:
    limiters: RateLimiters = request.app.state.rate_limiters
    return limiters


def _spend(limiter: TokenBucketLimiter, key: str) -> None:
    retry_after = limiter.acquire(key)
    if retry_after is not None:
        raise RateLimitedError(retry_after)


def limit_conversations(
    request: Request, client_id: Annotated[str, Depends(require_client_id)]
) -> None:
    """Runs after the client ID check, so a rejected request spends no token."""
    _spend(_limiters(request).conversations, client_id)


def limit_messages(request: Request, client_id: Annotated[str, Depends(require_client_id)]) -> None:
    """Questions have their own, smaller bucket: each one may cost a model call."""
    _spend(_limiters(request).messages, client_id)


def limit_search(request: Request) -> None:
    client_id = get_client_id(request)
    if client_id is not None:
        key = f"client:{client_id}"
    else:
        key = f"ip:{request.client.host if request.client else 'unknown'}"
    _spend(_limiters(request).search, key)
