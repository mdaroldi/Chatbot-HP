"""In-memory token buckets, one per key, with a bounded number of keys.

State lives in the process: each replica limits on its own, so N replicas allow up to
N times the configured rate (documented in the README). Calls run on the event loop
with no await inside, so no lock is needed.
"""

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class RateLimit:
    """A sustained rate and the burst a full bucket allows at once."""

    per_minute: int
    burst: int

    def __post_init__(self) -> None:
        if self.per_minute < 1 or self.burst < 1:
            raise ValueError("per_minute and burst must be at least 1")


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """Allow `burst` calls at once per key, refilled at `per_minute`.

    At most `max_keys` buckets are kept: the least recently used one is forgotten,
    and a forgotten key starts again with a full bucket. That caps memory against a
    flood of fresh keys; anonymous IDs can be rotated anyway (README).
    """

    def __init__(
        self,
        limit: RateLimit,
        *,
        max_keys: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_keys < 1:
            raise ValueError("max_keys must be at least 1")
        self.limit = limit
        self.max_keys = max_keys
        self._clock = clock
        self._rate = limit.per_minute / SECONDS_PER_MINUTE  # tokens per second
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def __len__(self) -> int:
        return len(self._buckets)

    def acquire(self, key: str) -> float | None:
        """Spend one token: None if allowed, else the seconds until one is available.

        A refused call spends nothing, so retrying early only delays the next success.
        """
        now = self._clock()
        bucket = self._bucket(key, now)
        # max(): a clock that steps backwards must never add tokens.
        elapsed = max(0.0, now - bucket.updated)
        bucket.tokens = min(float(self.limit.burst), bucket.tokens + elapsed * self._rate)
        bucket.updated = max(bucket.updated, now)
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return None
        return (1.0 - bucket.tokens) / self._rate

    def _bucket(self, key: str, now: float) -> _Bucket:
        bucket = self._buckets.get(key)
        if bucket is not None:
            self._buckets.move_to_end(key)
            return bucket
        bucket = _Bucket(tokens=float(self.limit.burst), updated=now)
        self._buckets[key] = bucket
        if len(self._buckets) > self.max_keys:
            self._buckets.popitem(last=False)
        return bucket
