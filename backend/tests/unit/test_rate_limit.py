"""The in-memory token bucket limiter: bursts, refill, retry hints and bounded memory."""

import pytest

from app.core.rate_limit import RateLimit, TokenBucketLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def limiter(
    per_minute: int = 60, burst: int = 3, max_keys: int = 100
) -> tuple[TokenBucketLimiter, Clock]:
    clock = Clock()
    return TokenBucketLimiter(RateLimit(per_minute, burst), max_keys=max_keys, clock=clock), clock


def test_a_burst_is_allowed_at_once_then_the_next_call_waits() -> None:
    bucket, _ = limiter(per_minute=60, burst=3)

    assert [bucket.acquire("a") for _ in range(3)] == [None, None, None]
    assert bucket.acquire("a") == pytest.approx(1.0)  # one token per second


def test_tokens_refill_at_the_per_minute_rate() -> None:
    bucket, clock = limiter(per_minute=10, burst=1)
    assert bucket.acquire("a") is None
    assert bucket.acquire("a") == pytest.approx(6.0)

    clock.now += 2
    assert bucket.acquire("a") == pytest.approx(4.0)  # a refused call costs nothing
    clock.now += 4
    assert bucket.acquire("a") is None


def test_an_idle_key_refills_only_up_to_the_burst() -> None:
    bucket, clock = limiter(per_minute=60, burst=2)
    bucket.acquire("a")
    clock.now += 3600

    assert [bucket.acquire("a") for _ in range(3)][:2] == [None, None]
    assert bucket.acquire("a") is not None


def test_keys_have_independent_buckets() -> None:
    bucket, _ = limiter(burst=1)
    bucket.acquire("a")

    assert bucket.acquire("a") is not None
    assert bucket.acquire("b") is None


def test_the_least_recently_used_key_is_evicted_beyond_max_keys() -> None:
    bucket, _ = limiter(burst=1, max_keys=2)
    bucket.acquire("a")
    bucket.acquire("b")
    assert bucket.acquire("a") is not None  # touches a: b is now the oldest

    bucket.acquire("c")  # evicts b

    assert len(bucket) == 2
    assert bucket.acquire("a") is not None  # kept, still empty
    assert bucket.acquire("b") is None  # forgotten, so it starts full again


def test_a_clock_that_goes_backwards_never_adds_tokens() -> None:
    bucket, clock = limiter(per_minute=60, burst=1)
    bucket.acquire("a")
    clock.now -= 100

    assert bucket.acquire("a") == pytest.approx(1.0)


@pytest.mark.parametrize(("per_minute", "burst"), [(0, 1), (1, 0), (-5, 3)])
def test_a_limit_needs_a_positive_rate_and_burst(per_minute: int, burst: int) -> None:
    with pytest.raises(ValueError, match="per_minute and burst must be at least 1"):
        RateLimit(per_minute, burst)


def test_max_keys_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_keys must be at least 1"):
        TokenBucketLimiter(RateLimit(1, 1), max_keys=0)
