"""Each simulated browser has its own `X-Client-ID`, accepted by the backend's rule."""

import random
import re
import uuid

from hpdocs_load.client_id import new_client_id

# Mirrors backend/app/core/middleware.py (_VALID_REQUEST_ID); the client ID is validated
# like the request ID (ADR 0010). The projects share no code, so the rule is repeated.
BACKEND_RULE = re.compile(r"[A-Za-z0-9-]{8,64}")


def test_client_id_is_a_uuid4_the_backend_accepts() -> None:
    value = new_client_id(random.Random(1))

    assert BACKEND_RULE.fullmatch(value)
    assert uuid.UUID(value).version == 4


def test_the_same_seed_gives_the_same_ids() -> None:
    first, second = random.Random(7), random.Random(7)

    assert [new_client_id(first) for _ in range(3)] == [new_client_id(second) for _ in range(3)]


def test_ids_from_one_generator_are_distinct() -> None:
    rng = random.Random(3)

    assert len({new_client_id(rng) for _ in range(1000)}) == 1000
