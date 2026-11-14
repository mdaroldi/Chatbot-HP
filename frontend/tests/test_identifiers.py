"""Client and request IDs: the backend's allowlist, applied before anything is sent
or shown.
"""

import re
import uuid

import pytest

from hpdocs_ui.identifiers import new_client_id, trusted_identifier

UUID4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


def test_new_client_ids_are_random_uuid4_strings() -> None:
    ids = {new_client_id() for _ in range(50)}

    assert len(ids) == 50
    for client_id in ids:
        assert UUID4.fullmatch(client_id), client_id
        assert uuid.UUID(client_id).version == 4


def test_a_new_client_id_passes_the_backend_allowlist() -> None:
    client_id = new_client_id()

    assert trusted_identifier(client_id) == client_id


@pytest.mark.parametrize(
    "raw",
    ["abcdefgh", "A1-b2-C3-d4", "a" * 64, "rid-12345678", "0f8fad5b-d9cb-469f-a165-70867728950e"],
)
def test_identifiers_matching_the_allowlist_are_kept(raw: str) -> None:
    assert trusted_identifier(raw) == raw


@pytest.mark.parametrize(
    ("raw", "nearest_valid"),
    [
        (None, "abcdefgh"),
        ("", "abcdefgh"),
        ("abcdefg", "abcdefgh"),  # 7 characters
        ("a" * 65, "a" * 64),
        ("abc defgh", "abc-defgh"),
        ("abcdefgh\n", "abcdefgh"),
        ("abcd_efgh", "abcd-efgh"),
        ("abcdéfgh", "abcdefgh"),
        ("abcdefgh\r\nX-Injected: 1", "abcdefgh"),
        ("../../etc/passwd", "etc-passwd"),
    ],
)
def test_identifiers_outside_the_allowlist_are_dropped(raw: str | None, nearest_valid: str) -> None:
    assert trusted_identifier(nearest_valid) == nearest_valid
    assert trusted_identifier(raw) is None
