"""The stack log audit's pure helpers: canary tokens, the scan and its public summary."""

import json
from urllib.parse import quote, quote_plus

import pytest

from app.audit.log_scan import (
    Canary,
    EmptyLogError,
    Leak,
    find_leaks,
    new_canary,
    summary,
)

NONCE = "0f3a9c"


def test_a_canary_token_is_lowercase_alphanumeric_and_carries_its_id_and_nonce() -> None:
    canary = new_canary("q1", NONCE)

    assert canary.id == "q1"
    assert canary.token.isalnum()
    assert canary.token == canary.token.lower()
    assert "q1" in canary.token
    assert NONCE in canary.token


def test_tokens_differ_by_id_and_by_nonce() -> None:
    tokens = {
        new_canary("q1", NONCE).token,
        new_canary("q2", NONCE).token,
        new_canary("q1", "77aa01").token,
    }

    assert len(tokens) == 3


@pytest.mark.parametrize("bad_id", ["", "Q1", "q-1", "q 1", "q1\n"])
def test_an_id_outside_lowercase_alphanumerics_is_rejected(bad_id: str) -> None:
    with pytest.raises(ValueError, match="canary id"):
        new_canary(bad_id, NONCE)


@pytest.mark.parametrize("bad_nonce", ["", "XYZ", "0f3a9c-", "short"])
def test_a_nonce_that_is_not_lowercase_hex_of_six_or_more_is_rejected(bad_nonce: str) -> None:
    with pytest.raises(ValueError, match="nonce"):
        new_canary("q1", bad_nonce)


def test_clean_logs_have_no_leaks() -> None:
    canary = new_canary("q1", NONCE)

    assert find_leaks({"backend": "request completed\nready\n"}, [canary]) == []


def _encodings(text: str) -> list[str]:
    return [text, json.dumps(text), quote(text), quote_plus(text), text.upper()]


@pytest.mark.parametrize("index", range(5))
def test_a_canary_is_found_raw_json_escaped_url_encoded_or_upper_cased(index: int) -> None:
    canary = new_canary("q1", NONCE)
    sent = f'how do I "reset" {canary.token} the printer?'
    line = _encodings(sent)[index]

    leaks = find_leaks({"backend": f"start\n{line}\nend\n"}, [canary])

    assert leaks == [Leak(container="backend", canary_id="q1", lines=1)]


def test_leaks_are_counted_per_line_per_container_and_canary() -> None:
    first, second = new_canary("q1", NONCE), new_canary("ua", NONCE)
    logs = {
        "backend": f"{first.token}\nok\n{first.token} {second.token}\n",
        "frontend": f"{second.token}\n",
        "postgres": "ready\n",
    }

    leaks = find_leaks(logs, [first, second])

    assert leaks == [
        Leak(container="backend", canary_id="q1", lines=2),
        Leak(container="backend", canary_id="ua", lines=1),
        Leak(container="frontend", canary_id="ua", lines=1),
    ]


def test_an_empty_container_log_fails_the_scan_instead_of_passing_it() -> None:
    canary = new_canary("q1", NONCE)

    with pytest.raises(EmptyLogError, match="qdrant"):
        find_leaks({"backend": "ok\n", "qdrant": " \n"}, [canary])


def test_a_scan_with_no_containers_or_no_canaries_is_rejected() -> None:
    canary = new_canary("q1", NONCE)

    with pytest.raises(ValueError, match="container"):
        find_leaks({}, [canary])
    with pytest.raises(ValueError, match="canar"):
        find_leaks({"backend": "ok\n"}, [])


def test_the_summary_names_containers_ids_and_counts_but_never_a_log_line() -> None:
    canary = new_canary("q1", NONCE)
    secret_line = f"user asked {canary.token} about ink"
    leaks = find_leaks({"backend": secret_line + "\n"}, [canary])

    text = summary(leaks)

    assert text == "1 leak(s): backend q1 lines=1"
    assert "ink" not in text
    assert canary.token not in text


def test_the_summary_of_a_clean_scan_says_so() -> None:
    assert summary([]) == "no leaks"


def test_canary_is_a_value_object() -> None:
    assert new_canary("q1", NONCE) == Canary(id="q1", token=new_canary("q1", NONCE).token)
