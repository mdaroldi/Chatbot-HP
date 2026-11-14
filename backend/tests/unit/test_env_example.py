"""Keep .env.example and Settings in sync, so a fresh clone is configured correctly."""

import pathlib
import re
from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from app.core.settings import Settings

ENV_EXAMPLE = pathlib.Path(__file__).resolve().parents[3] / ".env.example"
# Read by docker-compose.yml only (port publishing), never by the application.
COMPOSE_ONLY_KEYS = {"BIND_ADDRESS", "API_PORT", "FRONTEND_PORT"}
# The frontend's settings; frontend/tests/test_env_parity.py checks them both ways.
FRONTEND_PREFIX = "UI_"
ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


def example_values() -> dict[str, str]:
    assert ENV_EXAMPLE.is_file(), f"missing {ENV_EXAMPLE}"
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        match = ASSIGNMENT.match(line)
        if match:
            key, value = match.groups()
            assert key not in values, f"{key} is defined twice"
            values[key] = value
    return values


def settings_keys() -> set[str]:
    return {name.upper() for name in Settings.model_fields}


def test_every_settings_field_is_documented_in_env_example() -> None:
    assert settings_keys() - example_values().keys() == set()


def keys_beyond_settings(keys: Iterable[str]) -> set[str]:
    """Keys the backend does not read, except the frontend's own UI_* keys."""
    return {key for key in set(keys) - settings_keys() if not key.startswith(FRONTEND_PREFIX)}


def test_env_example_has_no_keys_beyond_settings_and_compose() -> None:
    assert keys_beyond_settings(example_values()) == COMPOSE_ONLY_KEYS


def test_an_unknown_key_is_reported_unless_it_belongs_to_the_frontend() -> None:
    keys = {"LOG_LEVEL", "UI_STREAM_READ_TIMEOUT_SECONDS", "BOGUS_KEY", "UIX_NOT_FRONTEND"}

    assert keys_beyond_settings(keys) == {"BOGUS_KEY", "UIX_NOT_FRONTEND"}


def use_example(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in example_values().items():
        if key not in COMPOSE_ONLY_KEYS and not key.startswith(FRONTEND_PREFIX):
            monkeypatch.setenv(key, value)


def test_env_example_selects_the_anthropic_provider_and_leaves_the_key_to_fill_in() -> None:
    values = example_values()

    assert values["LLM_PROVIDER"] == "anthropic"
    assert values["ANTHROPIC_API_KEY"] == ""


def test_env_example_values_are_valid_settings_once_the_key_is_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_example(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")  # not a real key

    Settings()


def test_env_example_without_the_key_fails_at_startup_naming_both_fixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The README's `cp .env.example .env`, then add the key: a forgotten key stops
    the backend at startup with the two ways out, instead of failing every question.
    """
    use_example(monkeypatch)

    with pytest.raises(ValidationError) as raised:
        Settings()

    assert "ANTHROPIC_API_KEY" in str(raised.value)
    assert "LLM_PROVIDER=stub" in str(raised.value)


def seconds(values: dict[str, str], key: str) -> float:
    return float(values[key])


def test_the_ui_stream_read_timeout_outlasts_the_backends_wait_for_the_first_byte() -> None:
    """No byte arrives before the first token: the rewrite, retrieval (5 s allowed) and
    the first-token deadline all pass in silence (ADR 0010, priming).
    """
    values = example_values()

    first_byte = (
        seconds(values, "LLM_REWRITE_TIMEOUT_SECONDS")
        + seconds(values, "LLM_FIRST_TOKEN_TIMEOUT_SECONDS")
        + 5
    )
    assert seconds(values, "UI_STREAM_READ_TIMEOUT_SECONDS") > first_byte


def test_the_ui_stream_read_timeout_outlasts_the_heartbeat() -> None:
    values = example_values()

    assert seconds(values, "UI_STREAM_READ_TIMEOUT_SECONDS") > seconds(
        values, "SSE_HEARTBEAT_SECONDS"
    )


def test_the_ui_stream_total_timeout_covers_the_longest_answer() -> None:
    values = example_values()

    longest = (
        seconds(values, "LLM_REWRITE_TIMEOUT_SECONDS")
        + seconds(values, "LLM_TOTAL_TIMEOUT_SECONDS")
        + 60
    )
    assert seconds(values, "UI_STREAM_TOTAL_TIMEOUT_SECONDS") >= longest
