"""Keep .env.example and Settings in sync, so a fresh clone is configured correctly."""

import pathlib
import re

import pytest
from pydantic import ValidationError

from app.core.settings import Settings

ENV_EXAMPLE = pathlib.Path(__file__).resolve().parents[3] / ".env.example"
# Read by docker-compose.yml only (port publishing), never by the application.
COMPOSE_ONLY_KEYS = {"BIND_ADDRESS", "API_PORT", "FRONTEND_PORT"}
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


def test_env_example_has_no_keys_beyond_settings_and_compose() -> None:
    assert example_values().keys() - settings_keys() == COMPOSE_ONLY_KEYS


def use_example(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in example_values().items():
        if key not in COMPOSE_ONLY_KEYS:
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
