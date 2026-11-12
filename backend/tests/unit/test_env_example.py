"""Keep .env.example and Settings in sync, so a fresh clone is configured correctly."""

import pathlib
import re

import pytest

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


def test_env_example_values_are_valid_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in example_values().items():
        if key not in COMPOSE_ONLY_KEYS:
            monkeypatch.setenv(key, value)

    Settings()
