"""The frontend's UI_* settings, .env.example and docker-compose.yml agree both ways.

The backend's parity test owns every other key and leaves UI_* to this one.
"""

import pathlib
import re

import pytest

from hpdocs_ui.settings import FrontendSettings

REPO = pathlib.Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO / ".env.example"
COMPOSE = REPO / "docker-compose.yml"
UI_PREFIX = "UI_"
# Wired by compose to the backend service address; not a user setting.
COMPOSE_WIRED = {"BACKEND_URL": "http://backend:8000"}
UI_KEYS = {
    "UI_STREAM_READ_TIMEOUT_SECONDS",
    "UI_STREAM_TOTAL_TIMEOUT_SECONDS",
    "UI_MESSAGE_MAX_CHARS",
}
ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")
UI_COMPOSE_ENTRY = re.compile(r"^\s+(UI_[A-Z0-9_]+):\s*\"?\$\{(UI_[A-Z0-9_]+):-([^}]*)\}\"?\s*$")


def example_values() -> dict[str, str]:
    assert ENV_EXAMPLE.is_file(), f"missing {ENV_EXAMPLE}"
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        match = ASSIGNMENT.match(line)
        if match:
            key, value = match.groups()
            values[key] = value
    return values


def example_ui_values() -> dict[str, str]:
    return {key: value for key, value in example_values().items() if key.startswith(UI_PREFIX)}


def frontend_service_lines() -> list[str]:
    assert COMPOSE.is_file(), f"missing {COMPOSE}"
    lines = COMPOSE.read_text().splitlines()
    start = lines.index("  frontend:")
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(r"^  \S", lines[i])), len(lines)
    )
    return lines[start + 1 : end]


def compose_ui_defaults() -> dict[str, str]:
    found: dict[str, str] = {}
    for line in frontend_service_lines():
        match = UI_COMPOSE_ENTRY.match(line)
        if match:
            key, variable, default = match.groups()
            assert key == variable, f"{key} is read from ${{{variable}}}"
            found[key] = default
    return found


def env_names() -> set[str]:
    return {name.upper() for name in FrontendSettings.model_fields}


def test_every_ui_setting_is_in_env_example_and_nothing_else_is_ui() -> None:
    assert env_names() - set(COMPOSE_WIRED) == UI_KEYS
    assert set(example_ui_values()) == UI_KEYS
    assert all(name.startswith(UI_PREFIX) for name in env_names() - set(COMPOSE_WIRED))


def test_env_example_ui_values_equal_the_settings_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in env_names():
        monkeypatch.delenv(key, raising=False)
    defaults = FrontendSettings()

    assert set(example_ui_values()) == UI_KEYS
    for key, value in example_ui_values().items():
        monkeypatch.setenv(key, value)
        from_example = FrontendSettings()
        monkeypatch.delenv(key)
        field = key.lower()
        assert getattr(from_example, field) == getattr(defaults, field), key


def test_compose_passes_every_ui_setting_with_the_env_example_default() -> None:
    assert set(compose_ui_defaults()) == UI_KEYS
    assert compose_ui_defaults() == example_ui_values()


def test_compose_wires_the_backend_url_to_the_settings_default() -> None:
    lines = [line.strip() for line in frontend_service_lines()]

    for key, value in COMPOSE_WIRED.items():
        assert f"{key}: {value}" in lines
        assert str(FrontendSettings.model_fields[key.lower()].default).rstrip("/") == value
