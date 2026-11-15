"""The load stack (Phase 6): docker-compose.load.yml over docker-compose.yml, under its
own project name, with loadtest/load.env after .env.

The override may set resource limits only, so every hardening invariant of the main
file holds in the load stack too. The load settings live in loadtest/load.env and may
only change load-profile knobs: never quality settings, never secrets.
"""

import pathlib
import re
from typing import Any

import pytest
from pydantic import SecretStr

from app.core.settings import Settings
from tests.unit.test_compose import COMPOSE, MAKEFILE, REPO, load

LOAD_COMPOSE = REPO / "docker-compose.load.yml"
LOAD_ENV = REPO / "loadtest" / "load.env"
RESOURCE_KEYS = {"cpus", "mem_limit", "memswap_limit"}
EXPECTED_LIMITS = {
    "backend": {"cpus": 4, "mem_limit": "2g", "memswap_limit": "2g"},
    "postgres": {"cpus": 1},
    "qdrant": {"cpus": 1},
}
HARDENING = {
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges:true"],
    "read_only": True,
}
# Settings that can change an answer (rule 5 of the brief): never in the load profile.
QUALITY_PREFIXES = ("RETRIEVAL_", "RERANKER_", "EMBEDDING_", "CHUNK", "LLM_ANSWER_", "CHAT_")
SECRETS = {"ANTHROPIC_API_KEY", "POSTGRES_PASSWORD"}
EXPECTED_ENV = {
    "APP_ENV": "production",
    "LOG_LEVEL": "INFO",
    "LLM_PROVIDER": "stub",
    "LLM_STUB_TOKEN_DELAY_MS": "160",
    "API_PORT": "8100",
}
RATE_LIMIT_KEYS = {
    "RATE_LIMIT_CONVERSATIONS_PER_MINUTE",
    "RATE_LIMIT_CONVERSATIONS_BURST",
    "RATE_LIMIT_SEARCH_PER_MINUTE",
    "RATE_LIMIT_SEARCH_BURST",
    "RATE_LIMIT_MESSAGES_PER_MINUTE",
    "RATE_LIMIT_MESSAGES_BURST",
}
INTERPOLATED = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)")


def env_values(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        assert sep, f"not KEY=VALUE: {key}"
        assert key not in values, f"duplicate key {key}"
        values[key] = value
    return values


@pytest.fixture(scope="module")
def override() -> dict[str, Any]:
    return load(LOAD_COMPOSE)


@pytest.fixture(scope="module")
def load_env() -> dict[str, str]:
    assert LOAD_ENV.is_file(), f"missing {LOAD_ENV}"
    return env_values(LOAD_ENV)


# ---------------------------------------------------------------- override


def test_override_sets_only_services(override: dict[str, Any]) -> None:
    assert set(override) == {"services"}


def test_override_sets_exactly_the_approved_resource_limits(override: dict[str, Any]) -> None:
    assert override["services"] == EXPECTED_LIMITS


def test_override_touches_resource_keys_only(override: dict[str, Any]) -> None:
    for name, service in override["services"].items():
        assert set(service) <= RESOURCE_KEYS, name


def test_every_overridden_service_exists_in_the_main_file(override: dict[str, Any]) -> None:
    assert set(override["services"]) <= set(load(COMPOSE)["services"])


@pytest.mark.parametrize("name", sorted(EXPECTED_LIMITS))
def test_the_merged_load_stack_keeps_the_hardening(override: dict[str, Any], name: str) -> None:
    merged = {**load(COMPOSE)["services"][name], **override["services"][name]}

    for key, value in HARDENING.items():
        assert merged[key] == value, (name, key)
    assert "ports" not in override["services"][name]
    assert merged.get("privileged") is not True


# ---------------------------------------------------------------- load.env


def test_load_env_sets_the_load_profile(load_env: dict[str, str]) -> None:
    for key, value in EXPECTED_ENV.items():
        assert load_env[key] == value, key


def test_load_env_raises_every_rate_limit_out_of_the_way(load_env: dict[str, str]) -> None:
    for key in RATE_LIMIT_KEYS:
        assert int(load_env[key]) == 10_000, key


def test_load_env_holds_no_secret(load_env: dict[str, str]) -> None:
    assert not set(load_env) & SECRETS


def test_load_env_changes_no_quality_setting(load_env: dict[str, str]) -> None:
    assert not [key for key in load_env if key.startswith(QUALITY_PREFIXES)]


def test_load_env_only_sets_variables_the_main_file_reads(load_env: dict[str, str]) -> None:
    assert set(load_env) <= set(INTERPOLATED.findall(COMPOSE.read_text()))


def test_load_env_is_valid_backend_configuration(load_env: dict[str, str]) -> None:
    backend_keys = set(Settings.model_fields)
    values = {k.lower(): v for k, v in load_env.items() if k.lower() in backend_keys}

    Settings(**values, postgres_password=SecretStr("unused"))  # type: ignore[arg-type]


# ---------------------------------------------------------------- Makefile


def make_line(name: str) -> str:
    match = re.search(rf"^{name} :?= (.+)$", MAKEFILE.read_text(), re.MULTILINE)
    assert match, f"{name} missing from the Makefile"
    return match.group(1)


def recipe(target: str) -> str:
    match = re.search(rf"^{target}:.*\n((?:\t.*\n)+)", MAKEFILE.read_text(), re.MULTILINE)
    assert match, f"target {target} missing"
    return match.group(1)


def test_the_load_stack_has_its_own_project_and_both_files() -> None:
    command = make_line("LOAD_COMPOSE")

    assert command.split() == [
        "docker",
        "compose",
        "-p",
        "hpdocs-load",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.load.yml",
        "--env-file",
        ".env",
        "--env-file",
        "loadtest/load.env",
    ]


def test_load_up_starts_only_the_backend_and_its_dependencies() -> None:
    lines = recipe("load-up").split("\n")

    up = next(line for line in lines if " up " in line)
    assert "$(LOAD_COMPOSE)" in up
    assert up.rstrip().endswith(" backend")
    assert "LLM_PROVIDER=$(LOAD_PROVIDER)" in up


def test_load_reset_clears_conversation_data_only() -> None:
    body = recipe("load-reset")

    assert "TRUNCATE messages, conversations" in body
    assert "index_builds" not in body
    assert "down" not in body.split()
    assert not re.search(r"\bvolume|\brm\b", body)


def test_load_targets_never_touch_the_main_project() -> None:
    for target in ("load-up", "load-down", "load-reset"):
        for line in recipe(target).splitlines():
            if "docker compose" in line or "$(LOAD_COMPOSE)" in line:
                assert "$(LOAD_COMPOSE)" in line, (target, line)


def test_load_runs_locust_headless_on_the_host_against_the_load_api() -> None:
    body = recipe("load")
    locust = next(line for line in body.splitlines() if " locust " in line)

    assert "cd $(LOADTEST) && $(LOAD_ENV) uv run" in locust
    assert "uv run --frozen locust -f locustfile.py --headless" in locust
    assert "--host $(LOAD_API_URL)" in locust
    assert make_line("LOAD_API_URL") == "http://127.0.0.1:8100"
    assert "docker" not in locust


def test_load_needs_a_run_name_and_passes_the_run_settings_on() -> None:
    body = recipe("load")
    env = make_line("LOAD_ENV")

    assert 'test -n "$(LOAD_RUN)"' in body
    for name in (
        "LOAD_RUN",
        "LOAD_STEPS",
        "LOAD_STEP_SECONDS",
        "LOAD_WARMUP_SECONDS",
        "LOAD_SPAWN_RATE",
        "LOAD_SEED",
        "LOAD_SAMPLE_SECONDS",
    ):
        assert f'{name}="$({name})"' in env.split(), name
