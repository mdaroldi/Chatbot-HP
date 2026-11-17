"""Invariants of the CI workflows that a review would otherwise have to catch.

The logs and caches of a public repository's workflows are readable by others: no
HP content may be cached or uploaded, and pytest must never print locals (chunk
text). actionlint checks the syntax; these tests check the intent.
"""

import pathlib
import re
import tomllib
from typing import Any

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
CI = REPO / ".github" / "workflows" / "ci.yml"
MAKEFILE = REPO / "Makefile"
PYPROJECTS = [REPO / project / "pyproject.toml" for project in ("backend", "frontend", "loadtest")]

SHA_PINNED = re.compile(r"^[\w.-]+(/[\w.-]+)+@[0-9a-f]{40}$")
USES_LINE = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)(.*)$")
VERSION_COMMENT = re.compile(r"^\s+#\s+v\d+(\.\d+)*$")
EXPECTED_ACTIONS = {
    "actions/checkout",
    "astral-sh/setup-uv",
    "actions/cache/restore",
    "actions/cache/save",
}
MODEL_CACHE_PATH = "backend/models"
MODEL_CACHE_KEY = "embedding-models-${{ hashFiles('backend/models.toml') }}"
TELEMETRY_OFF = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "ORT_DISABLE_TELEMETRY": "1",
}

# Any: parsed YAML is untyped by nature; each test asserts the shape it relies on.
Step = dict[str, Any]


def load(path: pathlib.Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    return data


def jobs(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = load(path).get("jobs") or {}
    return found


def all_steps() -> list[Step]:
    return [step for path in WORKFLOWS for job in jobs(path).values() for step in job["steps"]]


def action(step: Step) -> str:
    uses: str = step.get("uses", "")
    return uses.split("@", 1)[0]


def cache_steps() -> list[Step]:
    return [s for s in all_steps() if action(s).startswith("actions/cache")]


def test_every_action_is_pinned_to_a_commit_sha_with_its_version() -> None:
    lines = [
        m
        for path in WORKFLOWS
        for ln in path.read_text().splitlines()
        if (m := USES_LINE.match(ln))
    ]
    names = {m.group(1).split("@", 1)[0] for m in lines}

    assert names >= EXPECTED_ACTIONS
    assert [m.group(1) for m in lines if not SHA_PINNED.match(m.group(1))] == []
    assert [m.group(1) for m in lines if not VERSION_COMMENT.match(m.group(2))] == []


def test_restore_and_save_come_from_the_same_cache_commit() -> None:
    shas = {s["uses"].split("@", 1)[1] for s in cache_steps()}

    assert {action(s) for s in cache_steps()} == {"actions/cache/restore", "actions/cache/save"}
    assert len(shas) == 1


def test_no_workflow_uploads_artifacts_or_caches_with_the_combined_action() -> None:
    used = {action(s) for s in all_steps()}

    assert used >= EXPECTED_ACTIONS
    assert {a for a in used if "artifact" in a} == set()
    assert "actions/cache" not in used  # its implicit post-job save is not wanted


def test_only_the_embedding_models_are_cached_keyed_by_their_pins() -> None:
    restore = [s for s in cache_steps() if action(s) == "actions/cache/restore"]

    assert len(restore) == 1
    for step in cache_steps():
        assert step["with"]["path"] == MODEL_CACHE_PATH
        assert "restore-keys" not in step["with"]
    assert restore[0]["with"]["key"] == MODEL_CACHE_KEY
    assert not [s for s in cache_steps() if "corpus" in str(s).casefold()]


@pytest.fixture(scope="module")
def corpus_job() -> dict[str, Any]:
    candidates = [
        job
        for job in jobs(CI).values()
        if any(step.get("run") == "make test-int" for step in job["steps"])
    ]
    assert len(candidates) == 1, "exactly one job runs make test-int"
    return candidates[0]


def index(steps: list[Step], match: Any) -> int:
    found = [i for i, step in enumerate(steps) if match(step)]
    assert len(found) == 1
    return found[0]


def test_models_are_restored_before_the_download_and_saved_right_after_it(
    corpus_job: dict[str, Any],
) -> None:
    steps: list[Step] = corpus_job["steps"]
    restore = index(steps, lambda s: action(s) == "actions/cache/restore")
    download = index(steps, lambda s: s.get("run") == "make fetch-test-assets")
    save = index(steps, lambda s: action(s) == "actions/cache/save")
    tests = index(steps, lambda s: s.get("run") == "make test-int")
    restore_id = steps[restore]["id"]

    assert restore < download < save < tests
    assert save == download + 1
    assert steps[save]["if"] == f"steps.{restore_id}.outputs.cache-hit != 'true'"
    assert steps[save]["with"]["key"] == f"${{{{ steps.{restore_id}.outputs.cache-primary-key }}}}"


def test_the_corpus_job_runs_every_marker_and_is_never_optional(
    corpus_job: dict[str, Any],
) -> None:
    runs = [s["run"] for s in all_steps() if "make test-int" in s.get("run", "")]
    default = re.search(r"^TEST_INT_MARKERS \?= (.+)$", MAKEFILE.read_text(), re.MULTILINE)

    assert runs == ["make test-int"]
    assert default is not None
    assert default.group(1) == "integration or corpus or model"
    # A skipped or failure-tolerant job would let a red corpus run pass as green.
    assert {"if", "continue-on-error"} & set(corpus_job) == set()
    assert [s for s in corpus_job["steps"] if "continue-on-error" in s or "if" in s] == [
        s for s in corpus_job["steps"] if action(s) == "actions/cache/save"
    ]


def test_the_corpus_job_turns_third_party_telemetry_off(corpus_job: dict[str, Any]) -> None:
    assert {k: corpus_job.get("env", {}).get(k) for k in TELEMETRY_OFF} == TELEMETRY_OFF


@pytest.fixture(scope="module")
def stack_job() -> dict[str, Any]:
    candidates = [
        job
        for job in jobs(CI).values()
        if any(step.get("run") == "make test-log-audit" for step in job["steps"])
    ]
    assert len(candidates) == 1, "exactly one job runs the stack log audit"
    return candidates[0]


def step_text(step: Step) -> str:
    return str(step.get("run", ""))


def test_only_the_stack_job_writes_an_env_file_and_no_job_builds_with_compose(
    stack_job: dict[str, Any],
) -> None:
    other_jobs = [job for job in jobs(CI).values() if job != stack_job]
    other_text = yaml.safe_dump(other_jobs)
    text = "\n".join(path.read_text() for path in WORKFLOWS)

    assert other_jobs
    assert ".env" not in other_text
    assert "make images" in other_text
    assert "make images" in yaml.safe_dump(stack_job)
    assert "compose build" not in text
    assert "--build" not in text


def env_step(stack_job: dict[str, Any]) -> Step:
    steps: list[Step] = stack_job["steps"]
    return steps[index(steps, lambda s: "> .env" in step_text(s))]


def test_the_stack_env_comes_from_the_example_with_the_stub_and_no_key(
    stack_job: dict[str, Any],
) -> None:
    script = step_text(env_step(stack_job))

    assert ".env.example" in script
    assert "LLM_PROVIDER=stub" in script
    assert re.search(r"ANTHROPIC_API_KEY=(\s|$|/)", script)
    assert re.search(r"LLM_STUB_TOKEN_DELAY_MS=[1-9]", script)
    assert "secrets." not in yaml.safe_dump(stack_job)


def test_the_stack_password_is_generated_and_masked_before_anything_can_print_it(
    stack_job: dict[str, Any],
) -> None:
    lines = [ln.strip() for ln in step_text(env_step(stack_job)).splitlines() if ln.strip()]
    generate = [i for i, ln in enumerate(lines) if ln.startswith("password=")]
    mask = [i for i, ln in enumerate(lines) if ln.startswith('echo "::add-mask::${password}"')]

    assert len(generate) == 1
    assert "openssl rand" in lines[generate[0]]
    assert mask == [generate[0] + 1]
    assert "set -x" not in step_text(env_step(stack_job))
    assert "change-me" not in step_text(env_step(stack_job)).split("s/^POSTGRES_PASSWORD=")[-1]


def test_no_stack_step_prints_the_env_file(stack_job: dict[str, Any]) -> None:
    for step in stack_job["steps"]:
        script = step_text(step)
        assert not re.search(r"\b(cat|less|more|head|tail)\b[^\n|]*\.env\b", script)
        assert "docker compose config" not in script
        assert "printenv" not in script


def test_the_stack_job_runs_the_contract_tests_then_the_audit_in_order(
    stack_job: dict[str, Any],
) -> None:
    steps: list[Step] = stack_job["steps"]
    order = [
        index(steps, lambda s: step_text(s) == "make images"),
        index(steps, lambda s: "> .env" in step_text(s)),
        index(steps, lambda s: step_text(s) == "docker compose up --detach --wait"),
        index(steps, lambda s: step_text(s) == "make test-ui-int"),
        index(steps, lambda s: step_text(s) == "make test-eval-int"),
        index(steps, lambda s: step_text(s) == "make test-log-audit"),
    ]

    assert order == sorted(order)
    assert {"if", "continue-on-error"} & set(stack_job) == set()
    assert [s for s in steps if "continue-on-error" in s] == []


def test_container_logs_are_dumped_on_failure_only_if_the_audit_did_not_fail(
    stack_job: dict[str, Any],
) -> None:
    steps: list[Step] = stack_job["steps"]
    audit = steps[index(steps, lambda s: step_text(s) == "make test-log-audit")]
    dump = index(steps, lambda s: "docker compose logs" in step_text(s))

    assert steps[dump]["if"] == f"failure() && steps.{audit['id']}.outcome != 'failure'"
    assert dump > steps.index(audit)


def test_the_stack_is_always_torn_down_with_its_volumes_last(
    stack_job: dict[str, Any],
) -> None:
    steps: list[Step] = stack_job["steps"]

    assert step_text(steps[-1]) == "docker compose down --volumes"
    assert steps[-1]["if"] == "always()"


def pytest_commands() -> list[str]:
    sources = [MAKEFILE, *WORKFLOWS]
    return [ln for path in sources for ln in path.read_text().splitlines() if "pytest" in ln]


def test_pytest_never_shows_locals() -> None:
    commands = pytest_commands()
    addopts = [
        tomllib.loads(path.read_text())["tool"]["pytest"]["ini_options"]["addopts"]
        for path in PYPROJECTS
    ]
    workflow_text = "\n".join(path.read_text() for path in WORKFLOWS)

    assert len(commands) >= 4
    assert len(addopts) == 3
    for options in [*commands, *addopts]:
        assert "--showlocals" not in options
        assert not re.search(r"(^|\s)-l(\s|$)", options)
    assert "PYTEST_ADDOPTS" not in workflow_text
