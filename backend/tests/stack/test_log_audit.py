"""Stack log audit: a chat session full of canaries, then every container's logs.

Run with `make test-log-audit` against the running stub stack (README, testing). Each
user-controlled input carries its own canary: questions, a follow-up that goes through
the rewrite, the stub's refusal and failure directives, rejected inputs (too long,
malformed JSON, invalid client ID, cursor, path) and the User-Agent and Referer headers.
The audit then reads `docker compose logs` of every service and fails if any canary,
or the session's client ID, appears in any form. Failures print the summary only
(container, canary id, line count), never a log line: CI logs are public.

The control test plants a token in the backend's stdout through the same channel and
checks the audit reports it, so a scan that reads nothing cannot pass.
"""

import os
import secrets
import subprocess  # nosec B404 - fixed docker compose commands, no shell
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from app.audit.log_scan import Canary, Leak, find_leaks, new_canary, summary

pytestmark = pytest.mark.stack

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES = ("backend", "frontend", "ingest", "migrate", "postgres", "qdrant")
# Postgres must never write a statement's text or bound parameters for these paths.
POSTGRES_LOG_SETTINGS = {
    "log_statement": "none",
    "log_min_duration_statement": "-1",
    "log_min_duration_sample": "-1",
    "log_duration": "off",
    "log_parameter_max_length_on_error": "0",
}
CANARY_IDS = (
    "question",
    "followup",
    "refuse",
    "failbefore",
    "failafter",
    "toolong",
    "badjson",
    "badclient",
    "cursor",
    "path",
    "useragent",
    "referer",
)
TIMEOUT = httpx.Timeout(60.0, connect=5.0)
# In scope for both manuals' index, so the probe passes the relevance gate and reaches
# the (stub) model; a canary alone would be refused before it.
IN_SCOPE = "How do I clean the printhead on the HP ENVY 6000 printer?"


@dataclass(frozen=True)
class Session:
    statuses: dict[str, int]
    events: dict[str, set[str]]


def _events(body: str) -> set[str]:
    """Event names of an SSE body; the data lines are dropped unread."""
    return {
        line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event: ")
    }


def _api_url() -> str:
    return os.environ.get("LOG_AUDIT_API_URL", "http://127.0.0.1:8000")


def _compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 # nosec B603 B607 - fixed argv, docker from PATH
        ["docker", "compose", *args],  # noqa: S607 - docker from PATH, like make
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )


def _logs() -> dict[str, str]:
    return {
        service: _compose("logs", "--no-color", "--no-log-prefix", service).stdout
        for service in SERVICES
    }


@pytest.fixture(scope="module")
def nonce() -> str:
    return secrets.token_hex(4)


@pytest.fixture(scope="module")
def canaries(nonce: str) -> dict[str, Canary]:
    return {canary_id: new_canary(canary_id, nonce) for canary_id in CANARY_IDS}


@pytest.fixture(scope="module")
def client_ids() -> tuple[str, str]:
    """Two clients, so the session stays inside the messages burst of 5 per client."""
    return str(uuid.uuid4()), str(uuid.uuid4())


@pytest.fixture(scope="module")
def stack() -> str:
    url = _api_url()
    try:
        response = httpx.get(f"{url}/ready", timeout=5)
        body = response.json()
    except (httpx.HTTPError, ValueError) as error:
        pytest.fail(f"backend not reachable at {url} ({type(error).__name__})", pytrace=False)
    if not isinstance(body, dict) or body.get("llm") != "stub":
        pytest.exit("ABORTED: /ready does not report the stub; no question was sent", 2)
    if response.status_code != 200 or body.get("status") != "ready":
        pytest.fail(f"backend at {url} is not ready (HTTP {response.status_code})", pytrace=False)
    return url


def _ask(client: httpx.Client, conversation_id: str, content: str) -> tuple[int, str]:
    """Send one question and read the whole answer (stream or JSON error)."""
    with client.stream(
        "POST", f"/v1/conversations/{conversation_id}/messages", json={"content": content}
    ) as response:
        return response.status_code, response.read().decode()


def _new_conversation(client: httpx.Client) -> str:
    response = client.post("/v1/conversations")
    assert response.status_code == 201
    conversation_id = response.json()["id"]
    assert isinstance(conversation_id, str)
    return conversation_id


@pytest.fixture(scope="module")
def session(stack: str, canaries: dict[str, Canary], client_ids: tuple[str, str]) -> "Session":
    """Drive every path once; return each probe's HTTP status for the sanity checks."""
    headers = {
        "User-Agent": f"audit-agent {canaries['useragent'].token}",
        "Referer": f"http://audit.invalid/{canaries['referer'].token}?q=1",
    }
    statuses: dict[str, int] = {}
    events: dict[str, set[str]] = {}
    first, second = client_ids
    with httpx.Client(base_url=stack, timeout=TIMEOUT, headers=headers) as client:
        client.headers["X-Client-ID"] = first
        conversation = _new_conversation(client)
        statuses["question"], body = _ask(
            client, conversation, f"{IN_SCOPE} {canaries['question'].token}"
        )
        events["question"] = _events(body)
        statuses["followup"], body = _ask(
            client, conversation, f"And what about {canaries['followup'].token} after that?"
        )
        events["followup"] = _events(body)
        statuses["refuse"], _ = _ask(
            client, conversation, f"[stub:refuse] {IN_SCOPE} {canaries['refuse'].token}"
        )
        statuses["toolong"], _ = _ask(
            client, conversation, f"{canaries['toolong'].token} " + "x" * 1100
        )
        statuses["badjson"] = client.post(
            f"/v1/conversations/{conversation}/messages",
            content=f'{{"content": "{canaries["badjson"].token}"',
            headers={"Content-Type": "application/json"},
        ).status_code
        statuses["cursor"] = client.get(
            "/v1/conversations", params={"cursor": canaries["cursor"].token}
        ).status_code
        statuses["path"] = client.get(f"/v1/conversations/{canaries['path'].token}").status_code

        client.headers["X-Client-ID"] = second
        statuses["failbefore"], _ = _ask(
            client,
            _new_conversation(client),
            f"[stub:fail-before] {IN_SCOPE} {canaries['failbefore'].token}",
        )
        statuses["failafter"], body = _ask(
            client,
            _new_conversation(client),
            f"[stub:fail-after] {IN_SCOPE} {canaries['failafter'].token}",
        )
        events["failafter"] = _events(body)

        client.headers["X-Client-ID"] = f"{canaries['badclient'].token} !"
        statuses["badclient"] = client.post("/v1/conversations").status_code
    return Session(statuses=statuses, events=events)


def test_every_probe_reached_its_path(session: "Session") -> None:
    """The audit is only as good as the paths it drove: each probe got the expected status.

    Assertions compare statuses and event names only, never response bodies, which hold
    manual excerpts and must not reach CI logs.
    """
    assert session.events["question"] == {"token", "sources", "done"}
    assert session.events["followup"] == {"token", "sources", "done"}
    assert session.events["failafter"] == {"token", "error"}
    statuses = session.statuses
    assert statuses["question"] == 200
    assert statuses["followup"] == 200
    assert statuses["refuse"] == 200
    assert statuses["failafter"] == 200
    assert statuses["failbefore"] >= 500
    assert statuses["toolong"] == 422
    assert statuses["badjson"] == 422
    assert statuses["badclient"] == 400
    assert statuses["cursor"] in {400, 422}
    assert statuses["path"] in {404, 422}


@pytest.mark.usefixtures("stack")
def test_postgres_never_logs_statements_or_parameters() -> None:
    for name, expected in POSTGRES_LOG_SETTINGS.items():
        value = _compose(
            "exec",
            "-T",
            "postgres",
            "sh",
            "-c",
            f'psql -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SHOW {name}"',
        ).stdout.strip()
        assert (name, value) == (name, expected)


@pytest.mark.usefixtures("session")
def test_no_canary_or_client_id_reaches_any_container_log(
    canaries: dict[str, Canary], client_ids: tuple[str, str]
) -> None:
    watched = [
        *canaries.values(),
        *(Canary(id=f"clientid{index}", token=value) for index, value in enumerate(client_ids)),
    ]

    leaks = find_leaks(_logs(), watched)

    assert not leaks, summary(leaks)


@pytest.mark.usefixtures("session")
def test_control_a_planted_line_in_the_backend_stdout_is_reported(nonce: str) -> None:
    control = new_canary("control", nonce)
    _compose(
        "exec",
        "-T",
        "backend",
        "sh",
        "-c",
        f"printf '%s\\n' 'planted {control.token}' > /proc/1/fd/1",
    )

    leaks = find_leaks(_logs(), [control])

    assert leaks == [Leak(container="backend", canary_id="control", lines=1)]
