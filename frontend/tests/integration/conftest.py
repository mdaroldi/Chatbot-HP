"""The real page against the running compose stack, with the stub language model.

Run with `make test-ui-int` after starting the stub stack (README, testing). The
backend must be up: these tests fail, never skip, when it is not. They abort before
sending anything unless /ready names the stub, so they never call a paid model.
"""

import os
import pathlib
import uuid
from collections.abc import Callable

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from hpdocs_ui.http_api import HttpChatApi
from hpdocs_ui.settings import FrontendSettings

APP = pathlib.Path(__file__).resolve().parents[2] / "app.py"
# Generous: the stub streams token by token with the stack's configured delay.
PAGE_TIMEOUT_SECONDS = 120

OpenPage = Callable[..., AppTest]


def backend_url() -> str:
    default = f"http://127.0.0.1:{os.environ.get('API_PORT', '8000')}"
    return os.environ.get("UI_IT_BACKEND_URL") or default


@pytest.fixture(scope="session", autouse=True)
def stack() -> str:
    """The backend, checked once: unreachable or not ready fails every test; any model
    but the stub stops the run before a single question is sent.
    """
    url = backend_url()
    try:
        response = httpx.get(f"{url}/ready", timeout=5)
        body = response.json()
    except (httpx.HTTPError, ValueError) as error:
        pytest.fail(
            f"backend not reachable at {url} ({type(error).__name__}); "
            "start the stub stack first (README, testing)",
            pytrace=False,
        )
    llm = body.get("llm") if isinstance(body, dict) else None
    if llm != "stub":
        pytest.exit(
            f"ABORTED: /ready at {url} reports llm={llm!r}, not 'stub'. These tests send "
            "questions; restart the backend with LLM_PROVIDER=stub.",
            returncode=2,
        )
    if response.status_code != 200 or body.get("status") != "ready":
        pytest.fail(
            f"backend at {url} is not ready (HTTP {response.status_code}): {body}",
            pytrace=False,
        )
    return url


@pytest.fixture(autouse=True)
def wired(stack: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page and the client read the backend address as in compose."""
    monkeypatch.setenv("BACKEND_URL", stack)


@pytest.fixture
def client_id() -> str:
    """One per scenario: the backend rate-limits each client ID (burst of 5)."""
    return str(uuid.uuid4())


@pytest.fixture
def api(client_id: str) -> HttpChatApi:
    return HttpChatApi(FrontendSettings(), client_id)


@pytest.fixture
def open_page() -> OpenPage:
    def open_(client_id: str, conversation_id: str | None = None) -> AppTest:
        app = AppTest.from_file(str(APP), default_timeout=PAGE_TIMEOUT_SECONDS)
        app.query_params["cid"] = client_id
        if conversation_id is not None:
            app.query_params["conv"] = conversation_id
        app.run()
        assert not app.exception
        return app

    return open_
