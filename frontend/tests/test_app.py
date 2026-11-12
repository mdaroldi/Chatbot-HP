"""The real Streamlit script, run with AppTest and a faked fetch_status."""

import pathlib
from collections.abc import Callable

import pytest
from streamlit.testing.v1 import AppTest

from hpdocs_ui import status
from hpdocs_ui.status import BackendStatus

APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"
UP: dict[str, BackendStatus] = {
    "/health": BackendStatus(True, 200, {"status": "ok"}, "rid-12345678"),
    "/ready": BackendStatus(
        True, 200, {"status": "ready", "checks": {"postgres": {"status": "ok"}}}, "rid-12345678"
    ),
}
DOWN = BackendStatus(reachable=False, http_status=None, body=None, request_id=None)


class FakeFetch:
    def __init__(self, responses: Callable[[str], BackendStatus]) -> None:
        self.calls: list[tuple[str, str]] = []
        self.responses = responses

    def __call__(self, base_url: str, path: str) -> BackendStatus:
        self.calls.append((base_url, path))
        return self.responses(path)


@pytest.fixture
def run_app(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeFetch], AppTest]:
    monkeypatch.setenv("BACKEND_URL", "http://backend.test:8000")

    def run(fake: FakeFetch) -> AppTest:
        monkeypatch.setattr(status, "fetch_status", fake)
        app = AppTest.from_file(str(APP), default_timeout=10)
        return app.run()

    return run


def rendered_text(app: AppTest) -> str:
    parts = [e.value for e in (*app.success, *app.warning, *app.error, *app.text)]
    return "\n".join(str(part) for part in parts)


def test_backend_up_shows_success_and_dependency_status(
    run_app: Callable[[FakeFetch], AppTest],
) -> None:
    fake = FakeFetch(UP.__getitem__)

    app = run_app(fake)

    assert not app.exception
    assert [e.value for e in app.success] == ["Backend is up and ready."]
    assert "postgres: ok" in rendered_text(app)
    assert fake.calls == [
        ("http://backend.test:8000/", "/health"),
        ("http://backend.test:8000/", "/ready"),
    ]


def test_backend_down_shows_error_without_internals(
    run_app: Callable[[FakeFetch], AppTest],
) -> None:
    app = run_app(FakeFetch(lambda _: DOWN))

    assert not app.exception
    assert [e.value for e in app.error] == ["Backend is unreachable."]
    assert not app.success
    assert "Traceback" not in rendered_text(app)


def test_refresh_button_fetches_status_again(run_app: Callable[[FakeFetch], AppTest]) -> None:
    fake = FakeFetch(UP.__getitem__)
    app = run_app(fake)

    app.button[0].click().run()

    assert [e.label for e in app.button] == ["Refresh status"]
    assert len(fake.calls) == 4
