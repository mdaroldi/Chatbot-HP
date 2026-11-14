"""Static guarantees for the Streamlit app: privacy, safety, no stale status."""

import pathlib
import tomllib
from collections.abc import Iterator

import pytest
from streamlit import config as streamlit_config
from streamlit.proto.NewSession_pb2 import Config
from streamlit.runtime.app_session import _get_toolbar_mode
from streamlit.testing.v1 import AppTest

FRONTEND = pathlib.Path(__file__).resolve().parents[1]
CONFIG = FRONTEND / ".streamlit" / "config.toml"
SOURCES = [FRONTEND / "app.py", *sorted((FRONTEND / "hpdocs_ui").glob("*.py"))]


def load_config() -> dict[str, dict[str, object]]:
    assert CONFIG.is_file(), f"missing {CONFIG}"
    return tomllib.loads(CONFIG.read_text())


def test_usage_stats_are_off_and_server_is_headless() -> None:
    config = load_config()

    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["headless"] is True


@pytest.fixture
def loaded_by_streamlit(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The project config as Streamlit itself loads it ($CWD/.streamlit/config.toml),
    so a value the pinned Streamlit does not know shows up as a failure here.
    """
    monkeypatch.chdir(FRONTEND)
    streamlit_config.get_config_options(force_reparse=True)
    yield
    monkeypatch.undo()
    streamlit_config.get_config_options(force_reparse=True)


def test_toolbar_hides_deploy_and_the_developer_menu() -> None:
    assert load_config()["client"]["toolbarMode"] == "minimal"


@pytest.mark.usefixtures("loaded_by_streamlit")
def test_pinned_streamlit_accepts_the_minimal_toolbar() -> None:
    # Streamlit's own mapping from the option to the browser message; it raises on
    # a value this version does not know.
    assert _get_toolbar_mode() == Config.ToolbarMode.MINIMAL


def test_error_details_are_hidden_from_the_browser() -> None:
    assert load_config()["client"]["showErrorDetails"] == "none"


@pytest.mark.usefixtures("loaded_by_streamlit")
def test_an_uncaught_error_shows_no_message_type_or_traceback_in_the_page() -> None:
    app = AppTest.from_string("raise RuntimeError('detail-canary')").run()

    shown = app.exception[0]
    assert "detail-canary" not in shown.message
    assert shown.stack_trace == []
    assert "RuntimeError" not in shown.type


@pytest.mark.usefixtures("loaded_by_streamlit")
def test_pinned_streamlit_loads_usage_stats_off() -> None:
    assert streamlit_config.get_option("browser.gatherUsageStats") is False


def test_xsrf_and_cors_protection_keep_their_secure_defaults() -> None:
    server = load_config()["server"]

    assert "enableXsrfProtection" not in server
    assert "enableCORS" not in server


def test_sources_never_render_raw_html() -> None:
    assert len(SOURCES) >= 3
    for source in SOURCES:
        assert "unsafe_allow_html" not in source.read_text(), source.name


def test_status_results_are_never_cached() -> None:
    for source in SOURCES:
        text = source.read_text()
        assert "cache_data" not in text, source.name
        assert "cache_resource" not in text, source.name
