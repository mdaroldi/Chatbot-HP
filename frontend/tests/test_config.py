"""Static guarantees for the Streamlit app: privacy, safety, no stale status."""

import pathlib
import tomllib

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
