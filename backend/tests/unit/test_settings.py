"""Behavior of the environment-driven application settings."""

import pytest
from pydantic import SecretStr, ValidationError

from app.core.settings import Environment, LogLevel, Settings, get_settings


def test_safe_defaults_when_only_required_values_are_set() -> None:
    settings = Settings()

    assert settings.app_env is Environment.DEVELOPMENT
    assert settings.log_level is LogLevel.INFO
    assert settings.cors_allowed_origins == ["http://localhost:8501"]
    assert settings.postgres_host == "postgres"
    assert settings.postgres_port == 5432
    assert settings.postgres_db == "hpdocs"
    assert settings.postgres_user == "hpdocs"
    assert str(settings.qdrant_url) == "http://qdrant:6333/"
    assert settings.readiness_timeout_seconds == 2.0
    assert settings.api_docs_enabled is True


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("0", False), ("true", True)])
def test_api_docs_can_be_toggled_from_environment(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("API_DOCS_ENABLED", raw)

    assert Settings().api_docs_enabled is expected


def test_exception_messages_are_excluded_from_logs_by_default() -> None:
    assert Settings().log_include_exception_messages is False


@pytest.mark.parametrize("env", ["development", "test"])
def test_exception_messages_can_be_enabled_outside_production(
    monkeypatch: pytest.MonkeyPatch, env: str
) -> None:
    monkeypatch.setenv("APP_ENV", env)
    monkeypatch.setenv("LOG_INCLUDE_EXCEPTION_MESSAGES", "true")

    assert Settings().log_include_exception_messages is True


def test_exception_messages_in_logs_are_rejected_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_INCLUDE_EXCEPTION_MESSAGES", "true")

    with pytest.raises(ValidationError, match="not allowed in production"):
        Settings()


def test_invalid_api_docs_flag_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_DOCS_ENABLED", "maybe")

    with pytest.raises(ValidationError, match="api_docs_enabled"):
        Settings()


def test_values_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("QDRANT_URL", "http://vectors.internal:6333")
    monkeypatch.setenv("READINESS_TIMEOUT_SECONDS", "0.5")

    settings = Settings()

    assert settings.app_env is Environment.PRODUCTION
    assert settings.log_level is LogLevel.WARNING
    assert settings.postgres_host == "db.internal"
    assert settings.postgres_port == 6543
    assert str(settings.qdrant_url) == "http://vectors.internal:6333/"
    assert settings.readiness_timeout_seconds == 0.5


def test_postgres_password_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_PASSWORD")

    with pytest.raises(ValidationError, match="postgres_password"):
        Settings()


def test_postgres_password_is_secret_and_hidden_from_repr() -> None:
    settings = Settings()

    assert isinstance(settings.postgres_password, SecretStr)
    assert settings.postgres_password.get_secret_value() == "test-password"
    assert "test-password" not in repr(settings)
    assert "test-password" not in str(settings.model_dump())
    assert "test-password" not in str(settings.model_dump(mode="json"))
    assert "test-password" not in settings.model_dump_json()


def test_postgres_dsn_is_secret_and_hidden_from_repr() -> None:
    settings = Settings()

    assert isinstance(settings.postgres_dsn, SecretStr)
    assert "test-password" not in repr(settings.postgres_dsn)
    assert "test-password" not in str(settings.postgres_dsn)


def test_postgres_dsn_targets_asyncpg_with_configured_values() -> None:
    settings = Settings()

    assert settings.postgres_dsn.get_secret_value() == (
        "postgresql+asyncpg://hpdocs:test-password@postgres:5432/hpdocs"
    )


def test_postgres_dsn_escapes_special_characters_in_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss:w/rd")

    dsn = Settings().postgres_dsn.get_secret_value()

    assert dsn == "postgresql+asyncpg://hpdocs:p%40ss%3Aw%2Frd@postgres:5432/hpdocs"


def test_empty_postgres_password_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "")

    with pytest.raises(ValidationError, match="postgres_password"):
        Settings()


def test_cors_origins_are_split_on_commas_and_whitespace_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS", "  http://localhost:8501 ,https://chat.example.com  "
    )

    settings = Settings()

    assert settings.cors_allowed_origins == ["http://localhost:8501", "https://chat.example.com"]


def test_cors_origin_trailing_slash_is_normalized_away(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:8501/,https://chat.example.com/")

    settings = Settings()

    assert settings.cors_allowed_origins == ["http://localhost:8501", "https://chat.example.com"]


def test_cors_origins_given_as_list_are_normalized() -> None:
    settings = Settings(cors_allowed_origins=[" https://chat.example.com/ "])

    assert settings.cors_allowed_origins == ["https://chat.example.com"]


def test_cors_origins_of_unsupported_type_are_rejected() -> None:
    with pytest.raises(ValidationError, match="string or a list"):
        Settings.model_validate({"cors_allowed_origins": 42})


@pytest.mark.parametrize(
    "origins", ["", "   ", "http://localhost:8501,", "http://localhost:8501,,https://a.example"]
)
def test_cors_empty_entries_are_rejected(monkeypatch: pytest.MonkeyPatch, origins: str) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", origins)

    with pytest.raises(ValidationError, match="empty"):
        Settings()


@pytest.mark.parametrize(
    "origin",
    ["https://chat.example.com/app", "https://chat.example.com?x=1", "https://chat.example.com#x"],
)
def test_cors_origin_with_path_query_or_fragment_is_rejected(
    monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", origin)

    with pytest.raises(ValidationError, match="scheme and host"):
        Settings()


@pytest.mark.parametrize("origins", ["*", "http://localhost:8501,*"])
def test_wildcard_cors_origin_is_rejected(monkeypatch: pytest.MonkeyPatch, origins: str) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", origins)

    with pytest.raises(ValidationError, match="wildcard"):
        Settings()


def test_cors_origin_must_be_http_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "localhost:8501")

    with pytest.raises(ValidationError, match="cors_allowed_origins"):
        Settings()


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError, match="log_level"):
        Settings()


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port"])
def test_invalid_postgres_port_is_rejected(monkeypatch: pytest.MonkeyPatch, port: str) -> None:
    monkeypatch.setenv("POSTGRES_PORT", port)

    with pytest.raises(ValidationError, match="postgres_port"):
        Settings()


def test_readiness_timeout_upper_bound_of_30_seconds_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("READINESS_TIMEOUT_SECONDS", "30")

    assert Settings().readiness_timeout_seconds == 30.0


@pytest.mark.parametrize("timeout", ["0", "-1", "30.1"])
def test_readiness_timeout_outside_bounds_is_rejected(
    monkeypatch: pytest.MonkeyPatch, timeout: str
) -> None:
    monkeypatch.setenv("READINESS_TIMEOUT_SECONDS", timeout)

    with pytest.raises(ValidationError, match="readiness_timeout_seconds"):
        Settings()


def test_invalid_qdrant_url_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "not a url")

    with pytest.raises(ValidationError, match="qdrant_url"):
        Settings()


def test_get_settings_returns_a_cached_instance() -> None:
    assert get_settings() is get_settings()
