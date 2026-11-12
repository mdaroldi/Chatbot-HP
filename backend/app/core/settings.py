"""Application settings, loaded only from environment variables."""

from enum import StrEnum
from functools import lru_cache
from typing import Annotated
from urllib.parse import quote, urlsplit

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_ALLOWED_ORIGIN_SCHEMES = frozenset({"http", "https"})


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def _normalize_origin(origin: str) -> str:
    """Reduce an origin to scheme://host[:port], the form browsers send in `Origin`.

    A trailing slash is dropped; anything else beyond the host is a configuration
    mistake, because CORS matching never sees paths, queries or fragments.
    """
    if origin == "*":
        raise ValueError("wildcard origin '*' is not allowed")
    parts = urlsplit(origin)
    if parts.scheme not in _ALLOWED_ORIGIN_SCHEMES or not parts.netloc:
        raise ValueError(f"origin {origin!r} must be an http(s) URL")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError(f"origin {origin!r} must contain only scheme and host")
    return f"{parts.scheme}://{parts.netloc}"


class Settings(BaseSettings):
    """Typed configuration with safe defaults for everything except secrets.

    The database password has no default on purpose: a missing secret must fail
    at startup instead of silently falling back to a well-known value.
    """

    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, frozen=True)

    app_env: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    cors_allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:8501"]
    # Interactive docs and the OpenAPI schema; disable to shrink the public surface.
    api_docs_enabled: bool = True

    postgres_host: str = Field(default="postgres", min_length=1)
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = Field(default="hpdocs", min_length=1)
    postgres_user: str = Field(default="hpdocs", min_length=1)
    postgres_password: SecretStr = Field(min_length=1)

    qdrant_url: HttpUrl = HttpUrl("http://qdrant:6333")

    readiness_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> list[str]:
        """Accept a comma-separated string (env var form) or a list."""
        raw = value.split(",") if isinstance(value, str) else value
        if not isinstance(raw, list):
            # ValueError, not TypeError: pydantic only wraps ValueError into ValidationError.
            raise ValueError("cors_allowed_origins must be a string or a list")  # noqa: TRY004
        origins = [str(item).strip() for item in raw]
        if any(not origin for origin in origins):
            raise ValueError("cors_allowed_origins must not contain empty entries")
        return [_normalize_origin(origin) for origin in origins]

    @property
    def postgres_dsn(self) -> SecretStr:
        """Async SQLAlchemy DSN, wrapped so it can never be logged by accident."""
        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password.get_secret_value(), safe="")
        return SecretStr(
            f"postgresql+asyncpg://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Single settings instance per process, usable as a FastAPI dependency."""
    return Settings()
