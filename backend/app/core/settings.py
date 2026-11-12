"""Application settings, loaded only from environment variables."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import quote, urlsplit

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
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

    app_env: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="`development`, `test` or `production`.",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO, description="`DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`."
    )
    log_include_exception_messages: bool = Field(
        default=False,
        description="Log exception messages (they may contain user text). "
        "Local debugging only; rejected in production.",
    )
    cors_allowed_origins: Annotated[list[str], NoDecode] = Field(
        default=["http://localhost:8501"],
        description="Comma-separated browser origins allowed by CORS; `*` is rejected.",
    )
    api_docs_enabled: bool = Field(
        default=True, description="Serve `/docs`, `/redoc` and `/openapi.json`."
    )

    postgres_host: str = Field(
        default="postgres", min_length=1, description="Postgres host (fixed in Compose)."
    )
    postgres_port: int = Field(
        default=5432, ge=1, le=65535, description="Postgres port (fixed in Compose)."
    )
    postgres_db: str = Field(default="hpdocs", min_length=1, description="Database name.")
    postgres_user: str = Field(default="hpdocs", min_length=1, description="Database user.")
    postgres_password: SecretStr = Field(
        min_length=1, description="Database password. Required; never has a default."
    )
    postgres_pool_size: int = Field(
        default=5, ge=1, le=100, description="Pooled connections kept per API process."
    )
    postgres_max_overflow: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Extra connections beyond the pool; the cap is pool size plus this.",
    )
    postgres_connect_timeout_seconds: float = Field(
        default=5.0, gt=0, le=60, description="Timeout for opening a Postgres connection."
    )

    qdrant_url: HttpUrl = Field(
        default=HttpUrl("http://qdrant:6333"),
        description="Qdrant REST base URL (fixed in Compose).",
    )

    readiness_timeout_seconds: float = Field(
        default=2.0, gt=0, le=30, description="Per-dependency timeout for `GET /ready`."
    )

    corpus_dir: Path = Field(
        default=Path("corpus"),
        description="Cache directory for the verified source PDFs; reused offline.",
    )
    corpus_max_bytes: int = Field(
        default=32 * 1024 * 1024,
        gt=0,
        le=1024 * 1024 * 1024,
        description="Largest accepted source PDF, in bytes.",
    )
    corpus_download_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=3600,
        description="Per-phase HTTP timeout and total deadline for one PDF download.",
    )

    embedding_dense_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        min_length=1,
        description="fastembed dense model; its tokenizer also measures chunk size.",
    )
    embedding_sparse_model: str = Field(
        default="Qdrant/bm25", min_length=1, description="fastembed sparse model."
    )
    embedding_cache_dir: Path = Field(
        default=Path("models"),
        description="Cache directory for the downloaded embedding models; reused offline.",
    )
    embedding_batch_size: int = Field(
        default=32, ge=1, le=1024, description="Texts per embedding batch (bounds memory)."
    )

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

    @model_validator(mode="after")
    def _no_exception_messages_in_production(self) -> "Settings":
        if self.app_env is Environment.PRODUCTION and self.log_include_exception_messages:
            raise ValueError("log_include_exception_messages=true is not allowed in production")
        return self

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
