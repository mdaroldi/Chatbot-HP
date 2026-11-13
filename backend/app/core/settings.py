"""Application settings, loaded only from environment variables."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote, urlsplit

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.domain.ingestion import BM25_DEFAULT_AVG_LEN
from app.domain.retrieval import DEFAULT_RRF_K, SearchMode

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

    qdrant_collection_alias: str = Field(
        default="hpdocs_chunks",
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Alias that always names the live chunk collection.",
    )
    index_keep_previous: bool = Field(
        default=True,
        description="Keep the previous build's collection after a swap, for rollback; "
        "older builds are always deleted.",
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
    embedding_bm25_avg_len: (
        Annotated[float, Field(gt=0, le=10_000, allow_inf_nan=False)] | Literal["auto"]
    ) = Field(
        default=BM25_DEFAULT_AVG_LEN,
        description="BM25 average document length for the stored sparse vectors, or "
        "`auto` for the mean of the chunks being built; recorded per build (ADR 0009).",
    )

    chunk_max_tokens: int = Field(
        default=450,
        ge=32,
        le=512,
        description="Largest chunk in dense-model tokens, prefix included; bge-small reads "
        "at most 512 (ADR 0005).",
    )
    chunk_min_tokens: int = Field(
        default=65,
        ge=0,
        le=511,
        description="Sections below this many tokens merge with the next one under the "
        "same parent (ADR 0005).",
    )

    retrieval_mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="`dense`, `sparse` or `hybrid` (dense and BM25 fused with RRF).",
    )
    retrieval_rerank_enabled: bool = Field(
        default=True, description="Rerank the candidates with the cross-encoder."
    )
    retrieval_top_k: int = Field(
        default=5, ge=1, le=20, description="Chunks returned per question."
    )
    retrieval_rerank_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Candidates the reranker scores; at least `RETRIEVAL_TOP_K`.",
    )
    retrieval_prefetch_dense_limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Dense candidates before fusion; at least `RETRIEVAL_RERANK_CANDIDATES`.",
    )
    retrieval_prefetch_sparse_limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="BM25 candidates before fusion; at least `RETRIEVAL_RERANK_CANDIDATES`.",
    )
    retrieval_rrf_k: int = Field(
        default=DEFAULT_RRF_K,
        ge=1,
        le=1000,
        description="Reciprocal rank fusion constant k (Qdrant's default is 2).",
    )
    retrieval_product_filter_enabled: bool = Field(
        default=False,
        description="Restrict the search to one manual when the question names its product.",
    )
    retrieval_relevance_threshold: float | None = Field(
        default=-6.54,
        ge=-100,
        le=100,
        allow_inf_nan=False,
        description="Lowest top reranker score that counts as relevant. Calibrated for "
        "Xenova/ms-marco-MiniLM-L-6-v2 @ a09144355ade on the dev split (ADR 0009): 2 of "
        "48 in-scope questions refused, 4 of 4 unrelated and 0 of 4 near-domain "
        "out-of-scope questions caught. Fragile (8 out-of-scope questions); recalibrate "
        "when the reranker changes.",
    )
    retrieval_dense_query_prefix: str = Field(
        default="",
        max_length=200,
        description="Text put before the question for the dense query only (BGE instruction).",
    )
    reranker_model: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        min_length=1,
        description="fastembed cross-encoder; must be pinned in `models.toml`.",
    )
    inference_workers: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Threads that run embedding and reranking for requests.",
    )
    inference_onnx_threads: int = Field(
        default=2,
        ge=1,
        le=32,
        description="ONNX intra-op threads per model call; workers times threads should "
        "fit the container's CPUs.",
    )
    qdrant_search_timeout_seconds: float = Field(
        default=2.0, gt=0, le=30, description="Deadline for one Qdrant search."
    )
    query_max_chars: int = Field(
        default=1000, ge=1, le=4000, description="Longest accepted question, in characters."
    )
    rate_limit_conversations_per_minute: int = Field(
        default=120,
        ge=1,
        le=100_000,
        description="Conversation create, list, read and delete calls per client ID per minute.",
    )
    rate_limit_conversations_burst: int = Field(
        default=30,
        ge=1,
        le=10_000,
        description="Conversation calls a client may make at once (opening the UI loads several).",
    )
    rate_limit_search_per_minute: int = Field(
        default=30,
        ge=1,
        le=100_000,
        description="`POST /v1/search` calls per client ID (else per IP) per minute.",
    )
    rate_limit_search_burst: int = Field(
        default=10, ge=1, le=10_000, description="Search calls allowed at once."
    )
    rate_limit_max_keys: int = Field(
        default=10_000,
        ge=1,
        le=1_000_000,
        description="Client IDs or IPs tracked per limiter and replica; the least recent "
        "is forgotten.",
    )
    search_endpoint_enabled: bool = Field(
        default=False,
        description="Serve the `POST /v1/search` debug endpoint (returns chunk text); a "
        "warning is logged when enabled in production.",
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

    @field_validator("retrieval_relevance_threshold", mode="before")
    @classmethod
    def _blank_threshold_is_unset(cls, value: object) -> object:
        """An empty value unsets the threshold, e.g. for a reranker not yet calibrated."""
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _no_exception_messages_in_production(self) -> "Settings":
        if self.app_env is Environment.PRODUCTION and self.log_include_exception_messages:
            raise ValueError("log_include_exception_messages=true is not allowed in production")
        return self

    @model_validator(mode="after")
    def _merge_floor_below_the_limit(self) -> "Settings":
        if self.chunk_min_tokens >= self.chunk_max_tokens:
            raise ValueError("chunk_min_tokens must be below chunk_max_tokens")
        return self

    @model_validator(mode="after")
    def _retrieval_limits_nest(self) -> "Settings":
        if self.retrieval_rerank_candidates < self.retrieval_top_k:
            raise ValueError("retrieval_rerank_candidates must be at least retrieval_top_k")
        prefetch = min(self.retrieval_prefetch_dense_limit, self.retrieval_prefetch_sparse_limit)
        if prefetch < self.retrieval_rerank_candidates:
            raise ValueError("prefetch limits must be at least retrieval_rerank_candidates")
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
