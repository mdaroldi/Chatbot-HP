"""Behavior of the environment-driven application settings."""

import pathlib

import pytest
from pydantic import SecretStr, ValidationError

from app.core.settings import Environment, LogLevel, Settings, get_settings
from app.domain.retrieval import SearchMode


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
    assert settings.postgres_pool_size == 5
    assert settings.postgres_max_overflow == 5
    assert settings.postgres_connect_timeout_seconds == 5.0


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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("POSTGRES_POOL_SIZE", "0"),
        ("POSTGRES_POOL_SIZE", "101"),
        ("POSTGRES_MAX_OVERFLOW", "-1"),
        ("POSTGRES_MAX_OVERFLOW", "101"),
        ("POSTGRES_CONNECT_TIMEOUT_SECONDS", "0"),
        ("POSTGRES_CONNECT_TIMEOUT_SECONDS", "60.1"),
    ],
)
def test_postgres_pool_settings_are_bounded(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=name.lower()):
        Settings()


def test_invalid_qdrant_url_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "not a url")

    with pytest.raises(ValidationError, match="qdrant_url"):
        Settings()


def test_get_settings_returns_a_cached_instance() -> None:
    assert get_settings() is get_settings()


def test_corpus_defaults_fit_both_manuals_and_a_slow_link() -> None:
    settings = Settings()

    assert settings.corpus_dir == pathlib.Path("corpus")
    assert settings.corpus_max_bytes == 32 * 1024 * 1024
    assert settings.corpus_download_timeout_seconds == 120.0


def test_corpus_settings_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_DIR", "/var/cache/corpus")
    monkeypatch.setenv("CORPUS_MAX_BYTES", "1048576")
    monkeypatch.setenv("CORPUS_DOWNLOAD_TIMEOUT_SECONDS", "30")

    settings = Settings()

    assert settings.corpus_dir == pathlib.Path("/var/cache/corpus")
    assert settings.corpus_max_bytes == 1048576
    assert settings.corpus_download_timeout_seconds == 30.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CORPUS_MAX_BYTES", "0"),
        ("CORPUS_MAX_BYTES", "1073741825"),
        ("CORPUS_DOWNLOAD_TIMEOUT_SECONDS", "0"),
        ("CORPUS_DOWNLOAD_TIMEOUT_SECONDS", "3601"),
    ],
)
def test_corpus_limits_are_bounded(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=name.lower()):
        Settings()


def test_embedding_defaults_are_the_approved_local_models() -> None:
    settings = Settings()

    assert settings.embedding_dense_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_sparse_model == "Qdrant/bm25"
    assert settings.embedding_cache_dir == pathlib.Path("models")
    assert settings.embedding_batch_size == 32
    assert settings.embedding_bm25_avg_len == 256.0  # fastembed's own: the index stays as built


@pytest.mark.parametrize(("value", "expected"), [("auto", "auto"), ("107.5", 107.5), ("64", 64.0)])
def test_the_bm25_avg_len_is_a_number_or_auto(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: float | str
) -> None:
    monkeypatch.setenv("EMBEDDING_BM25_AVG_LEN", value)

    assert Settings().embedding_bm25_avg_len == expected


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "10001", "AUTO", "mean", ""])
def test_the_bm25_avg_len_rejects_anything_else(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EMBEDDING_BM25_AVG_LEN", value)

    with pytest.raises(ValidationError, match="embedding_bm25_avg_len"):
        Settings()


def test_embedding_settings_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DENSE_MODEL", "BAAI/bge-base-en-v1.5")
    monkeypatch.setenv("EMBEDDING_SPARSE_MODEL", "prithivida/Splade_PP_en_v1")
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", "/var/cache/models")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "8")

    settings = Settings()

    assert settings.embedding_dense_model == "BAAI/bge-base-en-v1.5"
    assert settings.embedding_sparse_model == "prithivida/Splade_PP_en_v1"
    assert settings.embedding_cache_dir == pathlib.Path("/var/cache/models")
    assert settings.embedding_batch_size == 8


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EMBEDDING_DENSE_MODEL", ""),
        ("EMBEDDING_SPARSE_MODEL", ""),
        ("EMBEDDING_BATCH_SIZE", "0"),
        ("EMBEDDING_BATCH_SIZE", "1025"),
    ],
)
def test_embedding_settings_are_bounded(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=name.lower()):
        Settings()


def test_index_defaults_name_the_live_alias_and_keep_one_previous_build() -> None:
    settings = Settings()

    assert settings.qdrant_collection_alias == "hpdocs_chunks"
    assert settings.index_keep_previous is True


def test_index_settings_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_ALIAS", "hpdocs-staging")
    monkeypatch.setenv("INDEX_KEEP_PREVIOUS", "false")

    settings = Settings()

    assert (settings.qdrant_collection_alias, settings.index_keep_previous) == (
        "hpdocs-staging",
        False,
    )


@pytest.mark.parametrize("alias", ["", "has space", "hpdocs/chunks", "hpdocs.chunks"])
def test_the_alias_must_be_a_plain_name(monkeypatch: pytest.MonkeyPatch, alias: str) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_ALIAS", alias)

    with pytest.raises(ValidationError, match="qdrant_collection_alias"):
        Settings()


def test_chunk_limits_default_to_chunking_v1() -> None:
    settings = Settings()

    # 450 includes the prefix and leaves room under bge-small's 512-token input (ADR 0005).
    assert (settings.chunk_max_tokens, settings.chunk_min_tokens) == (450, 65)


def test_chunk_limits_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "300")
    monkeypatch.setenv("CHUNK_MIN_TOKENS", "40")

    settings = Settings()

    assert (settings.chunk_max_tokens, settings.chunk_min_tokens) == (300, 40)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CHUNK_MAX_TOKENS", "513"),  # over the dense model's input
        ("CHUNK_MAX_TOKENS", "31"),
        ("CHUNK_MIN_TOKENS", "-1"),
    ],
)
def test_chunk_limits_are_bounded(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=name.lower()):
        Settings()


def test_chunk_merge_floor_must_be_below_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "100")
    monkeypatch.setenv("CHUNK_MIN_TOKENS", "99")
    assert Settings().chunk_min_tokens == 99  # just below is fine

    monkeypatch.setenv("CHUNK_MIN_TOKENS", "100")
    with pytest.raises(ValidationError, match="chunk_min_tokens must be below chunk_max_tokens"):
        Settings()


# ------------------------------------------------------------------ retrieval (Phase 2)


def test_retrieval_defaults_are_hybrid_with_rerank_and_the_calibrated_threshold() -> None:
    settings = Settings()

    assert settings.retrieval_mode is SearchMode.HYBRID
    assert settings.retrieval_rerank_enabled is True
    assert settings.retrieval_top_k == 5
    assert settings.retrieval_rerank_candidates == 20
    assert settings.retrieval_prefetch_dense_limit == 20
    assert settings.retrieval_prefetch_sparse_limit == 20
    assert settings.retrieval_rrf_k == 60
    assert settings.retrieval_product_filter_enabled is False
    # Calibrated on the dev split for the default reranker (ADR 0009).
    assert settings.retrieval_relevance_threshold == -6.54
    assert settings.retrieval_dense_query_prefix == ""
    assert settings.reranker_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert settings.inference_workers == 2
    assert settings.inference_onnx_threads == 2
    assert settings.qdrant_search_timeout_seconds == 2.0
    assert settings.query_max_chars == 1000
    assert settings.search_endpoint_enabled is False


def test_retrieval_settings_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "RETRIEVAL_MODE": "sparse",
        "RETRIEVAL_RERANK_ENABLED": "false",
        "RETRIEVAL_TOP_K": "3",
        "RETRIEVAL_RERANK_CANDIDATES": "10",
        "RETRIEVAL_PREFETCH_DENSE_LIMIT": "30",
        "RETRIEVAL_PREFETCH_SPARSE_LIMIT": "40",
        "RETRIEVAL_RRF_K": "2",
        "RETRIEVAL_PRODUCT_FILTER_ENABLED": "true",
        "RETRIEVAL_RELEVANCE_THRESHOLD": "-1.5",
        "RETRIEVAL_DENSE_QUERY_PREFIX": "Represent this sentence: ",
        "RERANKER_MODEL": "BAAI/bge-reranker-base",
        "INFERENCE_WORKERS": "4",
        "INFERENCE_ONNX_THREADS": "1",
        "QDRANT_SEARCH_TIMEOUT_SECONDS": "0.5",
        "QUERY_MAX_CHARS": "500",
        "SEARCH_ENDPOINT_ENABLED": "true",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings()

    assert settings.retrieval_mode is SearchMode.SPARSE
    assert settings.retrieval_rerank_enabled is False
    assert (settings.retrieval_top_k, settings.retrieval_rerank_candidates) == (3, 10)
    assert (settings.retrieval_prefetch_dense_limit, settings.retrieval_prefetch_sparse_limit) == (
        30,
        40,
    )
    assert settings.retrieval_rrf_k == 2
    assert settings.retrieval_product_filter_enabled is True
    assert settings.retrieval_relevance_threshold == -1.5
    assert settings.retrieval_dense_query_prefix == "Represent this sentence: "
    assert settings.reranker_model == "BAAI/bge-reranker-base"
    assert (settings.inference_workers, settings.inference_onnx_threads) == (4, 1)
    assert settings.qdrant_search_timeout_seconds == 0.5
    assert settings.query_max_chars == 500
    assert settings.search_endpoint_enabled is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RETRIEVAL_MODE", "keyword"),
        ("RETRIEVAL_TOP_K", "0"),
        ("RETRIEVAL_TOP_K", "21"),
        ("RETRIEVAL_RERANK_CANDIDATES", "101"),
        ("RETRIEVAL_PREFETCH_DENSE_LIMIT", "201"),
        ("RETRIEVAL_PREFETCH_SPARSE_LIMIT", "201"),
        ("RETRIEVAL_RRF_K", "0"),
        ("RETRIEVAL_RRF_K", "1001"),
        ("RETRIEVAL_RELEVANCE_THRESHOLD", "101"),
        ("RETRIEVAL_RELEVANCE_THRESHOLD", "-101"),
        ("RETRIEVAL_RELEVANCE_THRESHOLD", "nan"),
        ("RETRIEVAL_DENSE_QUERY_PREFIX", "x" * 201),
        ("RERANKER_MODEL", ""),
        ("INFERENCE_WORKERS", "0"),
        ("INFERENCE_WORKERS", "33"),
        ("INFERENCE_ONNX_THREADS", "0"),
        ("INFERENCE_ONNX_THREADS", "33"),
        ("QDRANT_SEARCH_TIMEOUT_SECONDS", "0"),
        ("QDRANT_SEARCH_TIMEOUT_SECONDS", "31"),
        ("QUERY_MAX_CHARS", "0"),
        ("QUERY_MAX_CHARS", "4001"),
    ],
)
def test_retrieval_settings_are_bounded(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=name.lower()):
        Settings()


def test_rerank_candidates_must_cover_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
    monkeypatch.setenv("RETRIEVAL_RERANK_CANDIDATES", "10")
    assert Settings().retrieval_rerank_candidates == 10  # equal is fine

    monkeypatch.setenv("RETRIEVAL_RERANK_CANDIDATES", "9")
    with pytest.raises(ValidationError, match="retrieval_rerank_candidates must be at least"):
        Settings()


@pytest.mark.parametrize(
    "name", ["RETRIEVAL_PREFETCH_DENSE_LIMIT", "RETRIEVAL_PREFETCH_SPARSE_LIMIT"]
)
def test_prefetch_limits_must_cover_the_rerank_candidates(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "20")
    assert Settings()  # equal to the candidates is fine

    monkeypatch.setenv(name, "19")
    with pytest.raises(ValidationError, match="prefetch limits must be at least"):
        Settings()


def test_the_search_endpoint_is_rejected_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_ENDPOINT_ENABLED", "true")
    monkeypatch.setenv("APP_ENV", "test")
    assert Settings().search_endpoint_enabled is True

    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(ValidationError, match="search_endpoint_enabled=true is not allowed"):
        Settings()


@pytest.mark.parametrize("raw", ["", "  "])
def test_a_blank_relevance_threshold_means_not_calibrated(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("RETRIEVAL_RELEVANCE_THRESHOLD", raw)

    assert Settings().retrieval_relevance_threshold is None
