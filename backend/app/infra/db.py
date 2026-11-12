"""Async SQLAlchemy engine and session factory for PostgreSQL (asyncpg driver)."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.settings import Settings

EngineOptions = dict[str, object]


def engine_options(settings: Settings) -> EngineOptions:
    """Engine options in one testable place.

    echo stays off: SQL logging would write bound parameters, i.e. user content.
    pool_pre_ping drops connections Postgres closed while idle. The pool is bounded so
    one replica cannot exhaust max_connections, and the connect timeout stops a dead
    host from stalling requests.
    """
    return {
        "echo": False,
        "pool_pre_ping": True,
        "pool_size": settings.postgres_pool_size,
        "max_overflow": settings.postgres_max_overflow,
        "connect_args": {"timeout": settings.postgres_connect_timeout_seconds},
    }


def create_engine(settings: Settings) -> AsyncEngine:
    """Create the engine; creating it opens no connection."""
    # The only place the DSN secret is unwrapped (enforced by a unit test).
    dsn = settings.postgres_dsn.get_secret_value()
    return create_async_engine(dsn, **engine_options(settings))


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Sessions keep loaded attributes after commit, so async code never lazy-loads."""
    return async_sessionmaker(engine, expire_on_commit=False)
