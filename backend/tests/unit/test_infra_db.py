"""Behavior of the async SQLAlchemy engine and session factory configuration."""

import pathlib

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.infra.db import create_engine, create_session_factory, engine_options
from tests.unit.conftest import TEST_PASSWORD, SettingsFactory

APP_DIR = pathlib.Path(__file__).resolve().parents[2] / "app"


def test_engine_options_pin_logging_pooling_and_connect_timeout(
    make_settings: SettingsFactory,
) -> None:
    settings = make_settings(
        postgres_pool_size=7, postgres_max_overflow=3, postgres_connect_timeout_seconds=2.5
    )

    assert engine_options(settings) == {
        "echo": False,
        "pool_pre_ping": True,
        "pool_size": 7,
        "max_overflow": 3,
        "connect_args": {"timeout": 2.5},
    }


async def test_engine_never_echoes_sql_and_uses_bounded_pool(
    make_settings: SettingsFactory,
) -> None:
    engine = create_engine(make_settings(postgres_pool_size=4))
    try:
        assert isinstance(engine, AsyncEngine)
        assert engine.echo is False
        assert engine.pool.size() == 4  # type: ignore[attr-defined]  # QueuePool API
    finally:
        await engine.dispose()


async def test_engine_targets_asyncpg_without_exposing_the_password(
    make_settings: SettingsFactory,
) -> None:
    engine = create_engine(make_settings(postgres_host="db.internal", postgres_port=6543))
    try:
        assert engine.url.drivername == "postgresql+asyncpg"
        assert (engine.url.host, engine.url.port) == ("db.internal", 6543)
        assert TEST_PASSWORD not in repr(engine)
        assert TEST_PASSWORD not in str(engine.url)
    finally:
        await engine.dispose()


async def test_session_factory_yields_async_sessions_that_keep_loaded_state(
    make_settings: SettingsFactory,
) -> None:
    engine = create_engine(make_settings())
    try:
        factory = create_session_factory(engine)
        async with factory() as session:
            assert isinstance(session, AsyncSession)
            assert session.bind is engine
        assert factory.kw["expire_on_commit"] is False
    finally:
        await engine.dispose()


def test_dsn_secret_is_unwrapped_in_exactly_one_place() -> None:
    """One audited spot turns the DSN into plain text: the engine factory."""
    hits = [
        (path.relative_to(APP_DIR).as_posix(), line_number)
        for path in APP_DIR.rglob("*.py")
        for line_number, line in enumerate(path.read_text().splitlines(), start=1)
        if "postgres_dsn.get_secret_value()" in line
    ]

    assert [path for path, _ in hits] == ["infra/db.py"]
