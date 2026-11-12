"""Alembic environment: async migrations using the application's engine and logging.

Neither the URL nor logging comes from alembic.ini. The engine is built by
create_engine (the single place the DSN secret is unwrapped) and logs go through
configure_logging, so migration output is the same JSON as the API's.
"""

import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.infra.db import create_engine
from app.infra.orm import Base

settings = get_settings()
configure_logging(
    settings.log_level, include_exception_messages=settings.log_include_exception_messages
)
target_metadata = Base.metadata


def _run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    engine = create_engine(settings)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    # Offline SQL generation needs the URL as plain text in Alembic's config.
    raise RuntimeError("Offline migrations are not supported; run them against a database.")

asyncio.run(_run_async_migrations())
