"""`python -m app.retention` (`make purge`): delete expired conversations now.

The same purge the backend runs every `RETENTION_PURGE_INTERVAL_MINUTES`, under the
same advisory lock, so it is safe while the backend runs. Prints the counts; exits 1
when another purge holds the lock or the purge failed (logged by error type only).
"""

import asyncio
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TextIO

from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.logging import configure_logging
from app.core.settings import Settings, get_settings
from app.infra.db import create_engine, create_session_factory
from app.infra.retention import PostgresPurgeLock, SqlExpiredChats
from app.services.retention import purge_expired

logger = logging.getLogger(__name__)

EngineFactory = Callable[[Settings], AsyncEngine]


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def run(
    settings: Settings,
    *,
    engine_factory: EngineFactory = create_engine,
    out: TextIO,
    clock: Callable[[], datetime] = _utc_now,
) -> int:
    engine = engine_factory(settings)
    days = settings.chat_retention_days
    try:
        counts = await purge_expired(
            SqlExpiredChats(create_session_factory(engine)),
            PostgresPurgeLock(engine),
            days=days,
            clock=clock,
        )
    except Exception as error:
        logger.error("retention purge failed", extra={"error_type": type(error).__name__})  # noqa: TRY400 - no traceback: it could quote values
        return 1
    finally:
        await engine.dispose()
    if counts is None:
        out.write("purge skipped: another purge holds the lock\n")
        return 1
    out.write(
        f"purged conversations={counts.conversations} messages={counts.messages} "
        f"retention_days={days}\n"
    )
    return 0


def main(*, settings: Settings | None = None, engine_factory: EngineFactory = create_engine) -> int:
    settings = settings or get_settings()
    configure_logging(
        settings.log_level, include_exception_messages=settings.log_include_exception_messages
    )
    return asyncio.run(run(settings, engine_factory=engine_factory, out=sys.stdout))
