"""`python -m app.retention` (`make purge`): one purge now, counts on stdout."""

import io
import logging
import pathlib
import runpy
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app import retention
from app.core.settings import Settings
from app.infra.db import create_session_factory
from app.infra.orm import ConversationRow
from app.infra.retention import RETENTION_LOCK_KEY
from tests.chat_rows import T0
from tests.unit.conftest import SettingsFactory
from tests.unit.retention_fakes import AdvisoryLocks, sqlite_engine
from tests.unit.test_infra_retention import age, conversation_with_turn

CANARY = "CANARY-purge-cli-91b2"


class Engines:
    """An engine factory over one SQLite file, recording that the command disposed it."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.built = 0

    def __call__(self, _: Settings) -> AsyncEngine:
        self.built += 1
        return self.engine


@pytest.fixture
def locks() -> AdvisoryLocks:
    return AdvisoryLocks()


@pytest.fixture
async def engine(tmp_path: pathlib.Path, locks: AdvisoryLocks) -> AsyncIterator[AsyncEngine]:
    async with sqlite_engine(tmp_path / "purge.db", locks) as engine:
        yield engine


async def ids(engine: AsyncEngine) -> set[str]:
    async with create_session_factory(engine)() as session:
        return set((await session.execute(select(ConversationRow.id))).scalars())


async def test_the_command_purges_expired_conversations_and_prints_counts(
    engine: AsyncEngine, make_settings: SettingsFactory
) -> None:
    sessions = create_session_factory(engine)
    old, newer = await conversation_with_turn(sessions), await conversation_with_turn(sessions)
    await age(sessions, old, timedelta(days=31))
    out = io.StringIO()
    engines = Engines(engine)

    code = await retention.run(
        make_settings(chat_retention_days=30), engine_factory=engines, out=out, clock=lambda: T0
    )

    assert code == 0
    assert out.getvalue() == "purged conversations=1 messages=2 retention_days=30\n"
    assert await ids(engine) == {newer}
    assert engines.built == 1


async def test_the_command_reports_a_busy_lock_and_fails(
    engine: AsyncEngine, locks: AdvisoryLocks, make_settings: SettingsFactory
) -> None:
    locks.holders[RETENTION_LOCK_KEY] = -1  # another session holds it
    out = io.StringIO()

    code = await retention.run(make_settings(), engine_factory=Engines(engine), out=out)

    assert code == 1
    assert out.getvalue() == "purge skipped: another purge holds the lock\n"


async def test_a_failure_is_logged_by_type_only_and_fails_the_command(
    make_settings: SettingsFactory, caplog: pytest.LogCaptureFixture
) -> None:
    class BrokenEngine:
        disposed = False

        def connect(self) -> object:
            raise ConnectionError(CANARY)

        async def dispose(self) -> None:
            BrokenEngine.disposed = True

    out = io.StringIO()

    code = await retention.run(
        make_settings(),
        engine_factory=lambda _: BrokenEngine(),  # type: ignore[arg-type,return-value]
        out=out,
    )

    assert code == 1
    assert BrokenEngine.disposed
    (record,) = [r for r in caplog.records if r.getMessage() == "retention purge failed"]
    assert record.levelno == logging.ERROR
    assert record.__dict__["error_type"] == "ConnectionError"
    assert record.exc_info is None
    assert CANARY not in caplog.text + out.getvalue()


def test_main_runs_with_the_given_settings(
    engine: AsyncEngine, make_settings: SettingsFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    code = retention.main(settings=make_settings(), engine_factory=Engines(engine))

    assert code == 0
    lines = capsys.readouterr().out.splitlines()  # JSON log lines go to stdout too
    assert "purged conversations=0 messages=0 retention_days=30" in lines


def test_the_module_runs_main_and_exits_with_its_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """As `make purge` runs it; here the default engine cannot reach a database."""
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "1")
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("app.retention", run_name="__main__")

    assert exited.value.code == 1
