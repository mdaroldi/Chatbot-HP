"""shielded(): the work finishes even when its caller is cancelled."""

import asyncio

import pytest

from app.core.tasks import shielded


async def test_the_result_reaches_the_caller() -> None:
    async def work() -> int:
        return 7

    assert await shielded(work()) == 7


async def test_cancelling_the_caller_does_not_stop_the_work() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    finished: list[bool] = []

    async def work() -> None:
        started.set()
        await release.wait()
        finished.append(True)

    caller = asyncio.create_task(shielded(work()))
    await started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    release.set()
    for _ in range(3):
        await asyncio.sleep(0)

    assert finished == [True]
