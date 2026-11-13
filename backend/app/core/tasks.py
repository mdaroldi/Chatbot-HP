"""Work that must finish even when the request that started it is cancelled."""

import asyncio
from collections.abc import Coroutine

# Shielded work in flight, kept referenced until done (the loop holds tasks weakly).
_RUNNING: set[object] = set()


async def shielded[T](work: Coroutine[object, None, T]) -> T:
    """Run `work` in its own task to the end, even if the caller is cancelled (again)
    while it runs; the caller still sees the cancellation.
    """
    task = asyncio.ensure_future(work)
    _RUNNING.add(task)
    task.add_done_callback(_RUNNING.discard)
    return await asyncio.shield(task)
