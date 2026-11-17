"""The burst client of the rate-limits run (ADR 0013).

At the compose default limits a normal user never sees a 429: one client ID each and
10-30 s of think time keep it at a few messages a minute. The burst client is one client
ID asking the fixed dev first questions back to back, with no think time and no backing
off on a 429, so it spends its message bucket and stays refused. It runs only when
LOAD_BURST=1, on top of the step's users, and records into its own file.
"""

import itertools
from collections.abc import Iterator

from hpdocs_load.workload import MESSAGE, Action, QuestionPool


def burst_questions(pool: QuestionPool) -> Iterator[Action]:
    """First questions only: the same small fixed list the normal users draw from."""
    return (Action(MESSAGE, text=text) for text in itertools.cycle(pool.first))


def shape_users(result: tuple[int, float] | None, *, burst: bool) -> tuple[int, float] | None:
    """Locust spawns the fixed-count burst user out of the total, so it is added on top."""
    if result is None or not burst:
        return result
    users, spawn_rate = result
    return users + 1, spawn_rate
