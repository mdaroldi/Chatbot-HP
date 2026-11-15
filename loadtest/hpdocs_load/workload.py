"""What one simulated browser does, and what each action costs in API calls (brief, 5.4).

Questions are the dev split of the frozen golden set, so no held-out question is ever
sent and the text is a small fixed list (rule 9). One plan is one conversation: think,
create, ask, list; then 1 to 3 follow-ups the same way; then, at fixed rates, select an
earlier conversation, reload the page, or delete the conversation. Think time comes
before each question: the user reads the answer and types the next one.
"""

import pathlib
import random
import tomllib
from dataclasses import dataclass

GOLDEN = pathlib.Path(__file__).resolve().parents[2] / "eval" / "golden" / "retrieval-v1.toml"

THINK, CREATE, MESSAGE, LIST, OPEN, DELETE, READY = (
    "think",
    "create",
    "message",
    "list",
    "open",
    "delete",
    "ready",
)
CURRENT, PREVIOUS = "current", "previous"

THINK_MIN_SECONDS, THINK_MAX_SECONDS = 10.0, 30.0
FOLLOW_UP_COUNTS, FOLLOW_UP_WEIGHTS = (1, 2, 3), (0.5, 0.3, 0.2)
SELECT_RATE, RELOAD_RATE, DELETE_RATE = 0.3, 0.1, 0.1
_DEV, _FOLLOW_UP = "dev", "follow_up"


@dataclass(frozen=True)
class QuestionPool:
    first: tuple[str, ...]
    follow_ups: tuple[str, ...]


@dataclass(frozen=True)
class Action:
    kind: str
    text: str | None = None
    seconds: float = 0.0
    target: str = CURRENT
    follow_up: bool = False


def load_question_pool(path: pathlib.Path) -> QuestionPool:
    """Follow-up questions are written to follow a turn, so they are only asked second."""
    with path.open("rb") as handle:
        questions = tomllib.load(handle)["questions"]
    dev = [q for q in questions if q["split"] == _DEV]
    pool = QuestionPool(
        first=tuple(q["question"] for q in dev if q["category"] != _FOLLOW_UP),
        follow_ups=tuple(q["question"] for q in dev if q["category"] == _FOLLOW_UP),
    )
    if not pool.first or not pool.follow_ups:
        raise ValueError("the question pool needs dev first questions and dev follow-ups")
    return pool


def think_seconds(rng: random.Random) -> float:
    return rng.uniform(THINK_MIN_SECONDS, THINK_MAX_SECONDS)


def page_open() -> tuple[Action, ...]:
    """A new browser has no conversation selected, so no thread is fetched."""
    return (Action(READY), Action(LIST))


def _think(rng: random.Random) -> Action:
    return Action(THINK, seconds=think_seconds(rng))


def _ask(
    rng: random.Random, texts: tuple[str, ...], *, follow_up: bool = False
) -> tuple[Action, ...]:
    message = Action(MESSAGE, text=rng.choice(texts), follow_up=follow_up)
    return (_think(rng), message, Action(LIST))


def plan_conversation(rng: random.Random, pool: QuestionPool) -> tuple[Action, ...]:
    first = _ask(rng, pool.first)
    plan = [first[0], Action(CREATE), *first[1:]]
    (count,) = rng.choices(FOLLOW_UP_COUNTS, weights=FOLLOW_UP_WEIGHTS)
    for _ in range(count):
        plan += _ask(rng, pool.follow_ups, follow_up=True)
    if rng.random() < SELECT_RATE:
        plan += [_think(rng), Action(OPEN, target=PREVIOUS)]
    if rng.random() < RELOAD_RATE:
        plan += [Action(READY), Action(LIST), Action(OPEN)]
    if rng.random() < DELETE_RATE:
        plan += [_think(rng), Action(DELETE), Action(LIST)]
    return tuple(plan)
