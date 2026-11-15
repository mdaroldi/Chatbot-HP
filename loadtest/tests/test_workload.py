"""The workload: dev golden questions only, a new conversation with 1 to 3 follow-ups,
think time before each question, and the browser cost of every action (brief, 5.4)."""

import pathlib
import random
import tomllib
from collections import Counter

import pytest

from hpdocs_load.workload import (
    CREATE,
    CURRENT,
    DELETE,
    GOLDEN,
    LIST,
    MESSAGE,
    OPEN,
    PREVIOUS,
    READY,
    THINK,
    THINK_MAX_SECONDS,
    THINK_MIN_SECONDS,
    Action,
    QuestionPool,
    load_question_pool,
    page_open,
    plan_conversation,
    think_seconds,
)

POOL = QuestionPool(first=("q1", "q2", "q3"), follow_ups=("f1", "f2"))


def golden_questions() -> list[dict[str, str]]:
    with GOLDEN.open("rb") as handle:
        questions: list[dict[str, str]] = tomllib.load(handle)["questions"]
    return questions


# ---------------------------------------------------------------- question pool


def test_the_real_pool_holds_dev_questions_only() -> None:
    pool = load_question_pool(GOLDEN)
    questions = golden_questions()
    heldout = {q["question"] for q in questions if q["split"] == "heldout"}
    dev_first = {
        q["question"] for q in questions if q["split"] == "dev" and q["category"] != "follow_up"
    }
    dev_follow = {
        q["question"] for q in questions if q["split"] == "dev" and q["category"] == "follow_up"
    }

    assert set(pool.first) == dev_first
    assert set(pool.follow_ups) == dev_follow
    assert not heldout & (set(pool.first) | set(pool.follow_ups))


def test_the_pool_order_follows_the_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "golden.toml"
    path.write_text(
        "\n".join(
            [
                '[[questions]]\nsplit = "dev"\ncategory = "factual"\nquestion = "b"',
                '[[questions]]\nsplit = "heldout"\ncategory = "factual"\nquestion = "x"',
                '[[questions]]\nsplit = "dev"\ncategory = "follow_up"\nquestion = "f"',
                '[[questions]]\nsplit = "dev"\ncategory = "table"\nquestion = "a"',
            ]
        )
    )

    assert load_question_pool(path) == QuestionPool(first=("b", "a"), follow_ups=("f",))


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            '[[questions]]\nsplit = "dev"\ncategory = "follow_up"\nquestion = "f"', id="no-first"
        ),
        pytest.param(
            '[[questions]]\nsplit = "dev"\ncategory = "factual"\nquestion = "q"', id="no-follow-up"
        ),
    ],
)
def test_a_pool_without_first_questions_or_follow_ups_is_rejected(
    tmp_path: pathlib.Path, body: str
) -> None:
    path = tmp_path / "golden.toml"
    path.write_text(body)

    with pytest.raises(ValueError, match="pool"):
        load_question_pool(path)


# ---------------------------------------------------------------- think time


def test_think_time_stays_in_the_approved_range() -> None:
    rng = random.Random(5)
    values = [think_seconds(rng) for _ in range(2000)]

    assert (THINK_MIN_SECONDS, THINK_MAX_SECONDS) == (10.0, 30.0)
    assert all(10.0 <= v <= 30.0 for v in values)
    assert min(values) < 11.0
    assert max(values) > 29.0


# ---------------------------------------------------------------- page open


def test_opening_the_page_checks_readiness_then_lists_conversations() -> None:
    assert page_open() == (Action(READY), Action(LIST))


# ---------------------------------------------------------------- conversation plan


def questions_of(plan: tuple[Action, ...]) -> list[Action]:
    return [a for a in plan if a.kind == MESSAGE]


def test_a_conversation_starts_with_thinking_then_creates_and_asks() -> None:
    plan = plan_conversation(random.Random(1), POOL)

    assert [a.kind for a in plan[:4]] == [THINK, CREATE, MESSAGE, LIST]
    assert plan[2].text in POOL.first
    assert not plan[2].follow_up
    assert sum(a.kind == CREATE for a in plan) == 1


def test_each_follow_up_is_think_message_list_with_a_follow_up_text() -> None:
    plan = plan_conversation(random.Random(2), POOL)
    follow_ups = questions_of(plan)[1:]

    assert follow_ups
    for message in follow_ups:
        at = plan.index(message)
        assert plan[at - 1].kind == THINK
        assert plan[at + 1].kind == LIST
        assert message.text in POOL.follow_ups
        assert message.follow_up  # its own SLO class: the rewrite comes before retrieval


def test_follow_ups_number_one_to_three_and_each_count_occurs() -> None:
    counts = Counter(
        len(questions_of(plan_conversation(random.Random(seed), POOL))) - 1 for seed in range(600)
    )

    assert set(counts) == {1, 2, 3}


def test_every_think_step_has_an_approved_duration() -> None:
    plan = plan_conversation(random.Random(3), POOL)

    thinks = [a for a in plan if a.kind == THINK]
    assert thinks
    assert all(THINK_MIN_SECONDS <= a.seconds <= THINK_MAX_SECONDS for a in thinks)


def test_select_reload_and_delete_follow_the_questions_at_their_rates() -> None:
    plans = [plan_conversation(random.Random(seed), POOL) for seed in range(3000)]

    def rate(match: tuple[Action, ...]) -> float:
        hits = 0
        for plan in plans:
            kinds = [(a.kind, a.target) for a in plan]
            wanted = [(a.kind, a.target) for a in match]
            hits += any(
                kinds[i : i + len(wanted)] == wanted for i in range(len(kinds) - len(wanted) + 1)
            )
        return hits / len(plans)

    select = (Action(THINK), Action(OPEN, target=PREVIOUS))
    reload_ = (Action(READY), Action(LIST), Action(OPEN))
    delete = (Action(THINK), Action(DELETE), Action(LIST))

    assert rate(select) == pytest.approx(0.3, abs=0.03)
    assert rate(reload_) == pytest.approx(0.1, abs=0.02)
    assert rate(delete) == pytest.approx(0.1, abs=0.02)


def test_extra_actions_come_after_the_last_question_in_a_fixed_order() -> None:
    for seed in range(400):
        plan = plan_conversation(random.Random(seed), POOL)
        last_list = max(i for i, a in enumerate(plan) if a.kind == MESSAGE) + 1
        tail = [(a.kind, a.target) for a in plan[last_list + 1 :] if a.kind != THINK]

        assert plan[last_list].kind == LIST
        allowed = [
            (OPEN, PREVIOUS),
            (READY, CURRENT),
            (LIST, CURRENT),
            (OPEN, CURRENT),
            (DELETE, CURRENT),
            (LIST, CURRENT),
        ]
        it = iter(allowed)
        assert all(step in it for step in tail), (seed, tail)


def test_the_same_seed_gives_the_same_plan() -> None:
    assert plan_conversation(random.Random(9), POOL) == plan_conversation(random.Random(9), POOL)
