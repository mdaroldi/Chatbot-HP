"""The burst client of the rate-limits run (ADR 0013): one client ID, no think time, the
fixed dev question list. It runs only when LOAD_BURST=1, next to the normal users and
not instead of one, and its records never mix with theirs."""

import ast
import itertools
import pathlib

from hpdocs_load.burst import burst_questions, shape_users
from hpdocs_load.workload import MESSAGE, THINK, QuestionPool

ROOT = pathlib.Path(__file__).resolve().parents[1]
POOL = QuestionPool(first=("q1", "q2", "q3"), follow_ups=("f1",))


def test_questions_are_the_first_questions_in_order_over_and_over() -> None:
    actions = list(itertools.islice(burst_questions(POOL), 7))

    assert [a.text for a in actions] == ["q1", "q2", "q3", "q1", "q2", "q3", "q1"]


def test_there_is_no_think_time_between_questions() -> None:
    actions = list(itertools.islice(burst_questions(POOL), 10))

    assert {a.kind for a in actions} == {MESSAGE}
    assert all(a.kind != THINK and a.seconds == 0.0 for a in actions)


def test_follow_up_questions_are_never_sent() -> None:
    actions = itertools.islice(burst_questions(POOL), 10)

    assert not {a.text for a in actions} & set(POOL.follow_ups)


def test_the_burst_client_comes_on_top_of_the_step_users() -> None:
    assert shape_users((16, 2.0), burst=True) == (17, 2.0)


def test_without_the_burst_client_the_step_is_unchanged() -> None:
    assert shape_users((16, 2.0), burst=False) == (16, 2.0)


def test_the_end_of_the_run_stays_the_end() -> None:
    assert shape_users(None, burst=True) is None
    assert shape_users(None, burst=False) is None


# The locustfile is glue outside the coverage gate, so the guard lines are pinned here.


def _class(name: str) -> ast.ClassDef:
    tree = ast.parse((ROOT / "locustfile.py").read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _assignments(node: ast.ClassDef) -> dict[str, str]:
    return {
        target.id: ast.unparse(statement.value)
        for statement in node.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }


def test_the_burst_user_exists_only_when_enabled_and_only_once() -> None:
    fields = _assignments(_class("Burst"))

    assert fields["abstract"] == "not CONFIG.burst"
    assert fields["fixed_count"] == "1"
    assert fields["wait_time"] == "constant(0)"


def test_the_burst_user_records_into_its_own_file_with_its_own_questions() -> None:
    source = ast.unparse(_class("Burst"))

    assert "files.add_burst_request" in source
    assert "files.add_request" not in source
    assert "burst_questions(POOL)" in source
    assert "new_client_id" in source


def test_normal_users_never_record_into_the_burst_file() -> None:
    source = ast.unparse(_class("Browser"))

    assert "files.add_request" in source
    assert "add_burst_request" not in source


def test_the_shape_adds_the_burst_user_only_through_the_config_flag() -> None:
    source = ast.unparse(_class("StepShape"))

    assert "return shape_users(tick(CONFIG.plan, now), burst=CONFIG.burst)" in source
    assert "now = self.get_run_time()" in source


def test_the_run_metadata_says_whether_the_burst_client_ran() -> None:
    tree = ast.parse((ROOT / "locustfile.py").read_text())
    meta = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_meta")

    assert "'burst': CONFIG.burst" in ast.unparse(meta)
