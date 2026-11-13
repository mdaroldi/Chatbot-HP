"""The benchmark CLI: commands, the held-out and preflight gates, exit codes, reports, logs."""

import hashlib
import json
import pathlib
import runpy
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from app.core.settings import Settings
from app.domain.evaluation import IndexedChunk
from app.evaluation import cli
from app.evaluation.cli import EvaluationError, Failure, Runtime, main
from app.evaluation.runner import IndexInfo, Retriever
from app.infra.inference import CpuAllowance, CpuSource
from app.services.retrieval import RetrievalService
from tests.unit.conftest import LogReader
from tests.unit.eval_fakes import ENVY, chunk
from tests.unit.retrieval_fakes import FakeQueryEmbedder, FakeReranker, FakeSearchStore, make_hit

SHA = "0123456789abcdef0123456789abcdef01234567"
NOW = datetime(2026, 11, 13, 16, 0, 0, tzinfo=UTC)
FACT = "tray holds 80"
CANARY = "CANARY-7f3a"
INDEX = IndexInfo(
    alias="hpdocs_chunks",
    collection="hpdocs_chunks_v1_20261113t100000z",
    build_id="20261113t100000z",
    chunk_count=1,
    chunking_version="v1",
)

REVISIONS = {
    "BAAI/bge-small-en-v1.5": "rev-d",
    "Qdrant/bm25": "rev-s",
    "Xenova/ms-marco-MiniLM-L-6-v2": "rev-r",
    "jinaai/jina-reranker-v1-turbo-en": "rev-j",
}

MATRIX = """
[bootstrap]
seed = 5
resamples = 20

[[configs]]
name = "sparse"
mode = "sparse"
rerank = false

[[configs]]
name = "hybrid-rerank"
"""


def golden_toml(fact: str = FACT) -> str:
    """Eight questions: three dev and one held-out per category, so the split is valid."""
    blocks = ['version = "1.0.0"']
    for n, split in enumerate(["dev", "dev", "dev", "heldout"], 1):
        blocks.append(
            f'[[questions]]\nid = "f{n}"\ncategory = "factual"\nsplit = "{split}"\n'
            f'question = "{CANARY} question f{n}?"\nmatch = "any"\n'
            f'[[questions.anchors]]\ndoc_id = "{ENVY}"\nfacts = ["{fact}"]'
        )
        blocks.append(
            f'[[questions]]\nid = "o{n}"\ncategory = "out_of_scope"\nsplit = "{split}"\n'
            f'question = "{CANARY} question o{n}?"\noos_kind = "unrelated"'
        )
    return "\n\n".join(blocks) + "\n"


@pytest.fixture
def eval_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "golden").mkdir()
    (tmp_path / "golden" / "retrieval-v1.toml").write_text(golden_toml(), encoding="utf-8")
    (tmp_path / "retrieval-matrix.toml").write_text(MATRIX, encoding="utf-8")
    return tmp_path


@dataclass
class FakeRuntime:
    """Runtime factory over fakes; records each opening and its collection."""

    chunks: list[IndexedChunk] = field(
        default_factory=lambda: [chunk("c1", f"{CANARY} the {FACT} sheets")]
    )
    error: Exception | None = None
    opened: list[str | None] = field(default_factory=list)
    rerankers: list[str] = field(default_factory=list)

    def __call__(
        self, settings: Settings, collection: str | None
    ) -> AbstractAsyncContextManager[Runtime]:
        assert settings.postgres_password.get_secret_value()
        return self._open(collection)

    @asynccontextmanager
    async def _open(self, collection: str | None) -> AsyncIterator[Runtime]:
        self.opened.append(collection)
        if self.error is not None:
            raise self.error
        hits = [make_hit("c1", 0.9, doc_id=ENVY, text=f"{CANARY} the {FACT} sheets")]

        async def all_chunks() -> list[IndexedChunk]:
            return list(self.chunks)

        async def retriever(name: str) -> Retriever:
            self.rerankers.append(name)
            return RetrievalService(
                FakeQueryEmbedder(), FakeSearchStore(hits), FakeReranker(model=name)
            )

        yield Runtime(
            index=INDEX,
            chunks=all_chunks,
            retriever=retriever,
            revisions=REVISIONS,
        )


@pytest.fixture(autouse=True)
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "configure_logging", lambda *_, **__: None)


def invoke(
    args: list[str],
    eval_dir: pathlib.Path,
    runtime: FakeRuntime,
    env: dict[str, str] | None = None,
) -> int:
    return main(
        [*args, "--eval-dir", str(eval_dir)],
        runtime=runtime,
        environ={"GIT_SHA": SHA} if env is None else env,
        clock=lambda: NOW,
        cpu=lambda: CpuAllowance(8.0, CpuSource.CGROUP),
    )


def run_dirs(eval_dir: pathlib.Path) -> list[pathlib.Path]:
    root = eval_dir / "reports" / "runs"
    return sorted(root.iterdir()) if root.exists() else []


def summary_of(eval_dir: pathlib.Path) -> dict[str, Any]:
    # Any: parsed JSON.
    (path,) = run_dirs(eval_dir)
    loaded: dict[str, Any] = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return loaded


def lines(logs: LogReader, message: str) -> list[dict[str, Any]]:
    return [line for line in logs() if line["message"] == message]


# ------------------------------------------------------------------ held-out gate


def test_the_heldout_split_is_refused_without_the_flag(
    eval_dir: pathlib.Path, json_logs: LogReader
) -> None:
    runtime = FakeRuntime()

    assert invoke(["run", "--split", "heldout"], eval_dir, runtime) == 2

    assert runtime.opened == []
    assert run_dirs(eval_dir) == []
    assert lines(json_logs, "held-out split refused; pass --allow-heldout")


def test_the_heldout_split_runs_with_the_flag(eval_dir: pathlib.Path) -> None:
    runtime = FakeRuntime()

    assert invoke(["run", "--split", "heldout", "--allow-heldout"], eval_dir, runtime) == 0

    assert summary_of(eval_dir)["golden"]["split"] == "heldout"
    assert summary_of(eval_dir)["golden"]["questions"] == 2


# ------------------------------------------------------------------ preflight


def test_a_clean_preflight_passes_and_writes_nothing(
    eval_dir: pathlib.Path, json_logs: LogReader
) -> None:
    runtime = FakeRuntime()

    assert invoke(["preflight"], eval_dir, runtime) == 0

    assert run_dirs(eval_dir) == []
    (done,) = lines(json_logs, "preflight passed")
    assert (done["chunks"], done["anchors"], done["match_counts"]) == (1, 4, {"1": 4})


def test_an_unresolved_anchor_fails_the_preflight_and_names_the_question(
    eval_dir: pathlib.Path, json_logs: LogReader
) -> None:
    runtime = FakeRuntime(chunks=[chunk("c1", "something else")])

    assert invoke(["preflight"], eval_dir, runtime) == 1

    problems = lines(json_logs, "anchor does not resolve")
    assert [(p["question_id"], p["split"], p["anchor"], p["matches"]) for p in problems] == [
        ("f1", "dev", 0, 0),
        ("f2", "dev", 0, 0),
        ("f3", "dev", 0, 0),
        ("f4", "heldout", 0, 0),
    ]
    assert {p["problem"] for p in problems} == {"unresolved"}
    (failed,) = lines(json_logs, "preflight failed")
    assert (failed["unresolved"], failed["too_generic"]) == (4, 0)
    assert FACT not in json.dumps(json_logs())


def test_a_run_stops_before_any_retrieval_when_the_preflight_fails(
    eval_dir: pathlib.Path,
) -> None:
    runtime = FakeRuntime(chunks=[])

    assert invoke(["run"], eval_dir, runtime) == 1

    assert runtime.rerankers == []
    assert run_dirs(eval_dir) == []


def test_the_preflight_reads_the_collection_it_is_given(eval_dir: pathlib.Path) -> None:
    runtime = FakeRuntime()

    assert invoke(["preflight", "--collection", "kept_v1_x"], eval_dir, runtime) == 0

    assert runtime.opened == ["kept_v1_x"]


# ------------------------------------------------------------------ run and report


def test_a_run_reports_golden_index_git_hardware_and_every_config(
    eval_dir: pathlib.Path,
) -> None:
    runtime = FakeRuntime()

    assert invoke(["run"], eval_dir, runtime) == 0

    (path,) = run_dirs(eval_dir)
    assert path.name == f"20261113t160000z-{SHA[:12]}"
    summary = summary_of(eval_dir)
    golden_bytes = (eval_dir / "golden" / "retrieval-v1.toml").read_bytes()
    assert summary["golden"] == {
        "version": "1.0.0",
        "sha256": hashlib.sha256(golden_bytes).hexdigest(),
        "split": "dev",
        "questions": 6,
    }
    assert summary["index"]["build_id"] == INDEX.build_id
    assert summary["index"]["collection"] == INDEX.collection
    assert summary["git_sha"] == SHA
    assert summary["bootstrap"] == {"seed": 5, "resamples": 20}
    assert summary["hardware"]["cpus"] == 8.0
    assert summary["hardware"]["cpu_source"] == "cgroup"
    assert summary["hardware"]["inference_workers"] == 2
    assert summary["hardware"]["onnx_threads"] == 2
    assert "machine" in summary["hardware"]
    # The models this run used: never a pinned reranker that no config reranked with.
    assert summary["models"] == {
        "BAAI/bge-small-en-v1.5": "rev-d",
        "Qdrant/bm25": "rev-s",
        "Xenova/ms-marco-MiniLM-L-6-v2": "rev-r",
    }
    assert [c["name"] for c in summary["configs"]] == ["sparse", "hybrid-rerank"]
    assert summary["configs"][1]["config"]["top_k"] == 20
    assert summary["preflight"]["problems"] == []
    results = (path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results) == 2 * 6
    assert CANARY not in "".join(p.read_text(encoding="utf-8") for p in path.iterdir())


def test_each_reranker_is_loaded_before_the_first_config_runs(eval_dir: pathlib.Path) -> None:
    runtime = FakeRuntime()

    assert invoke(["run"], eval_dir, runtime) == 0

    assert runtime.rerankers == ["Xenova/ms-marco-MiniLM-L-6-v2"]


def test_selected_configs_alone_are_run(eval_dir: pathlib.Path) -> None:
    assert invoke(["run", "--config", "sparse"], eval_dir, FakeRuntime()) == 0

    assert [c["name"] for c in summary_of(eval_dir)["configs"]] == ["sparse"]


def test_an_unknown_config_is_refused_before_anything_opens(eval_dir: pathlib.Path) -> None:
    runtime = FakeRuntime()

    assert invoke(["run", "--config", "nope"], eval_dir, runtime) == 2

    assert runtime.opened == []


@pytest.mark.parametrize("env", [{}, {"GIT_SHA": ""}, {"GIT_SHA": "main"}, {"GIT_SHA": "abc"}])
def test_a_run_needs_the_git_sha_it_measures(
    eval_dir: pathlib.Path, env: dict[str, str], json_logs: LogReader
) -> None:
    runtime = FakeRuntime()

    assert invoke(["run"], eval_dir, runtime, env) == 2

    assert runtime.opened == []
    assert lines(json_logs, "GIT_SHA must be a commit SHA, optionally with -dirty")


def test_a_dirty_tree_is_recorded_in_the_sha(eval_dir: pathlib.Path) -> None:
    dirty = f"{SHA}-dirty"

    assert invoke(["run"], eval_dir, FakeRuntime(), {"GIT_SHA": dirty}) == 0

    assert summary_of(eval_dir)["git_sha"] == dirty
    assert run_dirs(eval_dir)[0].name.endswith(f"{SHA[:12]}-dirty")


def test_the_run_collection_reaches_the_runtime(eval_dir: pathlib.Path) -> None:
    runtime = FakeRuntime()

    assert invoke(["run", "--collection", "kept_v1_x"], eval_dir, runtime) == 0

    assert runtime.opened == ["kept_v1_x"]


# ------------------------------------------------------------------ failures and logs


def test_an_index_problem_is_one_line_with_its_reason(
    eval_dir: pathlib.Path, json_logs: LogReader
) -> None:
    runtime = FakeRuntime(error=EvaluationError(Failure.NO_LIVE_INDEX))

    assert invoke(["preflight"], eval_dir, runtime) == 1

    (line,) = lines(json_logs, "evaluation failed")
    assert line["reason"] == "no_live_index"


def test_an_unexpected_failure_logs_its_type_and_never_its_message(
    eval_dir: pathlib.Path, json_logs: LogReader
) -> None:
    runtime = FakeRuntime(error=RuntimeError(CANARY))

    assert invoke(["run"], eval_dir, runtime) == 1

    (line,) = lines(json_logs, "evaluation failed")
    assert line["error_type"] == "RuntimeError"
    assert CANARY not in json.dumps(json_logs())


def test_an_invalid_golden_set_is_one_line_with_its_rule(
    eval_dir: pathlib.Path, json_logs: LogReader
) -> None:
    (eval_dir / "golden" / "retrieval-v1.toml").write_text('version = "1"\n', encoding="utf-8")
    runtime = FakeRuntime()

    assert invoke(["preflight"], eval_dir, runtime) == 1

    (line,) = lines(json_logs, "golden set is invalid")
    assert line["rule"] == "invalid_version"
    assert runtime.opened == []


def test_an_invalid_matrix_fails_before_anything_opens(
    eval_dir: pathlib.Path, json_logs: LogReader
) -> None:
    (eval_dir / "retrieval-matrix.toml").write_text("[bootstrap]\n", encoding="utf-8")
    runtime = FakeRuntime()

    assert invoke(["run"], eval_dir, runtime) == 1

    assert lines(json_logs, "retrieval matrix is invalid")
    assert runtime.opened == []


def test_logs_carry_ids_and_timings_but_never_question_or_chunk_text(
    eval_dir: pathlib.Path, json_logs: LogReader
) -> None:
    assert invoke(["run"], eval_dir, FakeRuntime()) == 0

    assert lines(json_logs, "retrieval done")  # the service logged, so the check is real
    (done,) = lines(json_logs, "benchmark written")
    assert done["run"] == f"20261113t160000z-{SHA[:12]}"
    assert CANARY not in json.dumps(json_logs())


def test_bad_arguments_exit_with_two() -> None:
    with pytest.raises(SystemExit) as exited:
        main(["run", "--split", "everything"])

    assert exited.value.code == 2


def test_the_module_runs_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "main", lambda: 7)

    with pytest.raises(SystemExit) as exited:
        runpy.run_module("app.evaluation", run_name="__main__")

    assert exited.value.code == 7


Factory = Callable[[Settings, str | None], AbstractAsyncContextManager[Runtime]]
_FACTORY: Factory = FakeRuntime()
