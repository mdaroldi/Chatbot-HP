"""Benchmark reports: what each file records, and that none of them holds document text."""

import dataclasses
import json
import pathlib
from datetime import UTC, datetime

import pytest

from app.domain.evaluation import Metric, Split
from app.domain.retrieval import SearchMode
from app.evaluation.report import (
    RunHeader,
    RunOutcome,
    render_markdown,
    result_record,
    run_dir_name,
    summary_record,
    write_run,
)
from app.evaluation.runner import (
    IndexInfo,
    QuestionRun,
    preflight,
    run_config,
    select_best,
    summarize_config,
)
from tests.unit.eval_fakes import (
    GOLDEN_SHA,
    RERANKER,
    Scripted,
    ScriptedRetriever,
    chunk,
    golden,
    hits,
    named,
    oos,
    question,
)

CHUNK_TEXT = "CANARY-CHUNK the tray holds 80 sheets"
QUESTION_TEXT = "CANARY-QUESTION how many sheets?"
CREATED = datetime(2026, 11, 13, 15, 30, 5, tzinfo=UTC)

HEADER = RunHeader(
    created_at=CREATED,
    git_sha="af4755b0123456789abcdef0123456789abcdef0",
    golden_version="1.0.0",
    golden_sha256=GOLDEN_SHA,
    split=Split.DEV,
    questions=2,
    index=IndexInfo(
        alias="hpdocs_chunks",
        collection="hpdocs_chunks_v1_20261113t100000z",
        build_id="20261113t100000z",
        chunk_count=287,
        chunking_version="v1",
        bm25_avg_len=107.25,
        bm25_avg_len_auto=True,
    ),
    models={"demo/dense": "rev-dense", RERANKER: "rev-rerank"},
    hardware={"cpus": 8.0, "cpu_source": "cpu_count", "inference_workers": 2},
    seed=20261113,
    resamples=50,
    warm_up_retrievals=1,
    reranker_warm_up_pairs=20,
)

GOLDEN = golden(question("q-in", text=QUESTION_TEXT), oos("q-out"))
CONFIGS = (named(), named("sparse", mode=SearchMode.SPARSE, rerank=False))


async def runs() -> dict[str, tuple[QuestionRun, ...]]:
    script = {QUESTION_TEXT: Scripted(hits(CHUNK_TEXT, "other")), "question q-out?": Scripted([])}
    return {
        c.name: await run_config(c, GOLDEN.questions, ScriptedRetriever(script)) for c in CONFIGS
    }


def test_a_run_directory_is_named_by_utc_time_and_short_sha() -> None:
    assert run_dir_name(HEADER) == "20261113t153005z-af4755b01234"
    dirty = dataclasses.replace(HEADER, git_sha="af4755b-dirty")
    assert run_dir_name(dirty) == "20261113t153005z-af4755b-dirty"


async def test_a_result_holds_ids_scores_and_timings_but_no_text() -> None:
    run = (await runs())["hybrid-rerank"][0]

    record = result_record(run)

    assert record["config"] == "hybrid-rerank"
    assert record["question_id"] == "q-in"
    assert record["category"] == "factual"
    assert record["chunk_ids"] == ["c1", "c2"]
    assert record["anchors_hit"] == [[0], []]
    assert record["rerank_scores"] == [1.0, 0.0]
    assert record["store_scores"] == [1.0, 0.5]
    assert record["top_score"] == 1.0
    assert record["filter"] is None
    assert isinstance(record["scores"], dict)
    assert record["scores"][Metric.RECALL_AT_5.value] == 1.0
    assert record["timings"] == {
        "dense_embed_ms": 1.0,
        "sparse_embed_ms": 1.0,
        "search_ms": 2.0,
        "rerank_ms": 5.0,
        "total_ms": 10.0,
    }
    dumped = json.dumps(record)
    assert "CANARY" not in dumped


async def test_an_out_of_scope_result_has_no_scores_and_its_kind() -> None:
    run = (await runs())["hybrid-rerank"][1]

    record = result_record(run)

    assert record["scores"] is None
    assert record["oos_kind"] == "unrelated"
    assert record["top_score"] is None


async def outcome() -> RunOutcome:
    by_config = await runs()
    summaries = tuple(summarize_config(c, by_config[c.name], seed=1, resamples=50) for c in CONFIGS)
    return RunOutcome(
        configs=CONFIGS,
        summaries=summaries,
        selection=select_best(summaries),
        preflight=preflight(GOLDEN, [chunk("c1", CHUNK_TEXT)]),
        runs=tuple(r for c in CONFIGS for r in by_config[c.name]),
    )


async def test_the_summary_records_golden_index_git_and_every_resolved_config() -> None:
    record = summary_record(HEADER, await outcome())

    assert record["golden"] == {
        "version": "1.0.0",
        "sha256": GOLDEN_SHA,
        "split": "dev",
        "questions": 2,
    }
    assert record["index"] == {
        "alias": "hpdocs_chunks",
        "collection": "hpdocs_chunks_v1_20261113t100000z",
        "build_id": "20261113t100000z",
        "chunk_count": 287,
        "chunking_version": "v1",
        "bm25_avg_len": 107.25,
        "bm25_avg_len_auto": True,
    }
    assert record["git_sha"] == HEADER.git_sha
    assert record["created_at"] == "2026-11-13T15:30:05+00:00"
    assert record["bootstrap"] == {"seed": 20261113, "resamples": 50}
    assert record["warm_up_retrievals"] == 1
    assert record["reranker_warm_up_pairs"] == 20
    assert record["hardware"] == HEADER.hardware
    assert record["models"] == HEADER.models
    assert record["scored_over"] == "the candidate list (top_k = rerank_candidates)"
    configs = record["configs"]
    assert isinstance(configs, list)
    first = configs[0]
    assert first["name"] == "hybrid-rerank"
    assert first["reranker"] == RERANKER
    assert first["config"]["mode"] == "hybrid"
    assert first["config"]["rrf_k"] == 60
    assert first["config"]["dense_query_prefix"] == ""
    assert configs[1]["reranker"] is None  # no rerank: the model is not part of it
    assert first["metrics"]["recall@5"] == {"n": 1, "mean": 1.0, "low": 1.0, "high": 1.0}
    assert first["by_category"]["factual"]["n"] == 1
    assert first["latency_ms"]["total"] == {"p50": 10.0, "p95": 10.0}
    assert first["calibration"]["in_scope_n"] == 1
    assert configs[1]["calibration"] is None
    assert configs[1]["latency_ms"]["rerank"] is None
    # Equal on every criterion: the first in matrix order wins.
    assert record["selection"] == {"best": "hybrid-rerank", "tied": ["hybrid-rerank", "sparse"]}
    assert record["preflight"] == {"chunks": 1, "anchors": 1, "problems": []}
    assert "CANARY" not in json.dumps(record)


async def test_the_markdown_summary_shows_the_header_and_each_config() -> None:
    text = render_markdown(HEADER, await outcome())

    for expected in (
        "1.0.0",
        GOLDEN_SHA,
        "20261113t100000z",
        HEADER.git_sha,
        "hybrid-rerank",
        "sparse",
        "| factual |",
        "Selected: `hybrid-rerank`",
        "rev-rerank",
        "BM25 avg_len 107.25 (auto)",
        "Warm-up: each reranker on load with 20 pairs at its 512-token limit, then 1 "
        "uncounted retrieval per configuration before timing",
    ):
        assert expected in text, expected
    assert "CANARY" not in text


async def test_writing_a_run_creates_the_three_files_in_its_directory(
    tmp_path: pathlib.Path,
) -> None:
    path = write_run(tmp_path, HEADER, await outcome())

    assert path == tmp_path / run_dir_name(HEADER)
    lines = (path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["config"] for line in lines] == [
        "hybrid-rerank",
        "hybrid-rerank",
        "sparse",
        "sparse",
    ]
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert summary["golden"]["sha256"] == GOLDEN_SHA
    assert (path / "summary.md").read_text(encoding="utf-8").startswith("# Retrieval benchmark")


async def test_a_run_never_overwrites_an_earlier_one(tmp_path: pathlib.Path) -> None:
    done = await outcome()
    write_run(tmp_path, HEADER, done)

    with pytest.raises(FileExistsError):
        write_run(tmp_path, HEADER, done)
