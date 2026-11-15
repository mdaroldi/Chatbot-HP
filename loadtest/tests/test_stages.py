"""Per-stage timings come from the backend's JSON logs, numbers only: no content, no
chunk IDs, no conversation IDs reach the run directory."""

import json

import pytest

from hpdocs_load.stages import BACKEND_CONTAINER, StageRow, logs_command, parse_stage_logs


def line(message: str, **fields: object) -> str:
    return json.dumps(
        {
            "timestamp": "2026-11-15T12:00:01.500+00:00",
            "message": message,
            "request_id": "r1",
            "conversation_id": "secret-ish",
            "chunk_ids": ["c"],
            **fields,
        }
    )


def test_retrieval_and_chat_lines_keep_their_timings_only() -> None:
    text = "\n".join(
        [
            "Started server process [7]",
            line("request completed", duration_ms=3.0),
            line(
                "retrieval done",
                dense_embed_ms=50.0,
                sparse_embed_ms=1.0,
                search_ms=20,
                rerank_ms=1800.5,
                total_ms=1880.0,
                query_chars=40,
            ),
            line("chat answer", ttft_ms=2100.0, generation_ms=3000.0, total_ms=5200.0),
            "",
        ]
    )

    assert parse_stage_logs(text) == [
        StageRow(
            at=1794744001.5,
            request_id="r1",
            kind="retrieval",
            timings={
                "dense_embed_ms": 50.0,
                "sparse_embed_ms": 1.0,
                "search_ms": 20.0,
                "rerank_ms": 1800.5,
                "total_ms": 1880.0,
            },
        ),
        StageRow(
            at=1794744001.5,
            request_id="r1",
            kind="chat",
            timings={"ttft_ms": 2100.0, "generation_ms": 3000.0, "total_ms": 5200.0},
        ),
    ]


def test_a_failed_answer_keeps_the_timings_it_has() -> None:
    text = line("chat answer", ttft_ms=None, generation_ms=None, total_ms=900.0)

    assert parse_stage_logs(text)[0].timings == {"total_ms": 900.0}


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"ttft_ms": "fast", "generation_ms": 1.0, "total_ms": 1.0}, id="text"),
        pytest.param({"ttft_ms": True, "generation_ms": 1.0, "total_ms": 1.0}, id="bool"),
        pytest.param({"generation_ms": 1.0, "total_ms": 1.0}, id="missing"),
    ],
)
def test_a_changed_log_contract_raises(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="ttft_ms"):
        parse_stage_logs(line("chat answer", **fields))


def test_logs_are_read_from_the_load_backend_since_the_run_start() -> None:
    assert BACKEND_CONTAINER == "hpdocs-load-backend-1"
    assert logs_command(1794744001.9) == (
        "docker",
        "logs",
        "--since",
        "1794744001",
        "hpdocs-load-backend-1",
    )
