"""The Batch API adapter over a fake Message Batches API (httpx2.MockTransport): the
batch it creates, how it reads the processing status, and how each result becomes a
`BatchOutcome` (billed or not, verdict text or judge error).

No test reaches the network; the key is a placeholder.
"""

import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from app.evaluation.answers.batch_api import AnthropicBatch, open_batch
from app.evaluation.answers.judge_run import BatchOutcome
from app.infra.llm_anthropic import API_URL

KEY = "placeholder-not-a-key"  # not a real key
BATCH_ID = "msgbatch_test01"
RESULTS_URL = f"{API_URL}/v1/messages/batches/{BATCH_ID}/results"
Json = dict[str, Any]  # Any: JSON bodies


def batch_json(status: str) -> Json:
    return {
        "id": BATCH_ID,
        "type": "message_batch",
        "processing_status": status,
        "request_counts": {
            "processing": 0,
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "created_at": "2026-11-15T10:00:00Z",
        "expires_at": "2026-11-16T10:00:00Z",
        "ended_at": None,
        "archived_at": None,
        "cancel_initiated_at": None,
        "results_url": RESULTS_URL if status == "ended" else None,
    }


def message(stop_reason: str, content: list[Json], usage: Json | None = None) -> Json:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 1200, "output_tokens": 300},
    }


def succeeded(cid: str, msg: Json) -> Json:
    return {"custom_id": cid, "result": {"type": "succeeded", "message": msg}}


def unbilled(cid: str, kind: str) -> Json:
    result: Json = {"type": kind}
    if kind == "errored":
        result["error"] = {
            "type": "error",
            "error": {"type": "api_error", "message": "upstream detail"},
        }
    return {"custom_id": cid, "result": result}


class FakeBatchAPI:
    def __init__(self, status: str = "ended", results: list[Json] | None = None) -> None:
        self.status = status
        self.results = results or []
        self.requests: list[httpx2.Request] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/v1/messages/batches":
            return httpx2.Response(200, json=batch_json("in_progress"))
        if request.method == "GET" and path == f"/v1/messages/batches/{BATCH_ID}":
            return httpx2.Response(200, json=batch_json(self.status))
        if request.method == "GET" and str(request.url) == RESULTS_URL:
            lines = "".join(json.dumps(line) + "\n" for line in self.results)
            return httpx2.Response(200, content=lines.encode())
        return httpx2.Response(404, json={"type": "error", "error": {"type": "not_found_error"}})


def adapter(api: FakeBatchAPI) -> AnthropicBatch:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(api.handle))
    return open_batch(SecretStr(KEY), http_client=client)


def text(value: str) -> Json:
    return {"type": "text", "text": value}


async def test_submit_sends_every_request_with_its_custom_id_and_returns_the_batch_id() -> None:
    api = FakeBatchAPI()
    params = {"model": "claude-opus-5-5", "max_tokens": 10, "messages": []}
    batch_id = await adapter(api).submit({"a--p1": params, "b--p2": params})

    assert batch_id == BATCH_ID
    sent = json.loads(api.requests[0].content)
    assert sent == {
        "requests": [
            {"custom_id": "a--p1", "params": params},
            {"custom_id": "b--p2", "params": params},
        ]
    }
    assert api.requests[0].url.host == "api.anthropic.com"
    assert api.requests[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize(
    ("status", "ended"), [("in_progress", False), ("canceling", False), ("ended", True)]
)
async def test_ended_is_true_only_once_processing_has_ended(status: str, ended: bool) -> None:
    assert await adapter(FakeBatchAPI(status)).ended(BATCH_ID) is ended


async def test_a_normal_response_gives_its_first_text_block_and_measured_usage() -> None:
    msg = message(
        "end_turn",
        [{"type": "thinking", "thinking": "", "signature": "s"}, text('{"v": 1}'), text("x")],
    )
    api = FakeBatchAPI(results=[succeeded("a--p1", msg)])

    assert await adapter(api).results(BATCH_ID) == [
        BatchOutcome("a--p1", '{"v": 1}', 1200, 300, None)
    ]


async def test_cache_tokens_count_as_input() -> None:
    usage = {
        "input_tokens": 100,
        "output_tokens": 30,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 5,
    }
    api = FakeBatchAPI(results=[succeeded("a--p1", message("end_turn", [text("{}")], usage))])

    [outcome] = await adapter(api).results(BATCH_ID)
    assert (outcome.input_tokens, outcome.output_tokens) == (125, 30)


async def test_a_normal_response_without_text_has_no_text_and_no_error() -> None:
    api = FakeBatchAPI(results=[succeeded("a--p1", message("end_turn", []))])

    assert await adapter(api).results(BATCH_ID) == [BatchOutcome("a--p1", None, 1200, 300, None)]


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "pause_turn"])
async def test_any_other_stop_is_a_billed_judge_error_named_by_its_stop_reason(
    stop_reason: str,
) -> None:
    api = FakeBatchAPI(results=[succeeded("a--p1", message(stop_reason, [text('{"v": 1}')]))])

    assert await adapter(api).results(BATCH_ID) == [
        BatchOutcome("a--p1", None, 1200, 300, stop_reason)
    ]


@pytest.mark.parametrize("kind", ["errored", "canceled", "expired"])
async def test_a_request_that_never_ran_is_an_unbilled_error(kind: str) -> None:
    api = FakeBatchAPI(results=[unbilled("a--p1", kind)])

    assert await adapter(api).results(BATCH_ID) == [BatchOutcome("a--p1", None, 0, 0, kind)]


async def test_every_result_line_is_returned_keyed_by_custom_id() -> None:
    api = FakeBatchAPI(
        results=[
            unbilled("b--p1", "expired"),
            succeeded("a--p1", message("end_turn", [text("{}")])),
        ]
    )

    outcomes = await adapter(api).results(BATCH_ID)
    assert [o.custom_id for o in outcomes] == ["b--p1", "a--p1"]
