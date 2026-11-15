"""The judge's `BatchPort` over the Anthropic Message Batches API.

As in the chat adapter, the client gets the key and the public API URL explicitly, so no
ambient credential or endpoint is used. Only the batch id, the processing status and
each result's custom id, stop reason, first text block and usage are read.

Billing follows the API: a request that errored, was canceled or expired never ran and
costs nothing. A response that ran is billed whatever its stop reason; any stop other
than `end_turn` (a refusal, `max_tokens`, anything unexpected) is a judge error named
by that stop reason, never a verdict. No `cache_control` is sent, so cache fields stay
empty; if the API ever reports them, they are counted as input, so the ledger never
misses tokens.
"""

from collections.abc import Mapping
from typing import cast

import httpx2
from anthropic import AsyncAnthropic
from anthropic.types import Message
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchIndividualResponse
from anthropic.types.messages.batch_create_params import Request
from pydantic import SecretStr

from app.evaluation.answers.judge_run import BatchOutcome
from app.infra.llm_anthropic import API_URL

ENDED = "ended"
NORMAL_STOP = "end_turn"
MAX_RETRIES = 2


class AnthropicBatch:
    def __init__(self, client: AsyncAnthropic) -> None:
        self._client = client

    async def submit(self, requests: Mapping[str, Mapping[str, object]]) -> str:
        batch = await self._client.messages.batches.create(
            requests=[
                Request(
                    custom_id=cid,
                    # The body is built and tested by `judge.request_params`.
                    params=cast(MessageCreateParamsNonStreaming, dict(params)),
                )
                for cid, params in requests.items()
            ]
        )
        return batch.id

    async def ended(self, batch_id: str) -> bool:
        batch = await self._client.messages.batches.retrieve(batch_id)
        return batch.processing_status == ENDED

    async def results(self, batch_id: str) -> list[BatchOutcome]:
        decoder = await self._client.messages.batches.results(batch_id)
        return [_outcome(response) async for response in decoder]


def _outcome(response: MessageBatchIndividualResponse) -> BatchOutcome:
    result = response.result
    if result.type != "succeeded":
        return BatchOutcome(response.custom_id, None, 0, 0, result.type)
    message = result.message
    input_tokens, output_tokens = _usage(message)
    if message.stop_reason != NORMAL_STOP:
        return BatchOutcome(
            response.custom_id, None, input_tokens, output_tokens, str(message.stop_reason)
        )
    text = next((block.text for block in message.content if block.type == "text"), None)
    return BatchOutcome(response.custom_id, text, input_tokens, output_tokens, None)


def _usage(message: Message) -> tuple[int, int]:
    usage = message.usage
    cached = (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
    return usage.input_tokens + cached, usage.output_tokens


def open_batch(api_key: SecretStr, http_client: httpx2.AsyncClient | None = None) -> AnthropicBatch:
    """The SDK retries transient failures of these small control calls; the results stay
    on the API for 29 days, so a failed poll loses nothing.
    """
    client = AsyncAnthropic(
        api_key=api_key.get_secret_value(),
        base_url=API_URL,
        max_retries=MAX_RETRIES,
        http_client=http_client,
    )
    return AnthropicBatch(client)
