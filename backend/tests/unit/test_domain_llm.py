"""The language-model port: request rules and the fixed-message errors."""

import pytest

from app.domain.chat import Role
from app.domain.llm import (
    GenerationRequest,
    LLMError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    PromptMessage,
    Purpose,
)

USER = PromptMessage(Role.USER, "Question?")
ASSISTANT = PromptMessage(Role.ASSISTANT, "Answer.")


def request(**overrides: object) -> GenerationRequest:
    fields: dict[str, object] = {
        "purpose": Purpose.ANSWER,
        "model": "model-id",
        "system": "Rules.",
        "messages": (USER,),
        "max_tokens": 100,
    }
    fields.update(overrides)
    return GenerationRequest(**fields)  # type: ignore[arg-type]  # fields typed per test


def test_a_multi_turn_request_alternates_and_ends_with_the_user() -> None:
    built = request(messages=(USER, ASSISTANT, USER), effort="low")

    assert [m.role for m in built.messages] == [Role.USER, Role.ASSISTANT, Role.USER]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"messages": ()}, "a request needs at least one message"),
        ({"messages": (ASSISTANT, USER)}, "the first message must be the user's"),
        ({"messages": (USER, ASSISTANT)}, "the last message must be the user's"),
        ({"messages": (USER, USER)}, "roles must alternate"),
        ({"model": ""}, "a request names a model"),
        ({"max_tokens": 0}, "max_tokens must be at least 1"),
    ],
)
def test_request_rules(overrides: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        request(**overrides)


def test_a_prompt_message_is_never_empty() -> None:
    with pytest.raises(ValueError, match="a prompt message has text"):
        PromptMessage(Role.USER, "")


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (LLMUnavailableError, "llm_unavailable"),
        (LLMRequestError, "llm_error"),
        (LLMTimeoutError, "llm_timeout"),
    ],
)
def test_each_failure_has_a_code_and_a_fixed_message(error: type[LLMError], code: str) -> None:
    raised = error()

    assert isinstance(raised, LLMError)
    assert raised.code == code
    assert raised.message
    assert str(raised) == raised.message


def test_the_failure_messages_differ_so_clients_can_tell_them_apart() -> None:
    errors = (LLMUnavailableError(), LLMRequestError(), LLMTimeoutError())
    assert len({e.message for e in errors}) == len(errors)
