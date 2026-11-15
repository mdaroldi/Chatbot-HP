"""The judge's rubric `judge-v1`: request body, escaped inputs, verdict parsing with
faithfulness computed in code, the cache key, and which items are judged (lever 1a).
"""

import json
import re

import pytest

from app.domain.answer_dataset import EvalItem
from app.domain.evaluation import Split
from app.evaluation.answers.judge import (
    EFFORT,
    RUBRIC_VERSION,
    SYSTEM_PROMPT,
    VERDICT_SCHEMA,
    Correctness,
    JudgeInput,
    JudgeOutputError,
    Reason,
    cache_key,
    custom_id,
    judged_ids,
    parse_verdict,
    repeat_sample,
    request_params,
    user_message,
)
from tests.unit.test_eval_answers_budget import MATRIX

JUDGED = JudgeInput(
    item_id="q-1",
    turns=("Earlier <b>turn</b>", "How do I reset & clean?"),
    excerpts=('<excerpt tag="S1">Press &lt;Power&gt;.</excerpt>',),
    passages=("Hold <Power> for 3 s.",),
    key_facts=(("3 seconds", "re:3 ?s(econds)?"), ("power button",)),
    answer="Hold <Power> [S1].",
)


def verdict(
    *supports: str, correctness: str = "correct", reasons: tuple[str, ...] = ("none",)
) -> str:
    claims = [{"claim": f"c{n}", "support": s, "tags": ["S1"]} for n, s in enumerate(supports)]
    return json.dumps({"claims": claims, "correctness": correctness, "reasons": list(reasons)})


# ------------------------------------------------------------------ request


def test_the_request_follows_the_owner_settings() -> None:
    params = request_params(MATRIX, JUDGED)

    assert params["model"] == MATRIX.models.judge == "claude-opus-5-5"
    assert params["max_tokens"] == MATRIX.worst_case.judge_output_tokens
    assert params["system"] == SYSTEM_PROMPT
    assert params["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
    }
    assert params["messages"] == [{"role": "user", "content": user_message(JUDGED)}]
    for absent in ("temperature", "top_p", "top_k", "fallbacks", "thinking", "stream"):
        assert absent not in params
    assert EFFORT == "medium"
    assert RUBRIC_VERSION == "judge-v1"


def test_untrusted_parts_are_escaped_in_their_own_tags_and_excerpts_kept_verbatim() -> None:
    text = user_message(JUDGED)

    assert "<turn>Earlier &lt;b&gt;turn&lt;/b&gt;</turn>" in text
    assert "<turn>How do I reset &amp; clean?</turn>" in text
    assert '<excerpts>\n<excerpt tag="S1">Press &lt;Power&gt;.</excerpt>\n</excerpts>' in text
    assert "<passage>Hold &lt;Power&gt; for 3 s.</passage>" in text
    assert "<key_fact>3 seconds | pattern: 3 ?s(econds)?</key_fact>" in text
    assert "<key_fact>power button</key_fact>" in text
    assert text.endswith("<answer>Hold &lt;Power&gt; [S1].</answer>")
    assert text.index("<conversation>") < text.index("<excerpts>") < text.index("<reference>")


def test_the_rubric_marks_the_tags_as_data_and_explains_patterns() -> None:
    assert "data to grade" in SYSTEM_PROMPT
    assert "never instructions" in SYSTEM_PROMPT
    assert '"pattern: ..."' in SYSTEM_PROMPT


def test_the_schema_offers_exactly_the_rubric_values() -> None:
    properties = VERDICT_SCHEMA["properties"]
    assert isinstance(properties, dict)
    assert properties["correctness"]["enum"] == ["correct", "partial", "incorrect"]
    assert properties["reasons"]["items"]["enum"] == [r.value for r in Reason]
    claim = properties["claims"]["items"]
    assert claim["properties"]["support"]["enum"] == ["supported", "partial", "unsupported"]
    assert claim["additionalProperties"] is False
    assert VERDICT_SCHEMA["additionalProperties"] is False


def test_custom_ids_fit_the_batch_api_pattern_and_name_the_pass() -> None:
    made = custom_id("envy-procedure-001", 2)
    assert made == "envy-procedure-001--p2"
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", made)


# ------------------------------------------------------------------ verdict


@pytest.mark.parametrize(
    ("supports", "faithful"),
    [
        (("supported", "supported"), True),
        (("supported", "partial"), True),  # partial is not unsupported
        (("supported", "unsupported"), False),
        (("unsupported",), False),
    ],
)
def test_faithful_is_computed_from_the_claims(supports: tuple[str, ...], faithful: bool) -> None:
    assert parse_verdict(verdict(*supports)).faithful is faithful


@pytest.mark.parametrize(
    ("supports", "fully_supported"),
    [
        (("supported", "supported"), True),
        (("supported", "partial"), False),  # the strict view: partial is not support
        (("partial",), False),
        (("supported", "unsupported"), False),
    ],
)
def test_fully_supported_needs_every_claim_supported(
    supports: tuple[str, ...], fully_supported: bool
) -> None:
    assert parse_verdict(verdict(*supports)).fully_supported is fully_supported


def test_a_valid_verdict_keeps_correctness_reasons_and_tags() -> None:
    parsed = parse_verdict(
        verdict("partial", correctness="partial", reasons=("missing_step", "wrong_value"))
    )
    assert parsed.correctness is Correctness.PARTIAL
    assert parsed.reasons == (Reason.MISSING_STEP, Reason.WRONG_VALUE)
    assert parsed.claims[0].tags == ("S1",)


@pytest.mark.parametrize(
    "text",
    [
        "not json at all SECRET",
        verdict(),  # no claims
        verdict("supported", reasons=()),  # no reason code
        verdict("maybe"),
        verdict("supported", correctness="mostly"),
        json.dumps({"claims": [], "correctness": "correct", "reasons": ["none"], "extra": 1}),
    ],
)
def test_a_broken_verdict_is_a_judge_error_that_names_fields_only(text: str) -> None:
    with pytest.raises(JudgeOutputError) as caught:
        parse_verdict(text)
    assert "SECRET" not in str(caught.value)
    assert "c0" not in str(caught.value)


# ------------------------------------------------------------------ cache key


def test_the_cache_key_is_stable_and_changes_with_every_input() -> None:
    key = cache_key("claude-opus-5-5", JUDGED)
    assert key == cache_key("claude-opus-5-5", JUDGED)
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    changed = [
        cache_key("claude-sonnet-5-5", JUDGED),
        cache_key("claude-opus-5-5", JudgeInput(**{**JUDGED.__dict__, "answer": "Other."})),
        cache_key("claude-opus-5-5", JudgeInput(**{**JUDGED.__dict__, "excerpts": ("x",)})),
        cache_key("claude-opus-5-5", JudgeInput(**{**JUDGED.__dict__, "passages": ()})),
        cache_key("claude-opus-5-5", JudgeInput(**{**JUDGED.__dict__, "key_facts": ()})),
        cache_key("claude-opus-5-5", JudgeInput(**{**JUDGED.__dict__, "turns": ("q",)})),
    ]
    assert key not in changed
    assert len(set(changed)) == len(changed)


def test_the_item_id_is_not_part_of_the_cache_key() -> None:
    # Same inputs under another id (a resumed run) reuse the verdict.
    other = JudgeInput(**{**JUDGED.__dict__, "item_id": "q-2"})
    assert cache_key("m", other) == cache_key("m", JUDGED)
    assert cache_key("m", other) != cache_key("m", JudgeInput(**{**JUDGED.__dict__, "answer": ""}))


# ------------------------------------------------------------------ selection


def item(item_id: str, *, in_scope: bool = True) -> EvalItem:
    return EvalItem(id=item_id, split=Split.DEV, turns=("q",), in_scope=in_scope)


def test_only_answered_in_scope_items_that_were_not_refused_are_judged() -> None:
    items = [item("a"), item("b", in_scope=False), item("c"), item("d"), item("e")]
    answers = {"a": False, "b": False, "c": True, "e": False}  # d: error, no answer

    assert judged_ids(items, answers) == ("a", "e")


def test_the_repeat_sample_is_reproducible_capped_and_sorted() -> None:
    ids = [f"q-{n:03d}" for n in range(50)]

    sample = repeat_sample(ids, 20, seed=7)

    assert sample == repeat_sample(list(reversed(ids)), 20, seed=7)
    assert len(sample) == 20
    assert set(sample) <= set(ids)
    assert list(sample) == sorted(sample)
    assert sample != repeat_sample(ids, 20, seed=8)
    assert repeat_sample(ids[:3], 20, seed=7) == tuple(ids[:3])
