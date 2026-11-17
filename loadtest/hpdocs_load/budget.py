"""Profile B's spend guard (ADR 0013): the harness stops the run before the provider
bill can pass the cap.

Rulings: cap $3.00, hard stop at 90 questions, every provider call attempt counted. The
backend runs Profile B with LLM_MAX_RETRIES=0, so one call is one attempt. A call's
usage is known only after it ends, so each call is charged its worst case before the
question is sent: the Phase 5 token bounds at the dated prices (eval/answer-matrix.toml).
A first question is one answer call; a follow-up adds one rewrite call. A question that
then fails stays charged. A refused question is not sent, and from then on nothing is;
the run's shape then ends the run on Locust's normal stop path (`until_spent`). Calling
`runner.quit()` from a user instead fires Locust's stop event twice.
"""

from collections.abc import Callable
from dataclasses import dataclass

_MICRO = 1_000_000

# eval/answer-matrix.toml [worst_case] at its 2026-11-03 prices, in micro-USD (a test
# recomputes both from the file):
#   answer, claude-sonnet-5-5: 12,000 input x $2/Mtok + 2,048 output x $10/Mtok
#   rewrite, claude-haiku-5-5: 4,000 input x $0.1/Mtok + 100 output x $0.5/Mtok
ANSWER_CALL_MICRO_USD = 12_000 * 2 + 2_048 * 10
REWRITE_CALL_MICRO_USD = (4_000 * 1 + 100 * 5) // 10


@dataclass(frozen=True)
class Budget:
    max_questions: int
    cap_micro_usd: int
    answer_call_micro_usd: int
    rewrite_call_micro_usd: int


PROFILE_B = Budget(
    max_questions=90,
    cap_micro_usd=3 * _MICRO,
    answer_call_micro_usd=ANSWER_CALL_MICRO_USD,
    rewrite_call_micro_usd=REWRITE_CALL_MICRO_USD,
)


class SpendGuard:
    """Integer micro-dollars, so a charge landing on the cap is exact.

    Locust users are gevent greenlets and `admit` never yields, so check and charge
    happen as one step.
    """

    def __init__(self, budget: Budget) -> None:
        self._budget = budget
        self.questions = 0
        self.calls = 0
        self.charged_micro_usd = 0
        self.stopped = False

    def admit(self, *, follow_up: bool) -> bool:
        if self.stopped:
            return False
        budget = self._budget
        charge = budget.answer_call_micro_usd + (budget.rewrite_call_micro_usd if follow_up else 0)
        if (
            self.questions >= budget.max_questions
            or self.charged_micro_usd + charge > budget.cap_micro_usd
        ):
            self.stopped = True
            return False
        self.questions += 1
        self.calls += 2 if follow_up else 1
        self.charged_micro_usd += charge
        return True

    def summary(self) -> dict[str, object]:
        return {
            "questions": self.questions,
            "provider_calls": self.calls,
            "charged_worst_case_usd": self.charged_micro_usd / _MICRO,
            "cap_usd": self._budget.cap_micro_usd / _MICRO,
            "max_questions": self._budget.max_questions,
            "stopped": self.stopped,
        }


def admit_all(*, follow_up: bool) -> bool:  # noqa: ARG001 - same signature as admit
    return True


def admission(*, enabled: bool, guard: SpendGuard) -> Callable[..., bool]:
    """Profile A (stub) runs unguarded; Profile B never runs without the guard."""
    return guard.admit if enabled else admit_all


def until_spent(result: tuple[int, float] | None, guard: SpendGuard) -> tuple[int, float] | None:
    """The shape's tick: None ends the run, as at the end of the last step."""
    return None if guard.stopped else result
