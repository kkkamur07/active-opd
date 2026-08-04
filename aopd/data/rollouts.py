"""Rollout records and verification summaries shared by training/evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .answers import (
    UNUSABLE_OUTCOMES,
    VerificationOutcome,
    VerificationResult,
    verify_exact_answer,
)

#: Every outcome the verifier can emit, in reporting order.
ALL_OUTCOMES: tuple[VerificationOutcome, ...] = (
    "correct",
    "wrong",
    "malformed",
    "truncated",
    "unverified",
    "skipped",
)


@dataclass
class Rollout:
    """One student-generated response and its optional tokenized form."""

    prompt: str
    response: str
    reference_answer: str | None = None
    rollout_id: str | None = None
    prompt_length: int | None = None
    input_ids: Any | None = None
    attention_mask: Any | None = None
    #: True when generation hit ``max_new_tokens`` without emitting EOS. This
    #: comes from the generator, not from inspecting the text, so a trace that
    #: closed ``</think>`` and was then cut off is still recognised.
    truncated: bool = False
    #: Number of generated (non-prompt, non-pad) tokens.
    response_length: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    verification: VerificationResult | None = None

    @property
    def text(self) -> str:
        return self.response

    def verify(self, reference_answer: str | None = None) -> VerificationResult:
        """Attach and return the verification result for this rollout.

        ``reference_answer`` is used for this comparison only; it does not
        overwrite the rollout's stored reference.
        """

        reference = (
            reference_answer if reference_answer is not None else self.reference_answer
        )
        self.verification = verify_exact_answer(
            self.response,
            reference,
            truncated=self.truncated,
        )
        return self.verification


@dataclass(frozen=True)
class VerificationSummary:
    """Auditable counts for a collection of verification outcomes."""

    counts: Mapping[VerificationOutcome, int]
    total: int
    retained: int

    def count(self, outcome: VerificationOutcome) -> int:
        return int(self.counts.get(outcome, 0))

    @property
    def correct(self) -> int:
        return self.count("correct")

    @property
    def wrong(self) -> int:
        return self.count("wrong")

    @property
    def malformed(self) -> int:
        return self.count("malformed")

    @property
    def truncated(self) -> int:
        return self.count("truncated")

    @property
    def unverified(self) -> int:
        return self.count("unverified")

    @property
    def skipped(self) -> int:
        return self.count("skipped")

    @property
    def retained_rate(self) -> float:
        return self.retained / self.total if self.total else 0.0

    @property
    def decided(self) -> int:
        """Rollouts the verifier could actually grade."""

        return self.correct + self.wrong

    @property
    def usable_rate(self) -> float:
        """Fraction of rollouts that produced a decided verdict.

        A low value means the generation budget or the extractor is the binding
        constraint, not the student's reasoning -- worth checking before
        reading anything into a retention rate.
        """

        return self.decided / self.total if self.total else 0.0


def summarize_verification(
    rollouts: Iterable[Rollout],
    *,
    retain_outcomes: tuple[VerificationOutcome, ...] = ("wrong",),
    reference_answer: str | None = None,
) -> tuple[list[Rollout], VerificationSummary]:
    """Verify rollouts, retain only requested outcomes, and count every result."""

    unusable = set(retain_outcomes) & set(UNUSABLE_OUTCOMES)
    if unusable:
        raise ValueError(
            f"retain_outcomes must not include {sorted(unusable)}: these rollouts "
            "have no verified answer and must never become training signal."
        )
    retained: list[Rollout] = []
    counts: Counter[VerificationOutcome] = Counter()
    total = 0
    for rollout in rollouts:
        result = rollout.verify(reference_answer)
        counts[result.outcome] += 1
        total += 1
        if result.outcome in retain_outcomes:
            retained.append(rollout)
    for outcome in ALL_OUTCOMES:
        counts.setdefault(outcome, 0)
    return retained, VerificationSummary(dict(counts), total, len(retained))


__all__ = [
    "ALL_OUTCOMES",
    "Rollout",
    "VerificationSummary",
    "summarize_verification",
]
