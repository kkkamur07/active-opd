"""Answer extraction and verification for math rollouts.

The verifier is a gate for rollout selection, not a training signal.  Three
properties matter for Active OPD and are enforced here:

1.  A rollout is only ever labelled ``wrong`` when the comparator positively
    decided the answers differ.  Anything undecidable is ``unverified`` and is
    retained by *no* selector.  Defaulting undecidable cases to ``wrong`` would
    feed them straight into the Active OPD pool, which is exactly backwards.
2.  A response that ran out of generation budget is ``truncated``, not
    ``malformed``.  Truncation is a length statistic, not a reasoning error,
    and conflating the two biases selection toward short problems.
3.  Extraction only ever reads the post-``</think>`` region when the model
    closed its reasoning block, so mid-trace prose such as "so the answer is
    ..." can never be scraped as a final answer.

Equivalence is decided by ``math_verify`` when it is installed (both sides are
wrapped in ``\\boxed{}`` first, which is the form its parser handles best).  A
conservative string comparator is used as a fallback and answers it cannot
decide become ``unverified`` rather than ``wrong``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

AnswerStatus = Literal["ok", "missing", "malformed", "truncated"]
VerificationOutcome = Literal[
    "correct", "wrong", "malformed", "truncated", "unverified", "skipped"
]

#: Outcomes that represent a decided comparison against a reference answer.
DECIDED_OUTCOMES: tuple[VerificationOutcome, ...] = ("correct", "wrong")

#: Outcomes that must never be used as training signal by any selector.
UNUSABLE_OUTCOMES: tuple[VerificationOutcome, ...] = (
    "malformed",
    "truncated",
    "unverified",
    "skipped",
)

THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"


@dataclass(frozen=True)
class AnswerExtraction:
    """The parsed answer and enough metadata to audit the verifier."""

    answer: str | None
    status: AnswerStatus
    source: str | None = None
    raw: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    """Result of comparing one rollout answer with a reference answer."""

    outcome: VerificationOutcome
    predicted: AnswerExtraction
    reference: str | None
    normalized_predicted: str | None
    normalized_reference: str | None
    comparator: str | None = None

    @property
    def is_retained_for_active_opd(self) -> bool:
        """Whether this result belongs in the verified-wrong training pool."""

        return self.outcome == "wrong"

    @property
    def is_decided(self) -> bool:
        """Whether the comparator positively decided correct vs wrong."""

        return self.outcome in DECIDED_OUTCOMES


_BOXED_MARKER = re.compile(r"\\(?:boxed|fbox)\s*(?=\{)")
_ANSWER_MARKER = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is)\s*(?:is|equals?|=|:)?\s*",
    flags=re.IGNORECASE,
)
_HASH_MARKER = re.compile(r"(?:^|\n)\s*####\s*(\S[^\n]{0,80})\s*$", flags=re.MULTILINE)


def _balanced_braced_content(text: str, opening_index: int) -> tuple[str | None, bool]:
    """Return content inside a brace and whether the brace pair is balanced."""

    if opening_index >= len(text) or text[opening_index] != "{":
        return None, False
    depth = 0
    content_start = opening_index + 1
    for index in range(opening_index, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index], True
            if depth < 0:
                return None, False
    return None, False


def strip_reasoning_block(text: str) -> tuple[str, bool]:
    """Return the answer region and whether reasoning was left unterminated.

    Qwen3 emits ``<think> ... </think>`` as ordinary (non-special) tokens, so
    the boundary survives decoding.  When the block is closed we read only what
    follows it; when it was opened and never closed the response ran out of
    budget mid-thought and has no answer region at all.
    """

    close_index = text.rfind(THINK_CLOSE)
    if close_index >= 0:
        return text[close_index + len(THINK_CLOSE) :], False
    if THINK_OPEN in text:
        return "", True
    return text, False


def normalize_answer(answer: str) -> str:
    """Normalize harmless formatting differences for string comparison.

    This is only used for reporting and for the fallback comparator.  It
    intentionally performs no symbolic algebra; ``math_verify`` does that when
    available, and undecidable cases become ``unverified`` rather than guessed.
    """

    normalized = answer.strip()
    normalized = normalized.replace("\u2212", "-")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\\d?frac", r"\\frac", normalized)
    normalized = re.sub(r"\\(?:text|mathrm|mbox|textbf)\s*\{([^{}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    # Only strip trailing punctuation; a leading '.' is part of the number.
    normalized = normalized.rstrip(".,;:")
    # Collapse thousands separators only for a single well-formed integer.
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", normalized):
        normalized = normalized.replace(",", "")
    return normalized


_HEADING_WORDS = frozenset(
    {"final answer", "answer", "solution", "result", "conclusion", "step"}
)


def _looks_like_a_value(candidate: str) -> bool:
    """Reject section headings masquerading as answers.

    ``####`` is both the GSM8K answer convention and a markdown H4, so
    ``#### Final Answer`` would otherwise be extracted as the answer itself.
    A value contains a digit or LaTeX; a heading is bare prose.
    """

    lowered = candidate.strip().lower().rstrip(":")
    if lowered in _HEADING_WORDS:
        return False
    return any(character.isdigit() for character in candidate) or "\\" in candidate


def _clean_candidate(candidate: str) -> str:
    candidate = candidate.strip()
    candidate = re.sub(r"^(?:is|equals?)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip()
    while candidate and candidate[-1] in ".,;:":
        candidate = candidate[:-1].rstrip()
    return candidate


def extract_final_answer(
    text: str | None,
    *,
    require_closed_reasoning: bool = True,
) -> AnswerExtraction:
    """Extract the most explicit final answer from a model response.

    ``\\boxed{...}``, ``#### ...`` and ``Final answer: ...`` are supported, in
    that order of authority.  When ``require_closed_reasoning`` is set (the
    default) an unterminated ``<think>`` block yields ``truncated`` and no
    answer is scraped from the reasoning text.
    """

    if text is None or not text.strip():
        return AnswerExtraction(None, "missing")

    region, unterminated = strip_reasoning_block(text)
    if unterminated and require_closed_reasoning:
        return AnswerExtraction(None, "truncated", "think", None)
    if not require_closed_reasoning and not region.strip():
        region = text

    # 1. \boxed{...} / \fbox{...}: prefer the last well-formed one, but fall
    #    back to earlier candidates instead of discarding them.
    malformed_boxed: AnswerExtraction | None = None
    for match in reversed(list(_BOXED_MARKER.finditer(region))):
        opening = match.end()
        candidate, balanced = _balanced_braced_content(region, opening)
        if not balanced:
            if malformed_boxed is None:
                malformed_boxed = AnswerExtraction(
                    None, "malformed", "boxed", region[match.start() :][:120]
                )
            continue
        cleaned = _clean_candidate(candidate or "")
        if not cleaned:
            if malformed_boxed is None:
                malformed_boxed = AnswerExtraction(None, "malformed", "boxed", candidate)
            continue
        return AnswerExtraction(cleaned, "ok", "boxed", cleaned)
    if malformed_boxed is not None:
        return malformed_boxed

    # 2. '#### answer', anchored to a whole short line. '####' is also a
    #    markdown H4, so a candidate that looks like a heading ("Final Answer")
    #    rather than a value is skipped.
    for match in reversed(list(_HASH_MARKER.finditer(region))):
        candidate = _clean_candidate(match.group(1))
        if candidate and _looks_like_a_value(candidate):
            return AnswerExtraction(candidate, "ok", "hash", candidate)

    # 3. Prose, restricted to the last non-empty line of the answer region so a
    #    mid-derivation "the answer is ..." cannot be picked up.
    lines = [line for line in region.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        prose = list(_ANSWER_MARKER.finditer(line))
        if not prose:
            continue
        candidate = _clean_candidate(line[prose[-1].end() :])
        if candidate:
            return AnswerExtraction(candidate, "ok", "prose", candidate)
    return AnswerExtraction(None, "missing")


@lru_cache(maxsize=1)
def _math_verify():
    """Import ``math_verify`` lazily; return ``None`` when it is unavailable."""

    try:
        from math_verify import parse, verify
    except ImportError:  # pragma: no cover - depends on environment
        return None
    return parse, verify


@lru_cache(maxsize=8192)
def _math_verify_equal(predicted: str, reference: str) -> bool | None:
    """Return True/False from ``math_verify``, or None when it cannot decide."""

    tools = _math_verify()
    if tools is None:
        return None
    parse, verify = tools
    try:
        # Wrapping in \boxed{} is the form math_verify's extractor handles most
        # reliably; bare LaTeX fragments frequently fail to parse at all.
        gold = parse(r"\boxed{" + reference + "}")
        pred = parse(r"\boxed{" + predicted + "}")
    except Exception:  # pragma: no cover - parser is third-party
        return None
    if not gold or not pred:
        return None
    try:
        return bool(verify(gold, pred))
    except Exception:  # pragma: no cover - parser is third-party
        return None


def _string_equal(predicted: str, reference: str) -> bool | None:
    """Conservative fallback comparator.

    Returns ``True`` only on an exact normalized match, ``False`` only when
    both sides are plain numbers that differ, and ``None`` (undecidable)
    otherwise so the caller can mark the rollout ``unverified``.
    """

    if predicted == reference:
        return True
    number = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)")
    if number.fullmatch(predicted) and number.fullmatch(reference):
        try:
            return float(predicted) == float(reference)
        except ValueError:  # pragma: no cover - guarded by the regex
            return None
    return None


def answers_equivalent(predicted: str, reference: str) -> tuple[bool | None, str]:
    """Compare two answer strings, reporting which comparator decided."""

    decision = _math_verify_equal(predicted, reference)
    if decision is not None:
        return decision, "math_verify"
    normalized_predicted = normalize_answer(predicted)
    normalized_reference = normalize_answer(reference)
    decision = _string_equal(normalized_predicted, normalized_reference)
    return decision, "string"


def verify_exact_answer(
    rollout_text: str | None,
    reference_answer: str | None,
    *,
    truncated: bool | None = None,
    require_closed_reasoning: bool = True,
) -> VerificationResult:
    """Apply the verification gate without conflating it with the loss.

    ``truncated`` lets the caller pass the generator's own finish reason (the
    rollout hit ``max_new_tokens`` without emitting EOS), which is more
    reliable than inspecting the decoded text.
    """

    predicted = extract_final_answer(
        rollout_text,
        require_closed_reasoning=require_closed_reasoning,
    )
    normalized_predicted = (
        normalize_answer(predicted.answer) if predicted.answer is not None else None
    )

    if reference_answer is None or not str(reference_answer).strip():
        return VerificationResult(
            "skipped", predicted, reference_answer, normalized_predicted, None
        )

    normalized_reference = normalize_answer(str(reference_answer))

    # Truncation is a budget outcome, not a reasoning error. Report it even
    # when an answer happened to be parseable from a partial trace.
    if truncated or predicted.status == "truncated":
        return VerificationResult(
            "truncated",
            predicted,
            reference_answer,
            normalized_predicted,
            normalized_reference,
        )
    if predicted.status != "ok" or predicted.answer is None:
        return VerificationResult(
            "malformed",
            predicted,
            reference_answer,
            normalized_predicted,
            normalized_reference,
        )

    decision, comparator = answers_equivalent(predicted.answer, str(reference_answer))
    if decision is None:
        outcome: VerificationOutcome = "unverified"
    else:
        outcome = "correct" if decision else "wrong"
    return VerificationResult(
        outcome,
        predicted,
        reference_answer,
        normalized_predicted,
        normalized_reference,
        comparator,
    )


extract_answer = extract_final_answer


__all__ = [
    "DECIDED_OUTCOMES",
    "UNUSABLE_OUTCOMES",
    "AnswerExtraction",
    "AnswerStatus",
    "VerificationOutcome",
    "VerificationResult",
    "answers_equivalent",
    "extract_answer",
    "extract_final_answer",
    "normalize_answer",
    "strip_reasoning_block",
    "verify_exact_answer",
]
