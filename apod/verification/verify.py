"""Grade a generation with Hugging Face Math-Verify."""

from __future__ import annotations

from typing import Any

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify as math_verify

# Boxed first, bare expressions second: a trace that followed the prompt and
# wrote \boxed{...} should be read there, not from whatever number came last.
PREDICTION_CONFIG = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]


def parse_prediction(text: str | None) -> list[Any]:
    """Everything math-verify can pull out of a generation, best candidate first.

    An empty list means math-verify found nothing to compare, which is a
    different failure from finding the wrong thing. Parse errors are folded into
    "found nothing" because that is what they mean downstream: no candidate.
    """

    if not text:
        return []
    try:
        return parse(str(text), extraction_config=PREDICTION_CONFIG)
    except Exception:
        return []


def parse_reference(reference: str | None) -> list[Any]:
    """The gold answer is a bare value, so box it before parsing."""

    if reference is None or not str(reference).strip():
        return []
    try:
        return parse(rf"\boxed{{{reference}}}")
    except Exception:
        return []


def grade(text: str | None, reference: str | None) -> dict[str, bool]:
    """Correctness plus the two signals needed to classify a failure.

    ``has_answer`` is loose on purpose: ``ExprExtractionConfig`` will happily
    return the last number in a truncated mid-computation trace, so it says
    "math-verify had something to compare", not "the model committed to an
    answer". ``has_boxed`` is the strict version -- the model actually wrote
    ``\\boxed{...}`` as the prompt asked. Reporting both keeps "no parseable
    answer" from being confused with "wrong answer".
    """

    prediction = parse_prediction(text)
    gold = parse_reference(reference)
    result = {
        "correct": False,
        "has_answer": bool(prediction),
        "has_boxed": bool(text) and "\\boxed" in str(text),
        # False means the GOLD answer failed to parse: every sample of that
        # problem scores 0 for every arm and round, which must be visible as
        # a dataset defect, not silently indistinguishable from wrong answers.
        "gold_parsed": bool(gold),
    }
    if not prediction or not gold:
        return result
    try:
        result["correct"] = bool(math_verify(gold, prediction))
    except Exception:
        result["correct"] = False
    return result


def verify_answer(text: str | None, reference: str | None) -> bool:
    return grade(text, reference)["correct"]
