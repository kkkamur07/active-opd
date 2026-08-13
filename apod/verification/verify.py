"""Grade a generation with Hugging Face Math-Verify."""

from __future__ import annotations

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify as math_verify


def verify_answer(text: str | None, reference: str | None) -> bool:
    if not text or not reference or not str(reference).strip():
        return False

    gold = parse(rf"\boxed{{{reference}}}")
    pred = parse(
        str(text),
        extraction_config=[
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ],
    )
    if not gold or not pred:
        return False

    try:
        return bool(math_verify(gold, pred))
    except Exception:
        return False
