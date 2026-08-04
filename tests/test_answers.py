import pytest

from aopd.data.answers import (
    answers_equivalent,
    extract_final_answer,
    strip_reasoning_block,
    verify_exact_answer,
)

# Pairs a strict string comparator gets wrong. Every false "wrong" here used to
# land directly in the Active OPD training pool.
EQUIVALENT_PAIRS = [
    (r"(3, \pi/2)", r"\left( 3, \frac{\pi}{2} \right)"),
    ("(3, π/2)", r"\left( 3, \frac{\pi}{2} \right)"),
    ("1/2", r"\frac{1}{2}"),
    ("0.5", r"\frac{1}{2}"),
    (r"2\sqrt{3}", r"2 \sqrt 3"),
    ("x = 5", "5"),
    ("1000", "1,000"),
    ("7.0", "7"),
    (r"\dfrac{3}{4}", r"\frac{3}{4}"),
    (r"\frac12", r"\frac{1}{2}"),
]

DIFFERENT_PAIRS = [
    ("3", "4"),
    (r"\frac{1}{3}", r"\frac{1}{2}"),
    ("-5", "5"),
]


@pytest.mark.parametrize("predicted,reference", EQUIVALENT_PAIRS)
def test_equivalent_answers_are_not_labelled_wrong(predicted, reference):
    decision, _ = answers_equivalent(predicted, reference)
    assert decision is not False, f"{predicted!r} vs {reference!r} judged different"


@pytest.mark.parametrize("predicted,reference", DIFFERENT_PAIRS)
def test_genuinely_different_answers_are_labelled_wrong(predicted, reference):
    decision, _ = answers_equivalent(predicted, reference)
    assert decision is False


def test_unterminated_reasoning_is_truncated_not_wrong():
    """The pilot's two 'wrong' rollouts were mid-thought prose scrapes."""

    text = (
        "<think>\nOkay, r is 3 and theta is pi/2.\n"
        "So, final answer: (3, π/2).\n\nBut let me check once again."
    )
    result = verify_exact_answer(text, r"\left( 3, \frac{\pi}{2} \right)")

    assert result.outcome == "truncated"
    assert result.predicted.answer is None


def test_generator_finish_reason_overrides_a_parseable_answer():
    """A trace can close </think> and still be cut off before its real answer."""

    text = r"<think>reasoning</think> The answer is \boxed{3}"
    result = verify_exact_answer(text, "3", truncated=True)

    assert result.outcome == "truncated"


def test_answer_is_read_only_from_the_post_think_region():
    text = r"<think>maybe \boxed{99}</think> Final answer: \boxed{42}"
    assert extract_final_answer(text).answer == "42"


def test_boxed_extraction_falls_back_instead_of_discarding_a_valid_answer():
    """Every branch used to `return`, so one malformed later box destroyed a
    valid earlier one."""

    text = r"</think> \boxed{37}. Let me double check: \boxed{3"
    extraction = extract_final_answer(text)

    assert extraction.answer == "37"


def test_boxed_marker_requires_an_immediately_following_brace():
    """`text.find("{", ...)` used to scan the entire remaining document."""

    text = r"</think> I used \boxed to format. The solution set is \{1,2\}."
    extraction = extract_final_answer(text)

    assert extraction.answer != "1,2\\"


def test_hash_marker_does_not_match_markdown_headings():
    text = "</think>\n#### Step 1\nsome work\n#### Final Answer\n"
    extraction = extract_final_answer(text)

    assert extraction.answer != "Final Answer"


def test_leading_decimal_point_is_preserved():
    """`strip('.,;:')` turned '.5' into '5'."""

    decision, _ = answers_equivalent(".5", "5")
    assert decision is False


def test_thousands_separators_are_collapsed():
    """Both the normalizer and math_verify read '1,000' as the integer 1000."""

    assert answers_equivalent("1,000", "1000")[0] is True
    assert answers_equivalent("1,000,000", "1000000")[0] is True


def test_strip_reasoning_block_reports_unterminated_thinking():
    assert strip_reasoning_block("<think>abc") == ("", True)
    assert strip_reasoning_block("<think>abc</think>xyz") == ("xyz", False)
    assert strip_reasoning_block("plain text") == ("plain text", False)


def test_missing_reference_is_skipped_not_graded():
    result = verify_exact_answer(r"</think>\boxed{4}", None)
    assert result.outcome == "skipped"


def test_boxed_answer_is_extracted_with_nested_braces():
    text = r"</think> The answer is \boxed{\frac{\sqrt{2}}{2}}"
    assert extract_final_answer(text).answer == r"\frac{\sqrt{2}}{2}"
