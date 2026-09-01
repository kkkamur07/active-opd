"""Grader behaviour on AIME-style integer answers (no normalisation layer).

MathArena stores AIME answers as int64; ``_example`` stringifies them and
``apod.verification.grade`` boxes the gold before Math-Verify parses it, so
``\\boxed{070}`` and ``\\boxed{70}`` both match gold ``70`` with no extra
code. Pinned here so a Math-Verify upgrade that changes any of this fails
loudly. Runs under pytest or as ``python -m tests.test_grade_aime``.
"""

from __future__ import annotations

from apod.verification import grade

CORRECT = [
    (r"so the answer is \boxed{70}.", "70"),
    (r"\boxed{070}", "70"),        # AIME three-digit convention
    (r"\boxed{007}", "7"),
    (r"\boxed{000}", "0"),
    (r"\boxed{70}", 70),           # int64 gold, as the Hub column delivers it
    (r"\boxed{70.0}", "70"),
    (r"\boxed{ 70 }", "70"),
    (r"\boxed{1,000}", "1000"),
    (r"\boxed{070}.", "70"),
]

INCORRECT = [
    (r"\boxed{71}", "70"),
    (r"\boxed{7}", "70"),
]


def test_integer_answers_match():
    for text, gold in CORRECT:
        verdict = grade(text, gold)
        assert verdict["correct"] and verdict["has_boxed"], (text, gold, verdict)


def test_wrong_integers_do_not_match():
    for text, gold in INCORRECT:
        assert not grade(text, gold)["correct"], (text, gold)


def test_unboxed_answer_is_strict_incorrect():
    # Loose credit exists (math-verify reads the last expression), but the
    # strict rule -- no \boxed = incorrect, cap-hit traces included -- keys
    # on has_boxed, which must be False here.
    verdict = grade("the answer is 70", "70")
    assert verdict["correct"] and not verdict["has_boxed"]
    assert not (verdict["correct"] and verdict["has_boxed"])


if __name__ == "__main__":
    test_integer_answers_match()
    test_wrong_integers_do_not_match()
    test_unboxed_answer_is_strict_incorrect()
    print("tests/test_grade_aime.py passed")
