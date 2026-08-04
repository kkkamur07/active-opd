import pytest

from aopd.data.datasets import example_from_record, iter_examples


def test_record_with_real_answer_column_is_used_verbatim():
    example = example_from_record(
        {"problem": "What is 2 + 2?", "answer": "4", "unique_id": "test/1"}
    )

    assert example.prompt == "What is 2 + 2?"
    assert example.reference_answer == "4"
    assert example.problem_id == "test/1"


def test_record_without_answer_column_raises_instead_of_scraping():
    """An OpenThoughts-shaped record has no answer column.

    Scraping one out of the assistant's reasoning trace produced references
    like "'yes' if", and every downstream correctness label inherited that
    noise. Refusing loudly is the point of this test.
    """

    with pytest.raises(ValueError, match="no ground-truth answer column"):
        example_from_record(
            {
                "system": "s",
                "conversations": [
                    {"from": "user", "value": "Generate an executable function."},
                    {"from": "assistant", "value": "...so the answer is yes if n>0"},
                ],
            }
        )


def test_solution_fallback_requires_an_explicit_opt_in_and_a_boxed_answer():
    boxed = example_from_record(
        {"problem": "p", "solution": r"lots of work \boxed{42}"},
        allow_solution_fallback=True,
    )
    assert boxed.reference_answer == "42"

    # Prose is not trustworthy enough to become ground truth even when opted in.
    with pytest.raises(ValueError):
        example_from_record(
            {"problem": "p", "solution": "so the answer is 42"},
            allow_solution_fallback=True,
        )


def test_iter_examples_limit_counts_yielded_not_scanned():
    records = [
        {"problem": "a", "answer": "1"},
        {"problem": "b"},  # invalid: no answer column
        {"problem": "c", "answer": "3"},
        {"problem": "d", "answer": "4"},
    ]

    examples = list(iter_examples(records, limit=3, skip_invalid=True))

    assert [example.reference_answer for example in examples] == ["1", "3", "4"]


def test_metadata_is_dropped_by_default():
    """Keeping the full row pinned whole reasoning traces in memory."""

    example = example_from_record(
        {"problem": "p", "answer": "1", "solution": "x" * 10_000}
    )

    assert example.metadata is None
