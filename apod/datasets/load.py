"""Load a math pool from OpenThoughts (or MATH-500).

See [../../docs/guide.md](../../docs/guide.md#dataset-contract-and-limits) for the
measurements behind the column mapping, the usability filter, and the prompt format.
"""

from __future__ import annotations

import random
from typing import Any

BOXED = "Please reason step by step, and put your final answer within \\boxed{}."

DATASETS = {
    "openthoughts": ("siyanzhao/Openthoughts_math_30k_opsd", "train"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
}

# Per-dataset column mapping. Explicit rather than a `row.get(a) or row.get(b)`
# chain because the two datasets disagree on both case and coverage: the gold is
# `Answer` in OpenThoughts and `answer` in MATH-500, and MATH-500 has no source,
# correctness, or chain-of-thought column at all. A fallback chain resolves by
# accident of non-emptiness, so an upstream schema change would swap the gold
# column without raising. ``None`` means the dataset does not carry that field
# and the example gets the documented default.
COLUMNS: dict[str, dict[str, str | None]] = {
    "openthoughts": {
        "problem": "problem",
        # `Answer` is the gold; `solution` is prose and `COT_Reason` is the
        # teacher's scratchpad. Neither is a graded value. See docs/guide.md.
        "answer": "Answer",
        "solution": "solution",
        "cot": "COT_Reason",
        "source": "source",
        "correct": "correct",
    },
    "math500": {
        "problem": "problem",
        "answer": "answer",
        "solution": "solution",
        "cot": None,
        "source": None,
        "correct": None,
    },
}


def format_prompt(problem: str) -> str:
    """Append the boxed instruction unconditionally.

    An earlier version skipped the instruction whenever the problem already
    contained ``\\boxed``. That fires on 17 OpenThoughts rows and is wrong on
    every one of them: the ``\\boxed`` there is display markup (calculator keys,
    the empty cells of a subtraction puzzle, a grading-scale table) or sits
    inside an AoPS ``<details>`` spoiler. None of them instruct the model, so
    the heuristic dropped the answer-format instruction from exactly the rows
    whose text is most confusing.
    """

    return f"{problem.strip()}\n\n{BOXED}"


def _example(dataset: str, columns: dict[str, str | None], row_index: int, row: Any) -> dict[str, Any] | None:
    """Build one example, or ``None`` if the row cannot be graded.

    Usable means a non-empty problem and a non-empty gold answer. Seven
    OpenThoughts rows carry an empty ``Answer`` (a literal ``\\boxed{}`` in the
    source solution, all of them proof questions with no value to state); those
    would grade as unconditionally wrong.
    """

    problem = row.get(columns["problem"]) if columns["problem"] else None
    answer = row.get(columns["answer"]) if columns["answer"] else None
    if not problem or not str(problem).strip():
        return None
    if answer is None or not str(answer).strip():
        return None

    def text(key: str) -> str:
        column = columns.get(key)
        value = row.get(column) if column else None
        return "" if value is None else str(value)

    correct = row.get(columns["correct"]) if columns["correct"] else None
    return {
        # Keyed on the position in the *unshuffled* split, so an id names the
        # same row no matter what n, seed, or filter produced the sample.
        "id": f"{dataset}:{row_index}",
        "row_index": row_index,
        "prompt": format_prompt(str(problem)),
        "answer": str(answer),
        "solution": text("solution"),
        "cot": text("cot"),
        "source": text("source"),
        "dataset_correct": bool(correct),
    }


def examples_from_rows(
    rows: Any,
    n: int = 512,
    dataset: str = "openthoughts",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample ``n`` usable examples uniformly from ``rows``.

    ``rows`` must be in the split's original order: ``row_index`` and therefore
    ``id`` are read from the enumeration position.

    Filtering happens before sampling. Taking the first ``n`` survivors of a
    shuffle instead biases the sample toward whatever the shuffle put early, and
    ties the identity of every example to the filter -- change the filter and
    the same seed yields a different set.
    """

    columns = COLUMNS.get(dataset, COLUMNS["openthoughts"])
    usable = [
        example
        for row_index, row in enumerate(rows)
        if (example := _example(dataset, columns, row_index, row)) is not None
    ]
    if len(usable) < n:
        raise ValueError(
            f"Requested {n} examples from {dataset!r} but only {len(usable)} rows are usable "
            f"(non-empty problem and non-empty answer)."
        )
    # Draw order, not sorted by row index: callers shard on list position, and a
    # sorted list would make shard k a contiguous slice of the source ordering.
    return random.Random(seed).sample(usable, n)


def load_examples(
    dataset: str = "openthoughts",
    n: int = 512,
    seed: int = 42,
    split: str | None = None,
) -> list[dict[str, Any]]:
    try:
        name, default_split = DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {sorted(DATASETS)}.") from exc

    from datasets import load_dataset

    records = load_dataset(name, split=split or default_split)
    columns = COLUMNS.get(dataset, COLUMNS["openthoughts"])
    # OpenThoughts stores the same trace two more times (`messages`,
    # `conversations`) plus an identical 1.1 KB `system` string on every row.
    # Projecting to the columns an example actually reads keeps the row-by-row
    # scan from decoding roughly ten times more text than it uses.
    wanted = [c for c in columns.values() if c and c in records.column_names]
    if wanted:
        records = records.select_columns(wanted)
    return examples_from_rows(records, n=n, dataset=dataset, seed=seed)
