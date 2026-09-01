"""Load a math pool from OpenThoughts (or an eval set: MATH-500, AIME 2025/2026).

See [../../docs/guide.md](../../docs/guide.md#dataset-contract-and-limits) for the
measurements behind the column mapping, the usability filter, and the prompt format.
"""

from __future__ import annotations

import random
from typing import Any

# Qwen 3.5 Math eval instruction is the same prompt -> prompt alignment with Rethinking OPD Li et. al. 
BOXED = "Please reason step by step, and put your final answer within \\boxed{}."

DATASETS = {
    "openthoughts": ("siyanzhao/Openthoughts_math_30k_opsd", "train"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    # MathArena AIME sets: 30 problems each, integer answers (int64 column,
    # stringified by _example; Math-Verify then equates \boxed{070} and 70).
    "aime2025": ("MathArena/aime_2025", "train"),
    "aime2026": ("MathArena/aime_2026", "train"),
}

# Pooled sets: the concatenation of several DATASETS entries. Each example
# keeps its source key as the id prefix (``aime2025:3``), so per-year splits
# stay recoverable from the pooled eval rows.
COMBINED = {
    "aime2526": ("aime2025", "aime2026"),
}

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
    "aime2025": {
        "problem": "problem",
        "answer": "answer",
        "solution": None,
        "cot": None,
        "source": None,
        "correct": None,
    },
    "aime2026": {
        "problem": "problem",
        "answer": "answer",
        "solution": None,
        "cot": None,
        "source": None,
        "correct": None,
    },
}


def format_prompt(problem: str) -> str:
    """Append the boxed instruction unconditionally for instruction following and answer generation."""

    return f"{problem.strip()}\n\n{BOXED}"


def _example(dataset: str, columns: dict[str, str | None], row_index: int, row: Any) -> dict[str, Any] | None:

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
    
    # The information box where we have everything. 
    return {
        "id": f"{dataset}:{row_index}",
        "row_index": row_index,
        "prompt": format_prompt(str(problem)),
        "answer": str(answer),
        "solution": text("solution"),
        "cot": text("cot"),
        "source": text("source"),
        "dataset_correct": bool(correct),
    }


def usable_examples(rows: Any, dataset: str = "openthoughts") -> list[dict[str, Any]]:
    """Every row with a non-empty problem and answer, in dataset order."""

    columns = COLUMNS.get(dataset, COLUMNS["openthoughts"])
    return [
        example
        for row_index, row in enumerate(rows)
        if (example := _example(dataset, columns, row_index, row)) is not None
    ]


def _sample(usable: list[dict[str, Any]], n: int, dataset: str, seed: int) -> list[dict[str, Any]]:
    if len(usable) < n:
        raise ValueError(
            f"Requested {n} examples from {dataset!r} but only {len(usable)} rows are usable "
            f"(non-empty problem and non-empty answer)."
        )

    # To get the usable samples. 
    return random.Random(seed).sample(usable, n)


def examples_from_rows(
    rows: Any,
    n: int = 512,
    dataset: str = "openthoughts",
    seed: int = 42,
) -> list[dict[str, Any]]:

    return _sample(usable_examples(rows, dataset), n, dataset, seed)


def _load_rows(dataset: str, split: str | None) -> Any:
    try:
        name, default_split = DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose from {sorted(DATASETS) + sorted(COMBINED)}."
        ) from exc

    from datasets import load_dataset

    records = load_dataset(name, split=split or default_split)

    columns = COLUMNS.get(dataset, COLUMNS["openthoughts"])

    wanted = [c for c in columns.values() if c and c in records.column_names]

    if wanted:
        records = records.select_columns(wanted)

    return records


def load_examples(
    dataset: str = "openthoughts",
    n: int = 512,
    seed: int = 42,
    split: str | None = None,
) -> list[dict[str, Any]]:

    parts = COMBINED.get(dataset, (dataset,))
    usable = [ex for part in parts for ex in usable_examples(_load_rows(part, split), part)]
    return _sample(usable, n, dataset, seed)
