"""Load a math pool from OpenThoughts (or MATH-500).

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


def examples_from_rows(
    rows: Any,
    n: int = 512,
    dataset: str = "openthoughts",
    seed: int = 42,
) -> list[dict[str, Any]]:

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

    # To get the usable samples. 
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

    wanted = [c for c in columns.values() if c and c in records.column_names]

    if wanted:
        records = records.select_columns(wanted)
        
    return examples_from_rows(records, n=n, dataset=dataset, seed=seed)
