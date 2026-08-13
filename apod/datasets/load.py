"""Load a small math pool from OpenThoughts (or MATH-500)."""

from __future__ import annotations

from typing import Any

BOXED = "Please reason step by step, and put your final answer within \\boxed{}."

DATASETS = {
    "openthoughts": ("siyanzhao/Openthoughts_math_30k_opsd", "train"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
}


def format_prompt(problem: str) -> str:
    problem = problem.strip()
    if "\\boxed" in problem:
        return problem
    return f"{problem}\n\n{BOXED}"


def examples_from_rows(rows: Any, n: int = 512) -> list[dict[str, Any]]:
    """Take the first ``n`` rows that have a problem and an answer."""

    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        problem = row.get("problem") or row.get("question")
        answer = row.get("Answer")
        if answer is None:
            answer = row.get("answer")
        if not problem or answer is None or not str(answer).strip():
            continue
        examples.append(
            {
                "id": str(row.get("id", index)),
                "prompt": format_prompt(str(problem)),
                "answer": str(answer),
                "source": row.get("source"),
            }
        )
        if len(examples) >= n:
            break
    if len(examples) < n:
        raise ValueError(f"Requested {n} examples but only parsed {len(examples)}.")
    return examples


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
    return examples_from_rows(records.shuffle(seed=seed), n=n)
