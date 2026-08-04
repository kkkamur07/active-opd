"""Lazy record helpers shared by trace profiling and generation.

These read raw dataset rows rather than :class:`MathExample`, because their job
is to measure and inspect a corpus (including rows that have no usable
ground-truth answer) rather than to build a training set.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aopd.data.answers import extract_final_answer
from aopd.data.datasets import load_dataset_records


def _first_value(row: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _conversation_messages(
    row: Mapping[str, Any],
    roles: tuple[str, ...],
) -> list[str]:
    conversations = row.get("conversations", row.get("messages"))
    if not isinstance(conversations, (list, tuple)):
        return []
    values: list[str] = []
    for message in conversations:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("from", message.get("role", ""))).lower()
        if role not in roles:
            continue
        value = _first_value(message, ("value", "content", "text"))
        if value is not None:
            values.append(str(value))
    return values


def record_texts(row: Mapping[str, Any]) -> dict[str, str | None]:
    """Return full prompt, reference, and assistant trace without clipping."""

    prompt = _first_value(row, ("problem", "question", "prompt", "input"))
    if prompt is None:
        user_messages = _conversation_messages(row, ("user", "human"))
        prompt = user_messages[0] if user_messages else None

    trace = _first_value(row, ("solution", "reasoning", "completion"))
    if trace is None:
        assistant_messages = _conversation_messages(
            row,
            ("assistant", "gpt", "bot"),
        )
        trace = assistant_messages[-1] if assistant_messages else None

    reference = _first_value(row, ("answer", "final_answer", "target"))
    if reference is None and trace is not None:
        reference = extract_final_answer(str(trace)).answer
    problem_id = _first_value(row, ("id", "problem_id", "index", "unique_id"))
    return {
        "problem_id": str(problem_id) if problem_id is not None else None,
        "prompt": str(prompt) if prompt is not None else None,
        "reference": str(reference) if reference is not None else None,
        "trace": str(trace) if trace is not None else None,
    }


def iter_records(
    *,
    dataset_name: str = "open-r1/OpenR1-Math-220k",
    split: str = "train",
    streaming: bool = True,
    limit: int | None = None,
):
    """Yield raw records with an explicit streaming/cleanup policy.

    A bounded non-streaming slice is useful for short profiling jobs: it
    avoids the background HTTP/Xet worker teardown bug in the installed
    ``datasets`` release while preserving the dataset's first-record order.
    """

    resolved_split = split
    if not streaming and limit is not None:
        resolved_split = f"{split}[:{limit}]"
    yield from load_dataset_records(
        dataset_name=dataset_name,
        split=resolved_split,
        streaming=streaming,
    )


def model_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        r"Return only the final answer in exactly the form \boxed{answer}."
    )
