"""JSONL + npz storage for examples, trajectories, and teacher logits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_npz(path: str | Path, batch: dict[str, Any]) -> Path:
    """Save a generation batch; includes ``logits`` when present (teacher)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "input_ids": np.asarray(batch["input_ids"], dtype=np.int32),
        "response_lengths": np.asarray(batch["response_lengths"], dtype=np.int32),
        "truncated": np.asarray(batch["truncated"], dtype=bool),
        "prompt_length": np.asarray(batch["prompt_length"], dtype=np.int32),
        "responses": np.asarray(batch["responses"], dtype=object),
    }
    if "finish_reasons" in batch:
        arrays["finish_reasons"] = np.asarray(batch["finish_reasons"], dtype=object)
    if "logits" in batch:
        arrays["logits"] = np.asarray(batch["logits"], dtype=np.float16)
    np.savez_compressed(path, **arrays)
    return path


def read_shards(directory: str | Path, pattern: str) -> list[dict[str, Any]]:
    """Concatenate every shard file matching ``pattern`` in ``directory``.

    Shards are written per GPU process (``trajectories.shard0.jsonl``, ...) so
    two processes never append to one file. Example indices are disjoint across
    shards, so concatenation is the whole merge.
    """

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob(pattern)):
        rows.extend(read_jsonl(path))
    return rows
