"""JSONL + npz storage for examples, trajectories, and teacher logits."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: str | Path, *, drop_torn_tail: bool = False) -> list[dict[str, Any]]:
    """Read a jsonl file; ``drop_torn_tail`` tolerates a truncated LAST line.

    Resume readers of append-mode shard files set it: a kill or power loss
    mid-append leaves a partial final line, and without tolerance every later
    resume dies on JSONDecodeError until someone hand-edits the file. Only
    the final line is forgiven -- corruption anywhere else still raises.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]

    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if drop_torn_tail and i == len(lines) - 1:
                break
            raise
    return rows


def write_jsonl(path: str | Path, rows: Any) -> None:
    """Atomic whole-file write (tmp + fsync + rename).

    Several callers use the file's bare existence as a done-marker
    (selected.jsonl, metrics.jsonl rewrite), so a crash mid-write must never
    leave a plausible-looking partial file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")

    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp, path)


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
