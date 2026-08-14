"""Trajectory selection: per prompt, keep k of num_rollouts by the arm's policy.

Interface defined in docs/pipeline.md. Pure CPU — the caller merges entropy
scores into the trajectory rows before calling when the arm needs them.
Truncated (cap-hit) trajectories are eligible (ADR 0002).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger

ARMS = ("entropy_top4", "random_top4", "all")


def _selected_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_index": row["example_index"],
        "rollout_index": row["rollout_index"],
        "entropy": row.get("entropy"),
        "correct": row["correct"],
        "truncated": row["truncated"],
        "response_length": row["response_length"],
    }


def select_trajectories(
    arm: str,
    trajectories: list[dict[str, Any]],
    *,
    k: int,
    num_rollouts: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select trajectories for one round under ``arm``.

    trajectories: merged rows for one round (entropy merged in when the
    arm needs it). Per example_index keep k of num_rollouts:
      entropy_top4: highest entropy, ties -> lower rollout_index
      random_top4:  default_rng(seed + example_index).choice, no replacement
      all:          keep everything
    Truncated rows are eligible. Returns selected.jsonl-shaped rows sorted
    by (example_index, rollout_index).
    """

    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")

    by_example: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        by_example[row["example_index"]].append(row)

    selected: list[dict[str, Any]] = []
    for example_index in sorted(by_example):
        rows = sorted(by_example[example_index], key=lambda r: r["rollout_index"])

        if len(rows) != num_rollouts:
            logger.warning(
                "example {} has {} trajectories, expected {}",
                example_index, len(rows), num_rollouts,
            )

        if arm == "all":
            selected.extend(_selected_row(r) for r in rows)
            continue

        if len(rows) < k:
            raise ValueError(
                f"example {example_index} has only {len(rows)} trajectories; "
                f"cannot select k={k} for arm {arm!r}"
            )

        if arm == "entropy_top4":
            missing = [r["rollout_index"] for r in rows if r.get("entropy") is None]
            if missing:
                raise ValueError(
                    f"arm 'entropy_top4' needs an entropy score on every row, but "
                    f"example {example_index} rollouts {missing} have none; merge "
                    "entropy/entropy.shard*.jsonl into the trajectories first"
                )
            chosen = sorted(rows, key=lambda r: (-r["entropy"], r["rollout_index"]))[:k]
        else:  # random_top4
            rng = np.random.default_rng(seed + example_index)
            indices = rng.choice(len(rows), size=k, replace=False)
            chosen = [rows[i] for i in indices]

        selected.extend(_selected_row(r) for r in chosen)

    selected.sort(key=lambda r: (r["example_index"], r["rollout_index"]))
    return selected


if __name__ == "__main__":
    # Self-test on synthetic rows: 2 examples x 4 rollouts, k=2.
    def _rows():
        rows = []
        for example_index in (0, 1):
            for rollout_index in range(4):
                rows.append({
                    "example_index": example_index,
                    "rollout_index": rollout_index,
                    "entropy": [0.5, 0.9, 0.9, 0.1][rollout_index],
                    "correct": rollout_index % 2 == 0,
                    "truncated": rollout_index == 3,
                    "response_length": 10 + rollout_index,
                })
        return rows

    top = select_trajectories("entropy_top4", _rows(), k=2, num_rollouts=4, seed=42)
    # Highest entropy 0.9 is tied between rollouts 1 and 2 -> lower index wins.
    assert [(r["example_index"], r["rollout_index"]) for r in top] == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert all(r["entropy"] == 0.9 for r in top)

    rand_a = select_trajectories("random_top4", _rows(), k=2, num_rollouts=4, seed=42)
    rand_b = select_trajectories("random_top4", _rows(), k=2, num_rollouts=4, seed=42)
    assert rand_a == rand_b, "random_top4 must be deterministic in seed"
    assert len(rand_a) == 4
    assert len({(r["example_index"], r["rollout_index"]) for r in rand_a}) == 4, "no replacement"

    everything = select_trajectories("all", _rows(), k=2, num_rollouts=4, seed=42)
    assert len(everything) == 8
    assert everything == sorted(everything, key=lambda r: (r["example_index"], r["rollout_index"]))
    # Truncated rollout 3 is kept (ADR 0002).
    assert any(r["truncated"] for r in everything)

    no_entropy = [{k_: v for k_, v in r.items() if k_ != "entropy"} for r in _rows()]
    try:
        select_trajectories("entropy_top4", no_entropy, k=2, num_rollouts=4, seed=42)
    except ValueError:
        pass
    else:
        raise AssertionError("entropy_top4 without entropy scores must raise")
    assert select_trajectories("all", no_entropy, k=2, num_rollouts=4, seed=42)[0]["entropy"] is None

    try:
        select_trajectories("random_top4", _rows()[:1], k=2, num_rollouts=4, seed=42)
    except ValueError:
        pass
    else:
        raise AssertionError("fewer than k rows must raise")

    print("apod/selection.py self-test passed")
