"""Trajectory selection: per question, keep k of num_rollouts by a rule.

Interface defined in docs/pipeline.md. Pure CPU -- the caller merges the
scores a rule needs (entropy from the entropy stage, reverse KL from
scripts/oracle_kl.py) into the trajectory rows before calling. Truncated
(cap-hit) trajectories are eligible (ADR 0002).

Rules (the ``arm`` argument; the step-based driver maps arm names to rules):
  all_k          train every generated rollout (no trajectory selection)
  random_k       k uniformly at random -- the no-selection baseline (CONTEXT.md)
  entropy_top_k  the k highest trajectory-entropy rollouts
  kl_high/mid/low the top / middle / bottom k by exact reverse KL among the
                 first 3k rollouts ranked by KL (kl50's tertile rule)
The pre-2026-09-01 arm names (entropy_top4, random_top4, all) remain valid
aliases so apod.main and older run dirs keep working.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger

RULES = ("all_k", "random_k", "entropy_top_k", "kl_high", "kl_mid", "kl_low")
KL_TERTILES = ("kl_high", "kl_mid", "kl_low")
LEGACY_RULES = {"entropy_top4": "entropy_top_k", "random_top4": "random_k", "all": "all_k"}
ARMS = tuple(LEGACY_RULES) + RULES


def canonical_rule(arm: str) -> str:
    """The rule name for ``arm``; legacy aliases map to their rule."""

    rule = LEGACY_RULES.get(arm, arm)
    if rule not in RULES:
        raise ValueError(f"Unknown selection rule {arm!r}; expected one of {ARMS}")
    return rule


def needs_entropy(arm: str) -> bool:
    return canonical_rule(arm) == "entropy_top_k"


def needs_reverse_kl(arm: str) -> bool:
    return canonical_rule(arm) in KL_TERTILES


def _selected_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_index": row["example_index"],
        "rollout_index": row["rollout_index"],
        "entropy": row.get("entropy"),
        "mean_reverse_kl": row.get("mean_reverse_kl"),
        "correct": row["correct"],
        "truncated": row["truncated"],
        "response_length": row["response_length"],
    }


def _require_score(rows: list[dict[str, Any]], key: str, rule: str, example_index: int) -> None:
    missing = [r["rollout_index"] for r in rows if r.get(key) is None]
    if missing:
        raise ValueError(
            f"rule {rule!r} needs a {key!r} score on every row, but question "
            f"{example_index} rollouts {missing} have none; merge the scores "
            "into the trajectories first"
        )


def select_trajectories(
    arm: str,
    trajectories: list[dict[str, Any]],
    *,
    k: int,
    num_rollouts: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select trajectories for one refresh under ``arm`` (a rule or alias).

    trajectories: merged rows for one refresh (entropy / mean_reverse_kl
    merged in when the rule needs them). Per example_index keep k of
    num_rollouts:
      entropy_top_k: highest entropy, ties -> lower rollout_index
      random_k:      default_rng(seed + example_index).choice, no replacement
      kl_high/mid/low: rank by mean_reverse_kl descending (ties -> lower
                     rollout_index); slice [0,k) / [k,2k) / [2k,3k)
      all_k:         keep everything
    Truncated rows are eligible. Returns selected.jsonl-shaped rows sorted
    by (example_index, rollout_index).
    """

    rule = canonical_rule(arm)

    by_example: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        by_example[row["example_index"]].append(row)

    selected: list[dict[str, Any]] = []
    for example_index in sorted(by_example):
        rows = sorted(by_example[example_index], key=lambda r: r["rollout_index"])

        if len(rows) != num_rollouts:
            logger.warning(
                "question {} has {} trajectories, expected {}",
                example_index, len(rows), num_rollouts,
            )

        if rule == "all_k":
            selected.extend(_selected_row(r) for r in rows)
            continue

        need = 3 * k if rule in KL_TERTILES else k
        if len(rows) < need:
            raise ValueError(
                f"question {example_index} has only {len(rows)} trajectories; "
                f"rule {rule!r} needs {need} to select k={k}"
            )

        if rule == "entropy_top_k":
            _require_score(rows, "entropy", rule, example_index)
            chosen = sorted(rows, key=lambda r: (-r["entropy"], r["rollout_index"]))[:k]
        elif rule in KL_TERTILES:
            _require_score(rows, "mean_reverse_kl", rule, example_index)
            ranked = sorted(rows, key=lambda r: (-r["mean_reverse_kl"], r["rollout_index"]))
            tertile = KL_TERTILES.index(rule)
            chosen = ranked[tertile * k : (tertile + 1) * k]
        else:  # random_k
            rng = np.random.default_rng(seed + example_index)
            indices = rng.choice(len(rows), size=k, replace=False)
            chosen = [rows[i] for i in indices]

        selected.extend(_selected_row(r) for r in chosen)

    selected.sort(key=lambda r: (r["example_index"], r["rollout_index"]))
    return selected


if __name__ == "__main__":
    # Self-test on synthetic rows: 2 questions x 4 rollouts, k=2.
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
    assert top == select_trajectories("entropy_top_k", _rows(), k=2, num_rollouts=4, seed=42)

    rand_a = select_trajectories("random_top4", _rows(), k=2, num_rollouts=4, seed=42)
    rand_b = select_trajectories("random_k", _rows(), k=2, num_rollouts=4, seed=42)
    assert rand_a == rand_b, "random_k must be deterministic in seed"
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

    # KL tertiles: 6 rollouts, k=2 -> high = top 2 by KL, mid = next 2, low = last 2.
    kl_rows = [
        {"example_index": 0, "rollout_index": j, "mean_reverse_kl": kl,
         "correct": True, "truncated": False, "response_length": 5}
        for j, kl in enumerate([0.3, 0.9, 0.1, 0.6, 0.9, 0.2])
    ]
    picks = {
        rule: [r["rollout_index"] for r in select_trajectories(rule, kl_rows, k=2, num_rollouts=6, seed=0)]
        for rule in KL_TERTILES
    }
    assert picks == {"kl_high": [1, 4], "kl_mid": [0, 3], "kl_low": [2, 5]}, picks
    try:
        select_trajectories("kl_mid", kl_rows[:5], k=2, num_rollouts=6, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("kl tertiles need 3k rows")
    assert needs_reverse_kl("kl_low") and not needs_reverse_kl("random_k")
    assert needs_entropy("entropy_top4") and not needs_entropy("all_k")

    print("apod/selection.py self-test passed")
