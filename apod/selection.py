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
  qkl_high/mid/low  question level: rank questions by the mean exact reverse KL
                 of their rollouts, keep every rollout of the top / middle /
                 bottom third of questions
  qent_high/mid/low question level, same tertile rule, but each question is
                 scored by the mean entropy-stage ``entropy`` of its rollouts
                 (student only: no teacher pass needed to select)
The pre-2026-09-01 arm names (entropy_top4, random_top4, all) remain valid
aliases so apod.main and older run dirs keep working.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger

RULES = (
    "all_k", "random_k", "entropy_top_k", "kl_high", "kl_mid", "kl_low",
    "qkl_high", "qkl_mid", "qkl_low", "qent_high", "qent_mid", "qent_low",
    "qent1_high", "qent1_mid", "qent1_low",
)
KL_TERTILES = ("kl_high", "kl_mid", "kl_low")
# Question-level selection (2026-11-13): each question's KL = mean exact reverse KL
# of its rollouts; questions ranked descending, cut into thirds; the arm keeps
# EVERY rollout of the questions in its third (k must equal num_rollouts and the
# driver over-samples questions 3x: driver.question_oversample=3).
QKL_TERTILES = ("qkl_high", "qkl_mid", "qkl_low")
# Question-level selection by student entropy (2026-11-15): identical tertile
# rule, but each question's score = mean entropy-stage entropy of its rollouts.
QENT_TERTILES = ("qent_high", "qent_mid", "qent_low")
# Single-rollout question entropy (2026-11-16): the same tertile rule, but each
# question's score = the entropy of its FIRST rollout (rollout_index 0) only; the
# arm still trains on all of the question's rollouts. Rollouts are i.i.d. given
# the question, so this is the cheap pipeline "1 rollout to score, then sample
# the rest" with the same training distribution.
QENT1_TERTILES = ("qent1_high", "qent1_mid", "qent1_low")
# rule -> (tertile names, per-rollout score key, per-question field on selected
# rows, score from rollout 0 only instead of the mean over rollouts)
QUESTION_RULES = {
    **{r: (QKL_TERTILES, "mean_reverse_kl", "question_kl", False) for r in QKL_TERTILES},
    **{r: (QENT_TERTILES, "entropy", "question_entropy", False) for r in QENT_TERTILES},
    **{r: (QENT1_TERTILES, "entropy", "question_entropy_r0", True) for r in QENT1_TERTILES},
}
LEGACY_RULES = {"entropy_top4": "entropy_top_k", "random_top4": "random_k", "all": "all_k"}
ARMS = tuple(LEGACY_RULES) + RULES


def canonical_rule(arm: str) -> str:
    """The rule name for ``arm``; legacy aliases map to their rule."""

    rule = LEGACY_RULES.get(arm, arm)
    if rule not in RULES:
        raise ValueError(f"Unknown selection rule {arm!r}; expected one of {ARMS}")
    return rule


def needs_entropy(arm: str) -> bool:
    return canonical_rule(arm) == "entropy_top_k" or canonical_rule(arm) in QENT_TERTILES + QENT1_TERTILES


def needs_reverse_kl(arm: str) -> bool:
    return canonical_rule(arm) in KL_TERTILES + QKL_TERTILES


def is_question_rule(arm: str) -> bool:
    return canonical_rule(arm) in QUESTION_RULES


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
      qkl_* / qent_*: question level -- score each question by the mean
                     mean_reverse_kl (qkl) or entropy (qent) of its rollouts,
                     rank descending (ties -> lower example_index), keep every
                     rollout of the high / middle / low third of questions
      qent1_*:       as qent_*, but the question's score is the entropy of
                     its rollout_index-0 rollout alone
      all_k:         keep everything
    Truncated rows are eligible. Returns selected.jsonl-shaped rows sorted
    by (example_index, rollout_index).
    """

    rule = canonical_rule(arm)

    by_example: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        by_example[row["example_index"]].append(row)

    if rule in QUESTION_RULES:
        tertiles, score_key, field, first_only = QUESTION_RULES[rule]
        if k != num_rollouts:
            raise ValueError(f"rule {rule!r} keeps every rollout of a question: k={k} must equal num_rollouts={num_rollouts}")
        if len(by_example) % 3:
            raise ValueError(f"rule {rule!r}: {len(by_example)} questions is not a multiple of 3")
        question_score = {}
        for example_index, rows in by_example.items():
            if first_only:
                first = [r for r in rows if r["rollout_index"] == 0]
                if len(first) != 1:
                    raise ValueError(f"rule {rule!r}: question {example_index} has {len(first)} rollout_index-0 rows, need 1")
                _require_score(first, score_key, rule, example_index)
                question_score[example_index] = float(first[0][score_key])
            else:
                _require_score(rows, score_key, rule, example_index)
                question_score[example_index] = float(np.mean([r[score_key] for r in rows]))
        ranked = sorted(by_example, key=lambda e: (-question_score[e], e))
        third = len(ranked) // 3
        tertile = tertiles.index(rule)
        keep = sorted(ranked[tertile * third : (tertile + 1) * third])
        out = []
        for example_index in keep:
            for r in sorted(by_example[example_index], key=lambda r: r["rollout_index"]):
                row = _selected_row(r)
                row[field] = question_score[example_index]
                out.append(row)
        return out

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

    # Question-level tertiles (qkl_* by KL, qent_* by entropy): 6 questions x 2
    # rollouts, k = num_rollouts = 2 -> each third = 2 questions, all rollouts kept.
    # Per-question means (exact binary fractions, so ties are exact):
    #   KL   q0 .5    q1 .125  q2 .875  q3 .25   q4 .75   q5 .25
    #   ent  q0 .25   q1 .875  q2 .5    q3 .625  q4 .125  q5 .625
    # (q3/q5 tie in both -> lower example_index ranks first.)
    qkl = {0: (.25, .75), 1: (.125, .125), 2: (.75, 1.), 3: (.25, .25), 4: (.75, .75), 5: (.125, .375)}
    qent = {0: (.125, .375), 1: (.875, .875), 2: (.5, .5), 3: (.5, .75), 4: (.125, .125), 5: (.625, .625)}
    q_rows = [
        {"example_index": e, "rollout_index": j, "mean_reverse_kl": qkl[e][j], "entropy": qent[e][j],
         "correct": False, "truncated": False, "response_length": 7}
        for e in range(6) for j in range(2)
    ]
    def _qpick(rule, rows=q_rows):
        out = select_trajectories(rule, rows, k=2, num_rollouts=2, seed=0)
        assert len(out) == 4 and all(sum(1 for r in out if r["example_index"] == e) == 2 for e in {r["example_index"] for r in out})
        return sorted({r["example_index"] for r in out})
    assert {r: _qpick(r) for r in QKL_TERTILES} == {"qkl_high": [2, 4], "qkl_mid": [0, 3], "qkl_low": [1, 5]}
    assert {r: _qpick(r) for r in QENT_TERTILES} == {"qent_high": [1, 3], "qent_mid": [2, 5], "qent_low": [0, 4]}
    hi = select_trajectories("qent_high", q_rows, k=2, num_rollouts=2, seed=0)
    assert all(abs(r["question_entropy"] - float(np.mean(qent[r["example_index"]]))) < 1e-12 for r in hi)
    assert "question_kl" not in hi[0]
    assert "question_entropy" not in select_trajectories("qkl_high", q_rows, k=2, num_rollouts=2, seed=0)[0]
    # qent ignores KL entirely: scrambling KL leaves qent picks unchanged.
    scrambled = [{**r, "mean_reverse_kl": 1.0 - r["mean_reverse_kl"]} for r in q_rows]
    assert all(_qpick(r, scrambled) == _qpick(r) for r in QENT_TERTILES)
    # Scores and shape requirements.
    for rule, drop in (("qent_mid", "entropy"), ("qkl_mid", "mean_reverse_kl")):
        try:
            select_trajectories(rule, [{**r, drop: None} for r in q_rows], k=2, num_rollouts=2, seed=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{rule} without {drop} must raise")
    no_kl = [{k_: v for k_, v in r.items() if k_ != "mean_reverse_kl"} for r in q_rows]
    assert _qpick("qent_low", no_kl) == [0, 4], "qent must not need reverse KL"
    for bad in (dict(rows=q_rows[:10], k=2, n=2), dict(rows=q_rows, k=1, n=2)):
        try:
            select_trajectories("qent_high", bad["rows"], k=bad["k"], num_rollouts=bad["n"], seed=0)
        except ValueError:
            pass
        else:
            raise AssertionError("qent needs a multiple of 3 questions and k == num_rollouts")
    # qent1: score = rollout 0's entropy only. rollout-0 entropies q0 .125 q1 .875
    # q2 .5 q3 .5 q4 .125 q5 .625 -> high [1,5], mid [2,3], low [0,4] (ties ->
    # lower index) -- differs from qent (high [1,3]) because q3's rollout 1 is ignored.
    assert {r: _qpick(r) for r in QENT1_TERTILES} == {"qent1_high": [1, 5], "qent1_mid": [2, 3], "qent1_low": [0, 4]}
    h1 = select_trajectories("qent1_high", q_rows, k=2, num_rollouts=2, seed=0)
    assert all(r["question_entropy_r0"] == qent[r["example_index"]][0] for r in h1) and len(h1) == 4
    # rollout-1 entropies are ignored entirely (scramble them, picks unchanged).
    r1_scrambled = [{**r, "entropy": (1.0 - r["entropy"]) if r["rollout_index"] else r["entropy"]} for r in q_rows]
    assert all(_qpick(r, r1_scrambled) == _qpick(r) for r in QENT1_TERTILES)
    try:
        select_trajectories("qent1_high", [r for r in q_rows if not (r["example_index"] == 2 and r["rollout_index"] == 0)] +
                            [{**q_rows[4], "rollout_index": 1}], k=2, num_rollouts=2, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("qent1 without a rollout-0 row must raise")
    assert all(needs_entropy(r) and not needs_reverse_kl(r) and is_question_rule(r) for r in QENT1_TERTILES)
    assert all(is_question_rule(r) for r in QKL_TERTILES + QENT_TERTILES)
    assert not is_question_rule("kl_high") and not is_question_rule("entropy_top_k")
    assert all(needs_entropy(r) and not needs_reverse_kl(r) for r in QENT_TERTILES)
    assert all(needs_reverse_kl(r) and not needs_entropy(r) for r in QKL_TERTILES)

    print("apod/selection.py self-test passed")
