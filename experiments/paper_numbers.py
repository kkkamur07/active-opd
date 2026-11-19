"""Recompute every number reported in the paper from the run directories.

    python experiments/paper_numbers.py --runs-dir outputs/runs [--full]

Without --full only metrics.jsonl is read (results/runs/ in this repo is enough).
--full also reads per-answer scores and the question bank (the complete run
directories) for the variance, overlap and training-pool analyses.
Writes results/paper_numbers.json (or --out); prints a summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterator
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SEEDS = (1, 2, 3)
SUMMARY_STEPS = (15, 30, 45, 60)
ALL_STEPS = (0, *SUMMARY_STEPS)
BASELINE = "random"
RUNS: dict[str, tuple[str, ...]] = {
    "answer-level": ("random", "kl_high", "kl_mid", "kl_low", "ent_top4"),
    "question-kl": ("qkl_high", "qkl_mid", "qkl_low"),
    "question-entropy": ("qent1_high",),
    "correctness": ("t_right_s_wrong", "both_right", "mixed", "both_wrong"),
}
RULES: tuple[str, ...] = tuple(arm for arms in RUNS.values() for arm in arms)
OUT = Path(__file__).resolve().parents[1] / "results" / "paper_numbers.json"

Curves = dict[tuple[int, str, int], tuple[Decimal, Decimal]]  # (seed, arm, step) -> (avg@4 %, pass@4 %), exact


def rounded(x: Decimal | float) -> float:
    """Two decimals, half away from zero (applied once, at the end)."""
    exact = x if isinstance(x, Decimal) else Decimal(repr(x))
    return float(exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def load_curves(runs_dir: Path) -> Curves:
    curves: Curves = {}
    for run in RUNS:
        for seed in SEEDS:
            for row in read_jsonl(runs_dir / f"{run}-s{seed}" / "metrics.jsonl"):
                ev = row.get("eval")
                if not isinstance(ev, dict) or "math500" not in ev:
                    continue
                m = ev["math500"]
                curves[(seed, str(row["arm"]), int(str(row["step"])))] = (100 * Decimal(str(m["strict_avg_at_n"])), 100 * Decimal(str(m["strict_pass_at_n"])))
    base = {seed: curves[(seed, BASELINE, 0)] for seed in SEEDS}
    for (seed, arm, _), _v in list(curves.items()):  # rules that reuse the base evaluation have no step-0 row
        curves.setdefault((seed, arm, 0), base[seed])
    return curves


def mean_sd(values: list[Decimal]) -> dict[str, float]:
    return {"mean": rounded(statistics.mean(values)), "sd": rounded(statistics.stdev(values))}


def summary(curves: Curves, arm: str, seed: int, metric: int, steps: tuple[int, ...]) -> Decimal:
    return statistics.mean(curves[(seed, arm, t)][metric] for t in steps)


def rule_table(curves: Curves, steps: tuple[int, ...]) -> dict[str, dict[str, object]]:
    """Accuracy and difference to standard OPD, mean and SD over seeds, for avg@4 and pass@4."""
    table: dict[str, dict[str, object]] = {}
    for arm in RULES:
        entry: dict[str, object] = {}
        for metric, name in ((0, "avg4"), (1, "pass4")):
            values = [summary(curves, arm, s, metric, steps) for s in SEEDS]
            deltas = [v - summary(curves, BASELINE, s, metric, steps) for v, s in zip(values, SEEDS)]
            entry[name] = mean_sd(values)
            if arm != BASELINE:
                entry[f"{name}_delta"] = mean_sd(deltas)
                entry[f"{name}_delta_per_seed"] = [rounded(d) for d in deltas]
        table[arm] = entry
    return table


def gain_curves(curves: Curves) -> dict[str, list[float]]:
    """Mean over seeds of (rule - standard OPD) at each evaluation step."""
    return {arm: [rounded(statistics.mean(curves[(s, arm, t)][0] - curves[(s, BASELINE, t)][0] for s in SEEDS)) for t in ALL_STEPS]
            for arm in RULES if arm != BASELINE}


def shard_scores(stage_dir: Path, stem: str, field: str) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    for path in sorted(stage_dir.glob(f"{stem}.shard*.jsonl")):
        for row in read_jsonl(path):
            scores[(int(str(row["example_index"])), int(str(row["rollout_index"])))] = float(str(row[field]))
    return scores


def between_question_share(scores: dict[tuple[int, int], float]) -> float:
    groups: dict[int, list[float]] = defaultdict(list)
    for (question, _), value in scores.items():
        groups[question].append(value)
    grand = statistics.mean(scores.values())
    total = sum((v - grand) ** 2 for v in scores.values())
    between = sum(len(vs) * (statistics.mean(vs) - grand) ** 2 for vs in groups.values())
    return 100 * between / total


def variance_shares(runs_dir: Path) -> dict[str, list[float]]:
    """Share of answer-score variance between questions, answer-level pools of standard OPD (seed x refresh)."""
    kl: list[float] = []
    entropy: list[float] = []
    for seed in SEEDS:
        for refresh in range(4):
            rdir = runs_dir / f"answer-level-s{seed}" / "arms" / BASELINE / "rounds" / f"round_{refresh:02d}"
            kl.append(rounded(between_question_share(shard_scores(rdir / "oracle", "oracle_kl", "mean_reverse_kl"))))
            entropy.append(rounded(between_question_share(shard_scores(rdir / "entropy", "entropy", "entropy"))))
    return {"reverse_kl": kl, "entropy": entropy}


def top_third(scores: dict[int, float]) -> set[int]:
    ranked = sorted(scores, key=lambda q: -scores[q])
    return set(ranked[: len(ranked) // 3])


def entropy_kl_overlap(runs_dir: Path) -> list[int]:
    """Per refresh: questions chosen by single-answer entropy that are also in the top third by mean reverse KL."""
    overlaps: list[int] = []
    for seed in SEEDS:
        for refresh in range(4):
            rdir = runs_dir / f"question-entropy-s{seed}" / "arms" / "qent1_high" / "rounds" / f"round_{refresh:02d}"
            entropy = shard_scores(rdir / "entropy", "entropy", "entropy")
            kl = shard_scores(rdir / "oracle", "oracle_kl", "mean_reverse_kl")
            per_question: dict[int, list[float]] = defaultdict(list)
            for (question, _), value in kl.items():
                per_question[question].append(value)
            one_answer = {q: v for (q, i), v in entropy.items() if i == 0}
            overlaps.append(len(top_third(one_answer) & top_third({q: statistics.mean(v) for q, v in per_question.items()})))
    return overlaps


def training_pool(bank_dir: Path) -> dict[str, int]:
    """Student labels of all sampled questions and of the teacher-labeled pool every rule draws from."""
    counts: dict[str, int] = defaultdict(int)
    for row in read_jsonl(bank_dir / "questions.jsonl"):
        label = str(row.get("student_label"))
        counts[f"sampled_{label}"] += 1
        if row.get("teacher_grades") is not None:
            counts[f"pool_{label}"] += 1
            counts[f"pool_bucket_{row.get('bucket')}"] += 1
    return dict(sorted(counts.items()))


def compute(runs_dir: Path, full: bool) -> dict[str, object]:
    curves = load_curves(runs_dir)
    numbers: dict[str, object] = {
        "base_avg4": rounded(statistics.mean(curves[(s, BASELINE, 0)][0] for s in SEEDS)),
        "mean_over_steps_15_60": rule_table(curves, SUMMARY_STEPS),
        "final_checkpoint_step_60": rule_table(curves, (60,)),
        "gain_per_step": {"steps": list(ALL_STEPS), **gain_curves(curves)},
    }
    if full:
        numbers["variance_between_questions_pct"] = variance_shares(runs_dir)
        numbers["entropy_vs_kl_top_third_overlap"] = entropy_kl_overlap(runs_dir)
        numbers["training_pool"] = training_pool(runs_dir / "bank-8k")
    return numbers


def print_summary(numbers: dict[str, object]) -> None:
    table = numbers["mean_over_steps_15_60"]
    assert isinstance(table, dict)
    print(f"{'rule':16s} {'avg@4':>14s} {'delta avg@4':>14s} {'delta pass@4':>14s}")
    for arm, e in table.items():
        avg, d, p = e["avg4"], e.get("avg4_delta"), e.get("pass4_delta")
        fmt = lambda x: "" if x is None else f"{x['mean']:+.2f} ± {x['sd']:.2f}"  # noqa: E731
        print(f"{arm:16s} {avg['mean']:8.2f} ± {avg['sd']:.2f} {fmt(d):>14s} {fmt(p):>14s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="also read per-answer scores and the question bank")
    parser.add_argument("--out", type=Path, default=OUT, help=f"output JSON (default {OUT})")
    args = parser.parse_args()
    numbers = compute(args.runs_dir, args.full)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(numbers, indent=1) + "\n")
    print_summary(numbers)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
