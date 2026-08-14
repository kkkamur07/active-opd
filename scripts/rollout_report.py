"""Read the collected trajectory shards and report how good the rollouts are.

This is the gate before spending GPU time on entropy: if almost every trace is
truncated, or every prompt is solved 16/16, the prompt pool is wrong and no
acquisition score will rescue it.

For the 512 x 1 pass the question is narrower -- how many trajectories were
wrong -- and "wrong" needs splitting. A truncated trace never reached a final
answer, and a trace math-verify could not extract anything from was never
graded against the gold in a meaningful way. Both count as not-correct, neither
is a wrong answer, and folding them together overstates the model's error rate.

Every field degrades to ``None`` when the underlying column is missing, so runs
collected before those columns existed still report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from apod.datasets import read_shards

PERCENTILES = (50, 80, 95, 99)


def summarize(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    stats = {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }
    for percentile in PERCENTILES:
        stats[f"p{percentile}"] = float(np.percentile(values, percentile))
    return stats


def share(count: int, total: int) -> dict[str, Any]:
    return {"count": int(count), "fraction": float(count) / total if total else None}


def column(rows: list[dict], key: str) -> np.ndarray | None:
    """Boolean column, or None if any row is missing it (an older run)."""

    if not all(key in row and row[key] is not None for row in rows):
        return None
    return np.asarray([bool(row[key]) for row in rows], dtype=bool)


def reparse_answers(rows: list[dict]) -> np.ndarray | None:
    """Recompute ``has_answer`` from the stored response text.

    Only on request: math-verify's parse is not free and a 512-trace file of
    16k-token traces takes a while.
    """

    if not all(row.get("response") is not None for row in rows):
        return None
    from apod.verification import parse_prediction

    return np.asarray([bool(parse_prediction(row["response"])) for row in rows], dtype=bool)


def incorrect_breakdown(
    incorrect: np.ndarray, truncated: np.ndarray, has_answer: np.ndarray | None
) -> dict[str, Any]:
    """Split the not-correct traces into failure modes that mean different things."""

    total = int(incorrect.sum())
    breakdown: dict[str, Any] = {
        "incorrect_total": total,
        "truncated": share(int((incorrect & truncated).sum()), total),
        "complete": share(int((incorrect & ~truncated).sum()), total),
    }
    if has_answer is None:
        # No has_answer column and no --reparse: say so rather than guessing,
        # because "unknown" and "the model answered wrong" are not the same
        # claim about the model.
        breakdown["no_parseable_answer"] = None
        breakdown["wrong_answer"] = None
        breakdown["cross"] = None
        return breakdown

    no_answer = ~has_answer
    breakdown["no_parseable_answer"] = share(int((incorrect & no_answer).sum()), total)
    breakdown["wrong_answer"] = share(int((incorrect & has_answer).sum()), total)
    breakdown["cross"] = {
        "truncated_no_answer": int((incorrect & truncated & no_answer).sum()),
        "truncated_with_answer": int((incorrect & truncated & has_answer).sum()),
        "complete_no_answer": int((incorrect & ~truncated & no_answer).sum()),
        "complete_wrong_answer": int((incorrect & ~truncated & has_answer).sum()),
    }
    return breakdown


def by_source(rows: list[dict], correct: np.ndarray, truncated: np.ndarray) -> dict[str, Any] | None:
    """Accuracy per dataset ``source``, when the dataloader carried one through."""

    if not any(row.get("source") for row in rows):
        return None
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get("source") or "unknown")].append(index)

    out: dict[str, Any] = {}
    for name in sorted(groups):
        idx = np.asarray(groups[name], dtype=np.int64)
        out[name] = {
            "traces": int(idx.size),
            "correct": int(correct[idx].sum()),
            "accuracy": float(correct[idx].mean()),
            "truncated_fraction": float(truncated[idx].mean()),
        }
    return out


def collection_throughput(directory: Path, pattern: str) -> dict[str, Any] | None:
    """Echo what the collection run measured, if those summaries are around."""

    shards = []
    for path in sorted(directory.glob(pattern)):
        try:
            summary = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("throughput"):
            shards.append({"file": path.name, **summary["throughput"],
                           "wall_seconds": summary.get("wall_seconds"),
                           "peak_gpu_memory_gib": summary.get("peak_gpu_memory_gib")})
    if not shards:
        return None

    generated = sum(s["generated_tokens"] for s in shards)
    prompt = sum(s["prompt_tokens"] for s in shards)
    sequences = sum(s["sequences"] for s in shards)
    # Shards are separate processes on separate GPUs running at the same time,
    # so the run's wall clock is the slowest shard, not the sum. Summing the
    # seconds and dividing would understate the aggregate rate by ~num_shards.
    span = max((s["seconds"] for s in shards), default=0.0)
    divisor = max(span, 1e-9)
    return {
        "shards": shards,
        "aggregate": {
            "shards": len(shards),
            "generate_seconds_slowest_shard": span,
            "prompt_tokens": prompt,
            "generated_tokens": generated,
            "total_tokens": prompt + generated,
            "sequences": sequences,
            "generated_tokens_per_s": generated / divisor,
            "total_tokens_per_s": (prompt + generated) / divisor,
            "sequences_per_s": sequences / divisor,
            "peak_gpu_memory_gib": max(
                (s["peak_gpu_memory_gib"] for s in shards if s.get("peak_gpu_memory_gib")), default=None
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", default="outputs/trajectories")
    parser.add_argument("--pattern", default="trajectories.shard*.jsonl")
    parser.add_argument("--summary-pattern", default="collection_summary.shard*.json")
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="recompute has_answer with math-verify for runs collected without it",
    )
    args = parser.parse_args(argv)

    out = Path(args.trajectories_dir)
    rows = read_shards(out, args.pattern)
    if not rows:
        print(f"No rows matched {out / args.pattern}")
        return 1

    by_example: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_example[row["example_index"]].append(row)

    correct = np.asarray([row["correct"] for row in rows], dtype=bool)
    truncated = np.asarray([row["truncated"] for row in rows], dtype=bool)
    lengths = np.asarray([row["response_length"] for row in rows], dtype=np.int64)
    incorrect = ~correct

    has_answer = column(rows, "has_answer")
    if has_answer is None and args.reparse:
        has_answer = reparse_answers(rows)
    has_boxed = column(rows, "has_boxed")
    if has_boxed is None and all(row.get("response") is not None for row in rows):
        # Cheap enough to always recover: it is a substring test, not a parse.
        has_boxed = np.asarray(["\\boxed" in str(row["response"]) for row in rows], dtype=bool)

    # Number of correct traces per prompt: the shape of this histogram is what
    # decides whether a prefilter has anything to choose between.
    solved = np.asarray([sum(r["correct"] for r in traces) for traces in by_example.values()])
    per_prompt = np.asarray([len(traces) for traces in by_example.values()])
    total = len(rows)

    report = {
        "prompts": len(by_example),
        "traces": total,
        "traces_per_prompt": summarize(per_prompt),
        "pass_at_1": float(correct.mean()),
        "pass_at_k": float((solved > 0).mean()),
        "all_correct_prompts": int((solved == per_prompt).sum()),
        "none_correct_prompts": int((solved == 0).sum()),
        "mixed_prompts": int(((solved > 0) & (solved < per_prompt)).sum()),
        "outcomes": {
            "correct": share(int(correct.sum()), total),
            "incorrect": share(int(incorrect.sum()), total),
            "truncated": share(int(truncated.sum()), total),
            "no_parseable_answer": share(int((~has_answer).sum()), total) if has_answer is not None else None,
            "no_boxed_answer": share(int((~has_boxed).sum()), total) if has_boxed is not None else None,
        },
        "incorrect_breakdown": incorrect_breakdown(incorrect, truncated, has_answer),
        "truncated_fraction": float(truncated.mean()),
        "correct_given_truncated": float(correct[truncated].mean()) if truncated.any() else None,
        "response_length": summarize(lengths),
        "response_length_correct": summarize(lengths[correct]),
        "response_length_incorrect": summarize(lengths[incorrect]),
        "response_length_truncated": summarize(lengths[truncated]),
        "by_source": by_source(rows, correct, truncated),
        "solved_histogram": {int(k): int(v) for k, v in zip(*np.unique(solved, return_counts=True))},
        "throughput": collection_throughput(out, args.summary_pattern),
    }

    path = out / "rollout_report.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"prompts {report['prompts']}  traces {report['traces']}")
    print(f"pass@1 {report['pass_at_1']:.3f}   pass@k {report['pass_at_k']:.3f}")
    print(
        f"prompts all-correct {report['all_correct_prompts']}  "
        f"mixed {report['mixed_prompts']}  none-correct {report['none_correct_prompts']}"
    )

    outcomes = report["outcomes"]
    for name in ("correct", "incorrect", "truncated", "no_parseable_answer", "no_boxed_answer"):
        entry = outcomes[name]
        if entry is None:
            print(f"{name:20} n/a (field missing from these rows)")
        else:
            print(f"{name:20} {entry['count']:>7} / {report['traces']}  ({entry['fraction']:.3f})")

    breakdown = report["incorrect_breakdown"]
    print(f"of {breakdown['incorrect_total']} incorrect traces:")
    for name in ("truncated", "complete", "no_parseable_answer", "wrong_answer"):
        entry = breakdown[name]
        if entry is None:
            print(f"  {name:22} n/a (needs has_answer or --reparse)")
        else:
            print(f"  {name:22} {entry['count']:>7}  ({entry['fraction']:.3f} of incorrect)")
    if breakdown["cross"]:
        print(f"  cross-tab {breakdown['cross']}")

    for label in ("response_length", "response_length_correct", "response_length_incorrect"):
        stats = report[label]
        if not stats["n"]:
            continue
        print(
            f"{label:28} n={stats['n']}  mean {stats['mean']:.0f}  median {stats['median']:.0f}  "
            + "  ".join(f"p{p}={stats[f'p{p}']:.0f}" for p in PERCENTILES)
        )

    if report["by_source"]:
        print("accuracy by source:")
        for name, stats in report["by_source"].items():
            print(
                f"  {name:28} {stats['correct']:>6}/{stats['traces']:<6} "
                f"acc {stats['accuracy']:.3f}  truncated {stats['truncated_fraction']:.3f}"
            )

    if report["throughput"]:
        aggregate = report["throughput"]["aggregate"]
        print(
            f"throughput ({aggregate['shards']} shard(s)): "
            f"{aggregate['generated_tokens_per_s']:.0f} generated tok/s  "
            f"{aggregate['total_tokens_per_s']:.0f} total tok/s  "
            f"{aggregate['sequences_per_s']:.2f} seq/s over "
            f"{aggregate['generate_seconds_slowest_shard']:.0f}s of generate"
        )
        if aggregate["peak_gpu_memory_gib"]:
            print(f"peak GPU memory: {aggregate['peak_gpu_memory_gib']:.2f} GiB")

    print(f"correct traces per prompt: {report['solved_histogram']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
