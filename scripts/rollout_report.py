"""Read the collected trajectory shards and report how good the rollouts are.

This is the gate before spending GPU time on entropy: if almost every trace is
truncated, or every prompt is solved 16/16, the prompt pool is wrong and no
acquisition score will rescue it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", default="outputs/trajectories")
    parser.add_argument("--pattern", default="trajectories.shard*.jsonl")
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

    # Number of correct traces per prompt: the shape of this histogram is what
    # decides whether a prefilter has anything to choose between.
    solved = np.asarray([sum(r["correct"] for r in traces) for traces in by_example.values()])
    per_prompt = np.asarray([len(traces) for traces in by_example.values()])

    report = {
        "prompts": len(by_example),
        "traces": len(rows),
        "traces_per_prompt": summarize(per_prompt),
        "pass_at_1": float(correct.mean()),
        "pass_at_k": float((solved > 0).mean()),
        "all_correct_prompts": int((solved == per_prompt).sum()),
        "none_correct_prompts": int((solved == 0).sum()),
        "mixed_prompts": int(((solved > 0) & (solved < per_prompt)).sum()),
        "truncated_fraction": float(truncated.mean()),
        "correct_given_truncated": float(correct[truncated].mean()) if truncated.any() else None,
        "response_length": summarize(lengths),
        "response_length_correct": summarize(lengths[correct]),
        "response_length_incorrect": summarize(lengths[~correct]),
        "solved_histogram": {int(k): int(v) for k, v in zip(*np.unique(solved, return_counts=True))},
    }

    path = out / "rollout_report.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"prompts {report['prompts']}  traces {report['traces']}")
    print(f"pass@1 {report['pass_at_1']:.3f}   pass@k {report['pass_at_k']:.3f}")
    print(
        f"prompts all-correct {report['all_correct_prompts']}  "
        f"mixed {report['mixed_prompts']}  none-correct {report['none_correct_prompts']}"
    )
    print(f"truncated {report['truncated_fraction']:.3f} of traces")
    length = report["response_length"]
    print(
        f"response tokens: mean {length['mean']:.0f}  median {length['median']:.0f}  "
        + "  ".join(f"p{p}={length[f'p{p}']:.0f}" for p in PERCENTILES)
    )
    print(f"correct traces per prompt: {report['solved_histogram']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
