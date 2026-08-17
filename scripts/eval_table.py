"""Per-round eval table with a subset-matched round-0 baseline.

Intermediate rounds evaluate only the first ``intermediate_num_problems``
problems, and that prefix is HARDER than the full set (baseline strict 0.2375
on problems 0-99 vs 0.2725 on all 500), so comparing an intermediate round
against the full-set round-0 numbers flatters it by ~3.5 strict points. This
script recomputes round 0 restricted to exactly the problem set each round
evaluated, and reports deltas against that matched baseline.

Usage: uv run python scripts/eval_table.py --run-dir outputs/runs/apod
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_eval_rows(round_dir: Path) -> list[dict]:
    rows = []
    for shard in sorted((round_dir / "eval").glob("eval.shard*.jsonl")):
        with shard.open() as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def metrics(rows: list[dict]) -> dict | None:
    """strict/loose avg@n and pass@n, cap_hit, mean length over eval rows."""

    if not rows:
        return None
    by_problem: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_problem[r["problem_index"]].append(r)
    n = len(by_problem)
    strict = [r["correct"] and r["has_boxed"] for r in rows]
    loose = [r["correct"] for r in rows]
    return {
        "n_problems": n,
        "strict": sum(strict) / len(rows),
        "loose": sum(loose) / len(rows),
        "strict_pass": sum(
            any(s["correct"] and s["has_boxed"] for s in samples)
            for samples in by_problem.values()
        )
        / n,
        "pass": sum(any(s["correct"] for s in samples) for samples in by_problem.values()) / n,
        "cap_hit": sum(r["truncated"] for r in rows) / len(rows),
        "len": sum(r["response_length"] for r in rows) / len(rows),
    }


def strict_by_problem(rows: list[dict]) -> dict[int, float]:
    """Per-problem strict avg@n -- the unit for paired tests."""

    by_problem: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_problem[r["problem_index"]].append(r)
    return {
        idx: sum(s["correct"] and s["has_boxed"] for s in samples) / len(samples)
        for idx, samples in by_problem.items()
    }


def paired_t(cur: dict[int, float], ref: dict[int, float]) -> str:
    """Paired per-problem t on strict avg@n; the pairing matters because most
    problems are unchanged round-to-round and unpaired SEs understate precision."""

    common = sorted(cur.keys() & ref.keys())
    deltas = [cur[i] - ref[i] for i in common]
    n = len(deltas)
    if n == 0:
        # e.g. probe run dirs whose round_00 holds only a checkpoint symlink
        return "n/a (no overlapping problems in reference round)"
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    se = (var / n) ** 0.5
    up = sum(d > 0 for d in deltas)
    down = sum(d < 0 for d in deltas)
    t = mean / se if se > 0 else float("inf")
    return f"{mean:+.4f} t={t:.2f} ({up} up/{down} down/{n - up - down} same)"


def fmt(m: dict, base: dict | None = None) -> str:
    def delta(key: str) -> str:
        if base is None:
            return ""
        return f" ({m[key] - base[key]:+.3f})"

    return (
        f"strict {m['strict']:.4f}{delta('strict')}  loose {m['loose']:.4f}{delta('loose')}  "
        f"s_pass@n {m['strict_pass']:.3f}{delta('strict_pass')}  pass@n {m['pass']:.3f}{delta('pass')}  "
        f"cap {m['cap_hit']:.3f}{delta('cap_hit')}  len {m['len']:.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    for arm_dir in sorted((args.run_dir / "arms").iterdir()):
        rounds = sorted((arm_dir / "rounds").glob("round_*"))
        base_rows = read_eval_rows(rounds[0]) if rounds else []
        print(f"\n== {arm_dir.name}")
        prev_strict: dict[int, float] | None = None
        for round_dir in rounds:
            rows = read_eval_rows(round_dir)
            m = metrics(rows)
            if m is None:
                print(f"  {round_dir.name}: (no eval rows yet)")
                continue
            cur_strict = strict_by_problem(rows)
            if round_dir is rounds[0]:
                print(f"  {round_dir.name}: {fmt(m)}  [n={m['n_problems']}]")
                prev_strict = cur_strict
                continue
            # Baseline restricted to exactly the problems this round evaluated.
            evaluated = {r["problem_index"] for r in rows}
            base_sub = [r for r in base_rows if r["problem_index"] in evaluated]
            base = metrics(base_sub)
            print(f"  {round_dir.name}: {fmt(m, base)}  [n={m['n_problems']}, deltas vs r0 same problems]")
            line = f"      paired strict vs r0: {paired_t(cur_strict, strict_by_problem(base_sub))}"
            if prev_strict is not None:
                line += f"  |  vs prev round: {paired_t(cur_strict, prev_strict)}"
            print(line)
            prev_strict = cur_strict


if __name__ == "__main__":
    main()
