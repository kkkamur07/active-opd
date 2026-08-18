"""Per-trajectory KL drift across a training round (USER 2026-08-18: "per
trajectory kl logging would help" -- the band-drift question behind kl_mid's
round-1 peak -> round-2 regression).

For a given (arm, round) the round's rollouts were scored BEFORE training
with the generating checkpoint (<round>/oracle). This script re-scores the
SAME trajectories with the POST-train checkpoint (<round>/checkpoint) into
<round>/oracle_post, then reports per-trajectory before/after reverse KL:

  - trained vs untrained rows (selected.jsonl membership), by pre-train
    tertile: did training pull the student toward the teacher only on the
    trained band, everywhere, or nowhere?
  - band drift: rank each prompt's 12 rollouts by POST-train reverse KL and
    measure overlap of the post mid-tertile with the pre mid-tertile. Low
    overlap = the learnability band moved -- next round's mid bucket is a
    different kind of trajectory, the non-stationarity hypothesis for the
    regression.

Exact same quantity as training-time logging would give (fp32 log-softmax
over the full vocab per position) without trainer surgery: the liger-fused
loss never materializes per-sequence KL, so in-trainer logging would need a
custom compute_loss; this gets the numbers with the existing scorer.

GPU phase (2 shards):  uv run python scripts/kl_drift.py --arm kl_mid --round 1
CPU analysis:          uv run python scripts/kl_drift.py --arm kl_mid --round 1 --analyze
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM_ROLLOUTS = 12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=ROOT / "outputs/runs/oracle16k")
    p.add_argument("--arm", default="kl_mid")
    p.add_argument("--round", type=int, default=1, dest="round_index")
    p.add_argument("--analyze", action="store_true")
    return p.parse_args()


def round_dir(args) -> Path:
    return args.run_dir / "arms" / args.arm / "rounds" / f"round_{args.round_index:02d}"


def read_rows(directory: Path) -> dict[tuple[int, int], dict]:
    rows = {}
    for shard in sorted(directory.glob("oracle_kl.shard*.jsonl")):
        for line in shard.open():
            r = json.loads(line)
            rows[(r["example_index"], r["rollout_index"])] = r
    return rows


def score_post(args) -> None:
    rdir = round_dir(args)
    out_dir = rdir / "oracle_post"
    existing = len(read_rows(out_dir)) if out_dir.exists() else 0
    if existing >= 128 * NUM_ROLLOUTS:
        print(f"[drift] post-scoring already done ({existing} rows)")
        return
    checkpoint = rdir / "checkpoint"
    assert (checkpoint / "config.json").exists(), f"no post checkpoint at {checkpoint}"
    procs = []
    for shard in (0, 1):
        cmd = [
            sys.executable, "scripts/oracle_kl.py",
            "--run-dir", str(args.run_dir), "--arm", args.arm,
            "--round", str(args.round_index),
            "--shard", str(shard), "--num-shards", "2",
            "--student-path", str(checkpoint),
            "--tokens-dir", str(rdir / "rollouts" / "tokens"),
            "--out-dir", str(out_dir),
        ]
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": str(shard)}
        print(f"[drift] launch post-score shard{shard} (student {checkpoint})", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=ROOT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"post-scoring exited {codes}")


def analyze(args) -> None:
    rdir = round_dir(args)
    pre = read_rows(rdir / "oracle")
    post = read_rows(rdir / "oracle_post")
    selected = {
        (r["example_index"], r["rollout_index"])
        for r in map(json.loads, (rdir / "selected" / "selected.jsonl").open())
    }
    keys = sorted(set(pre) & set(post))
    print(f"[drift] {args.arm} round {args.round_index}: {len(keys)} trajectories "
          f"(pre {len(pre)}, post {len(post)}, trained {len(selected)})")

    # Pre-train tertile per prompt (ranks 0-3 high / 4-7 mid / 8-11 low).
    by_example: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for k in keys:
        by_example[k[0]].append(k)
    tertile_of: dict[tuple[int, int], str] = {}
    post_mid_overlap = []
    for example_index, ks in by_example.items():
        pre_ranked = sorted(ks, key=lambda k: pre[k]["mean_reverse_kl"], reverse=True)
        for rank, k in enumerate(pre_ranked):
            tertile_of[k] = ("high", "mid", "low")[min(rank // 4, 2)]
        post_ranked = sorted(ks, key=lambda k: post[k]["mean_reverse_kl"], reverse=True)
        pre_mid = set(pre_ranked[4:8])
        post_mid = set(post_ranked[4:8])
        post_mid_overlap.append(len(pre_mid & post_mid))

    print(f"\n{'group':24s} {'n':>5s} {'pre rkl':>9s} {'post rkl':>9s} {'delta':>9s}")
    for label, subset in (
        ("trained (selected)", [k for k in keys if k in selected]),
        ("untrained", [k for k in keys if k not in selected]),
        *((f"tertile {t} / {'trained' if s else 'untrained'}",
           [k for k in keys if tertile_of[k] == t and ((k in selected) == s)])
          for t in ("high", "mid", "low") for s in (True, False)),
    ):
        subset = [k for k in subset]
        if not subset:
            continue
        pre_m = sum(pre[k]["mean_reverse_kl"] for k in subset) / len(subset)
        post_m = sum(post[k]["mean_reverse_kl"] for k in subset) / len(subset)
        print(f"{label:24s} {len(subset):5d} {pre_m:9.4f} {post_m:9.4f} {post_m - pre_m:+9.4f}")

    mean_overlap = sum(post_mid_overlap) / len(post_mid_overlap)
    print(f"\nband drift: post-train mid-tertile overlaps pre-train mid-tertile "
          f"{mean_overlap:.2f}/4 (no drift = 4.0, random = 1.33)")


def main() -> None:
    args = parse_args()
    if args.analyze:
        analyze(args)
    else:
        score_post(args)


if __name__ == "__main__":
    main()
