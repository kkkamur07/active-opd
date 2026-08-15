"""Oracle-KL diagnostic (todo: run ONCE, standalone; USER 2026-08-15: run
after the apod run finishes).

Computes the ACTUAL per-trajectory reverse KL(student || teacher) -- the exact
quantity the GKD objective (beta=1.0) minimizes -- for every stored round-0
rollout, then asks whether the cheap selection statistic (mean token entropy)
ranks trajectories the way the oracle does. This is the attribution tool for
the entropy-vs-random null: if entropy-top4 and KL-top4 pick nearly the same
trajectories, the selection PREMISE is weak; if they pick differently, mean
entropy is the wrong proxy and a better statistic is the fix.

Per trajectory it writes: mean reverse KL over response tokens, plus the
stored entropy/logprob joined from the entropy stage. Analysis (per-prompt
rank correlation, top-4 overlap, KL range percentiles) runs with --analyze
after all shards finish.

Run (per GPU shard):
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/oracle_kl.py \
      --run-dir outputs/runs/apod --arm entropy_top4 --round 0 \
      --shard 0 --num-shards 2
  CUDA_VISIBLE_DEVICES=1 ... --shard 1 --num-shards 2
Then:
  uv run python scripts/oracle_kl.py --run-dir outputs/runs/apod \
      --arm entropy_top4 --round 0 --analyze
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STUDENT_ID = "Qwen/Qwen3.5-2B"
TEACHER_ID = "Qwen/Qwen3.5-9B"
POSITION_CHUNK = 1024  # fp32 log-softmax over 248k vocab, 1024 positions ~ 1 GiB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--arm", default="entropy_top4")
    p.add_argument("--round", type=int, default=0, dest="round_index")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--analyze", action="store_true", help="analyze finished shards, no GPU")
    return p.parse_args()


def round_dir(args) -> Path:
    return args.run_dir / "arms" / args.arm / "rounds" / f"round_{args.round_index:02d}"


def out_path(args, shard: int) -> Path:
    return round_dir(args) / "oracle" / f"oracle_kl.shard{shard}.jsonl"


def compute(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda"
    npzs = sorted((round_dir(args) / "rollouts" / "tokens").glob("example_*.npz"))
    npzs = [p for i, p in enumerate(npzs) if i % args.num_shards == args.shard]

    out = out_path(args, args.shard)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as f:
            done = {(r["example_index"], r["rollout_index"]) for r in map(json.loads, f)}

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, dtype=torch.bfloat16, device_map=device
    ).eval()
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=torch.bfloat16, device_map=device
    ).eval()

    with out.open("a") as f:
        for npz_path in npzs:
            example_index = int(npz_path.stem.split("_")[1])
            with np.load(npz_path, allow_pickle=True) as d:
                ids_all = d["input_ids"]
                rlens = d["response_lengths"]
                plen = int(d["prompt_length"])
                truncated = d["truncated"]
            for k in range(ids_all.shape[0]):
                if (example_index, k) in done:
                    continue
                rlen = int(rlens[k])
                ids = torch.tensor(ids_all[k, : plen + rlen][None], device=device)
                with torch.no_grad():
                    t_logits = teacher(input_ids=ids).logits[0]
                    s_logits = student(input_ids=ids).logits[0]
                # Reverse KL at response positions: prediction at position t-1
                # scores the token emitted at t, so slice [plen-1, plen+rlen-1).
                kl_sum, n = 0.0, 0
                for lo in range(plen - 1, plen + rlen - 1, POSITION_CHUNK):
                    hi = min(lo + POSITION_CHUNK, plen + rlen - 1)
                    lp_s = torch.log_softmax(s_logits[lo:hi].float(), dim=-1)
                    lp_t = torch.log_softmax(t_logits[lo:hi].float(), dim=-1)
                    kl = (lp_s.exp() * (lp_s - lp_t)).sum(-1)  # KL(S||T) per position
                    kl_sum += float(kl.sum())
                    n += kl.shape[0]
                    del lp_s, lp_t, kl
                del t_logits, s_logits
                torch.cuda.empty_cache()
                f.write(
                    json.dumps(
                        {
                            "example_index": example_index,
                            "rollout_index": k,
                            "mean_reverse_kl": kl_sum / max(n, 1),
                            "response_length": rlen,
                            "truncated": bool(truncated[k]),
                        }
                    )
                    + "\n"
                )
                f.flush()
            print(f"example {example_index} done", flush=True)


def analyze(args) -> None:
    rdir = round_dir(args)
    oracle = []
    for shard_file in sorted((rdir / "oracle").glob("oracle_kl.shard*.jsonl")):
        with shard_file.open() as f:
            oracle.extend(map(json.loads, f))
    entropy = []
    for shard_file in sorted((rdir / "entropy").glob("entropy.shard*.jsonl")):
        with shard_file.open() as f:
            entropy.extend(map(json.loads, f))
    ent = {(r["example_index"], r["rollout_index"]): r["entropy"] for r in entropy}

    by_example: dict[int, list[dict]] = defaultdict(list)
    for r in oracle:
        r["entropy"] = ent.get((r["example_index"], r["rollout_index"]))
        by_example[r["example_index"]].append(r)

    kls = np.array([r["mean_reverse_kl"] for r in oracle])
    print(f"n trajectories: {len(oracle)}  (examples: {len(by_example)})")
    print(
        "reverse KL distribution: "
        + "  ".join(f"p{p}={np.percentile(kls, p):.4f}" for p in (5, 25, 50, 75, 95))
        + f"  mean={kls.mean():.4f}"
    )

    # Per-example Spearman rank correlation entropy vs oracle KL, and top-4
    # selection overlap between the two statistics (and expected-random 4/12).
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean()
        ry -= ry.mean()
        denom = np.sqrt((rx**2).sum() * (ry**2).sum())
        return float((rx * ry).sum() / denom) if denom > 0 else float("nan")

    rhos, overlaps = [], []
    for rows in by_example.values():
        if len(rows) < 2 or any(r["entropy"] is None for r in rows):
            continue
        e = [r["entropy"] for r in rows]
        k = [r["mean_reverse_kl"] for r in rows]
        rho = spearman(e, k)
        if not np.isnan(rho):
            rhos.append(rho)
        top_e = set(np.argsort(e)[::-1][:4])
        top_k = set(np.argsort(k)[::-1][:4])
        overlaps.append(len(top_e & top_k))
    print(
        f"per-example Spearman(entropy, oracle KL): mean {np.mean(rhos):+.3f}  "
        f"median {np.median(rhos):+.3f}  (n={len(rhos)})"
    )
    print(
        f"top-4 overlap entropy-vs-oracle: mean {np.mean(overlaps):.2f}/4  "
        f"(random baseline 1.33/4)"
    )


def main() -> None:
    args = parse_args()
    if args.analyze:
        analyze(args)
    else:
        compute(args)


if __name__ == "__main__":
    main()
