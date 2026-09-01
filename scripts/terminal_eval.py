"""Terminal eval of one checkpoint: MATH-500 avg@4 + AIME 2025/2026 avg@16 at a
long cap, strict scored, with honest error bars (ADR 0005).

Reuses ``apod.stages.rollout_eval`` (one vLLM engine session per shard and
dataset) through a derived run dir, ``<run>/terminal_eval/cap<cap>/``:

  resolved_config.yaml                     source config, cap + max_model_len swapped
  pool/eval_problems.jsonl                 -> source pool (same MATH-500 index map
                                              as every round's monitor eval)
  pool/eval_problems_aime2526.jsonl        materialized once, like the driver does
  arms/<arm>/rounds/round_{R}/checkpoint   -> source checkpoint (symlink)
  arms/<arm>/rounds/round_{R+1}/eval/               MATH-500 shards (stage layout)
  arms/<arm>/rounds/round_{R+1}/eval_aime2526/      AIME shards
  arms/<arm>/rounds/round_{R+1}/terminal_summary.json   the table below

The stage's own "round R+1 evaluates round_R/checkpoint" resolution measures
the chosen checkpoint, exactly as the driver's terminal eval-only round does,
and its row-level resume makes a killed run pick up where it stopped.

Scoring is STRICT: no \\boxed answer = incorrect, cap-hit traces included.
Error bars per dataset: the naive Bernoulli SE sqrt(p(1-p)/(n k)) and a
problem-level cluster bootstrap 95% interval (resample problems with
replacement, keeping all k samples of a problem together), the one that
respects the between-problem variance a 30-problem set is dominated by.
Pooled sets are also split per source (aime2025 / aime2026) from the id
prefix.

Usage:
  uv run python scripts/terminal_eval.py --run-dir outputs/runs/kl50w --arm kl_mid
  uv run python scripts/terminal_eval.py --run-dir outputs/runs/kl50w --arm kl_mid \\
      --round 1 --max-new-tokens 32768 --gpus 1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apod import paths  # noqa: E402
from apod.datasets import load_examples, read_shards, write_jsonl  # noqa: E402
from apod.stages.rollout_eval import select_eval_set  # noqa: E402

DEFAULT_DATASETS = ("math500", "aime2526")


def log(msg: str) -> None:
    print(f"[terminal_eval] {msg}", flush=True)


def latest_checkpoint_round(run_dir: Path, arm: str) -> int:
    """Newest round whose checkpoint still has weights (retention prunes old ones)."""

    with_weights = [
        int(d.name.split("_")[1])
        for d in sorted((run_dir / "arms" / arm / "rounds").glob("round_*"))
        if any((d / "checkpoint").glob("*.safetensors"))
    ]
    if not with_weights:
        raise FileNotFoundError(f"no checkpoint with weights under {run_dir / 'arms' / arm}")
    return max(with_weights)


def eval_protocol(cfg, dataset: str):
    """(eval subdir, pool file, eval cfg) for ``dataset`` without touching ``cfg``."""

    scratch = copy.deepcopy(cfg)
    subdir, pool_file = select_eval_set(scratch, dataset)
    return subdir, pool_file, scratch.eval


def prepare_run_dir(source: Path, arm: str, rnd: int, cap: int, datasets: list[str]) -> Path:
    derived = source / "terminal_eval" / f"cap{cap}"
    derived.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(source / "resolved_config.yaml")
    cfg.output_dir = str(derived)
    cfg.resume = True
    cfg.sampling.max_new_tokens = cap
    # Same headroom rule as the existing configs (8192 -> 16384, 16384 -> 24576).
    cfg.engine.max_model_len = max(int(cfg.engine.max_model_len), cap + 8192)
    OmegaConf.save(config=cfg, f=derived / "resolved_config.yaml", resolve=True)

    pool = derived / "pool"
    pool.mkdir(exist_ok=True)
    for dataset in datasets:
        _, pool_file, eval_cfg = eval_protocol(cfg, dataset)
        target = pool / pool_file
        if target.exists():
            continue
        source_pool = source / "pool" / pool_file
        if source_pool.exists():
            # The monitor set: keep problem_index -> problem identical to the
            # run's own evals rather than re-deriving it from the Hub.
            target.symlink_to(source_pool.resolve())
            log(f"pool: {pool_file} -> {source_pool}")
            continue
        problems = load_examples(
            str(eval_cfg.dataset), n=int(eval_cfg.num_problems), seed=int(cfg.seed)
        )
        write_jsonl(target, problems)
        log(f"pool: wrote {len(problems)} problems to {target}")

    src_ckpt = paths.checkpoint_dir(source, arm, rnd).resolve()
    if not any(src_ckpt.glob("*.safetensors")):
        raise FileNotFoundError(f"{src_ckpt} has no weights (pruned by keep_checkpoints?)")
    link = paths.checkpoint_dir(derived, arm, rnd)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != src_ckpt:
            link.unlink()
            link.symlink_to(src_ckpt)
    elif link.exists():
        raise FileExistsError(f"{link} is a real directory, expected a symlink")
    else:
        link.symlink_to(src_ckpt)
    return derived


def run_stage(derived: Path, arm: str, stage_round: int, dataset: str, gpus: list[str]) -> None:
    """One rollout_eval --eval-only process per GPU; skipped when all markers exist."""

    subdir, _, _ = eval_protocol(OmegaConf.load(derived / "resolved_config.yaml"), dataset)
    out_dir = paths.round_dir(derived, arm, stage_round) / subdir
    if all((out_dir / f"done.shard{k}").exists() for k in range(len(gpus))):
        log(f"{dataset}: already done ({out_dir})")
        return
    procs = []
    for shard, gpu in enumerate(gpus):
        cmd = [
            sys.executable, "-m", "apod.stages.rollout_eval",
            "--run-dir", str(derived), "--arm", arm, "--round", str(stage_round),
            "--shard", str(shard), "--num-shards", str(len(gpus)),
            "--eval-only", "--eval-dataset", dataset,
        ]
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": gpu}
        log(f"launch (GPU {gpu}): {' '.join(cmd[3:])}")
        procs.append(subprocess.Popen(cmd, env=env, cwd=ROOT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"{dataset}: rollout_eval shards exited {codes}")


# --- scoring -----------------------------------------------------------------


def bootstrap_ci(per_problem: np.ndarray, resamples: int, seed: int = 0) -> tuple[float, float]:
    """Problem-level cluster bootstrap: a problem's k samples move together."""

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_problem), size=(resamples, len(per_problem)))
    means = per_problem[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(rows: list[dict], num_samples: int, resamples: int) -> dict:
    by_problem: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_problem[r["problem_index"]].append(r)
    short = {i: len(s) for i, s in by_problem.items() if len(s) != num_samples}
    if short:
        raise RuntimeError(f"problems without exactly {num_samples} samples: {short}")
    n, k = len(by_problem), num_samples
    strict = lambda r: bool(r["correct"] and r["has_boxed"])  # noqa: E731
    avg = np.array([np.mean([strict(r) for r in s]) for s in by_problem.values()])
    passed = np.array([float(any(strict(r) for r in s)) for s in by_problem.values()])
    p = float(avg.mean())
    return {
        "num_problems": n,
        "num_samples": k,
        "avg_at_k": p,
        "naive_se": float(np.sqrt(p * (1 - p) / (n * k))),
        "avg_at_k_ci95": bootstrap_ci(avg, resamples),
        "pass_at_k": float(passed.mean()),
        "pass_at_k_ci95": bootstrap_ci(passed, resamples),
        "loose_avg_at_k": float(np.mean([r["correct"] for r in rows])),
        "cap_hit_rate": float(np.mean([r["truncated"] for r in rows])),
        "mean_response_length": float(np.mean([r["response_length"] for r in rows])),
        "bootstrap_resamples": resamples,
    }


def summarize_dataset(rows: list[dict], eval_cfg, resamples: int) -> dict:
    k, n = int(eval_cfg.num_samples), int(eval_cfg.num_problems)
    if len(rows) != n * k:
        raise RuntimeError(f"{len(rows)} rows, expected {n} problems x {k} samples = {n * k}")
    summary = summarize(rows, k, resamples)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["id"].split(":")[0]].append(r)
    if len(by_source) > 1:  # pooled set: per-source split from the id prefix
        summary["sources"] = {
            src: summarize(sub, k, resamples) for src, sub in sorted(by_source.items())
        }
    return summary


HEADER = f"{'dataset':<12}{'n':>5}{'k':>4}{'avg@k':>8}{'se':>7}{'95% CI (cluster boot)':>24}{'pass@k':>8}{'cap_hit':>9}{'len':>7}"


def fmt_row(name: str, m: dict) -> str:
    lo, hi = m["avg_at_k_ci95"]
    return (
        f"{name:<12}{m['num_problems']:>5}{m['num_samples']:>4}{m['avg_at_k']:>8.4f}"
        f"{m['naive_se']:>7.3f}{f'[{lo:.3f}, {hi:.3f}]':>24}{m['pass_at_k']:>8.3f}"
        f"{m['cap_hit_rate']:>9.3f}{m['mean_response_length']:>7.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--round", type=int, default=None, dest="round_index",
                        help="evaluate round_R/checkpoint (default: newest round with weights)")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                        help="eval set keys (conf/eval/<name>.yaml or the run's monitor set)")
    parser.add_argument("--gpus", default=None,
                        help="comma-separated GPU ids, one shard each (default: 0..num_gpus-1)")
    parser.add_argument("--resamples", type=int, default=10000, help="bootstrap resamples")
    args = parser.parse_args()

    source = args.run_dir.resolve()
    cfg = OmegaConf.load(source / "resolved_config.yaml")
    gpus = (args.gpus or ",".join(str(k) for k in range(int(cfg.num_gpus)))).split(",")
    rnd = args.round_index if args.round_index is not None else latest_checkpoint_round(source, args.arm)
    stage_round = rnd + 1
    cap = args.max_new_tokens

    derived = prepare_run_dir(source, args.arm, rnd, cap, args.datasets)
    checkpoint = paths.checkpoint_dir(source, args.arm, rnd)
    log(f"{args.arm} round_{rnd:02d}/checkpoint at cap {cap} on {gpus} -> {derived}")
    for dataset in args.datasets:
        run_stage(derived, args.arm, stage_round, dataset, gpus)

    out_round = paths.round_dir(derived, args.arm, stage_round)
    derived_cfg = OmegaConf.load(derived / "resolved_config.yaml")
    results = {}
    for dataset in args.datasets:
        subdir, _, eval_cfg = eval_protocol(derived_cfg, dataset)
        rows = read_shards(out_round / subdir, "eval.shard*.jsonl")
        results[dataset] = summarize_dataset(rows, eval_cfg, args.resamples)

    summary = {
        "arm": args.arm,
        "checkpoint_round": rnd,
        "checkpoint": str(checkpoint),
        "max_new_tokens": cap,
        "sampling": OmegaConf.to_container(derived_cfg.sampling, resolve=True),
        "scoring": "strict (correct and has_boxed; cap-hit = incorrect)",
        "datasets": results,
    }
    with open(out_round / "terminal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n== {args.arm} round_{rnd:02d}/checkpoint  cap {cap}  strict scoring  ({checkpoint})")
    print(HEADER)
    for dataset, m in results.items():
        print(fmt_row(dataset, m))
        for src, sub in m.get("sources", {}).items():
            print(fmt_row(f"  {src}", sub))
    print(f"\nwritten: {out_round / 'terminal_summary.json'}")


if __name__ == "__main__":
    main()
