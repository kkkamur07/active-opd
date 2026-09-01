"""CPU stand-ins for the GPU stages, for ``driver.dry_run=true``.

``run_stub(cmd, run_dir)`` takes the exact command line the driver would
have launched and writes what that stage writes -- rows, npz, checkpoint
files, done-markers -- with synthetic values, so the driver's loop (step
accounting, resume from any refresh boundary, arm / question-source /
selection dispatch, checkpoint pruning, metrics rows, plot) can be
exercised in seconds without a GPU. The stubs mirror each stage's resume
and validation behaviour where the driver depends on it (train refuses to
run without the previous optimizer state under persist_optimizer, the
oracle shards by npz position, eval markers only after every row).
Synthetic accuracies drift upward with the step and differ per arm so the
plot has distinguishable curves; nothing here is a measurement.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from apod import paths
from apod.datasets.io import append_jsonl, read_jsonl, save_npz, write_jsonl


def _parse(cmd: list[str]) -> tuple[str, dict[str, Any]]:
    """(stage module or script, {flag: value | True}) from a launch command."""

    stage = None
    args: dict[str, Any] = {}
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok == "-m":
            stage = cmd[i + 1]  # the LAST -m wins: torchrun -m ... -m apod.stages.train
            i += 2
            continue
        if tok.endswith("oracle_kl.py"):
            stage = "oracle_kl"
        if tok.startswith("--"):
            key = tok[2:]
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("-"):
                args[key] = cmd[i + 1]
                i += 2
                continue
            args[key] = True
        i += 1
    if stage is None:
        raise ValueError(f"dry-run cannot identify the stage in {cmd}")
    return stage, args


def _rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("/".join(str(p) for p in parts).encode()))


def _arm_offset(arm: str) -> float:
    return (zlib.crc32(arm.encode()) % 7) / 100.0


def run_stub(cmd: list[str], run_dir: Path) -> None:
    stage, args = _parse(cmd)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    arm, refresh = args["arm"], int(args["round"])
    rdir = paths.round_dir(run_dir, arm, refresh)
    if stage == "apod.stages.rollout_eval":
        _rollout_eval(cfg, run_dir, rdir, arm, refresh, args)
    elif stage == "apod.stages.entropy":
        _entropy(rdir, int(args["shard"]), int(args["num-shards"]))
    elif stage == "oracle_kl":
        _oracle(rdir, int(args["shard"]), int(args["num-shards"]), args.get("estimator", "exact"))
    elif stage == "apod.stages.train":
        _train(cfg, run_dir, rdir, arm, refresh, args)
    else:
        raise ValueError(f"dry-run has no stub for {stage}")


def _rollout_eval(cfg, run_dir: Path, rdir: Path, arm: str, refresh: int, args: dict[str, Any]) -> None:
    shard, num_shards = int(args["shard"]), int(args["num-shards"])
    dataset = args.get("eval-dataset")
    if dataset is None or dataset == str(cfg.eval.dataset):
        subdir, pool_file, protocol = "eval", "eval_problems.jsonl", cfg.eval
    else:
        subdir, pool_file, protocol = f"eval_{dataset}", f"eval_problems_{dataset}.jsonl", cfg.eval_sets[dataset]
    problems = read_jsonl(run_dir / "pool" / pool_file)
    step = refresh * int(cfg.driver.refresh_every)
    rng = _rng("eval", arm, refresh, shard, subdir)
    p = min(0.9, 0.25 + 0.004 * step + _arm_offset(arm))
    eval_dir = rdir / subdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(problems):
        if i % num_shards != shard:
            continue
        for s in range(int(protocol.num_samples)):
            truncated = bool(rng.random() < max(0.05, 0.5 - 0.003 * step))
            correct = bool(rng.random() < p) and not truncated
            rows.append({
                "problem_index": i, "sample_index": s, "id": ex["id"],
                "response_length": int(rng.integers(500, int(cfg.sampling.max_new_tokens))),
                "truncated": truncated, "correct": correct, "has_answer": correct or bool(rng.random() < 0.5),
                "has_boxed": correct and bool(rng.random() < 0.95), "gold_parsed": True,
            })
    write_jsonl(eval_dir / f"eval.shard{shard}.jsonl", rows)
    (eval_dir / f"done.shard{shard}").touch()
    if args.get("eval-only"):
        return

    pool = read_jsonl(run_dir / "pool" / "prompts.jsonl")
    rollouts_dir = rdir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    (rollouts_dir / "model_path.json").write_text(json.dumps({"model_path": _model_path(cfg, run_dir, arm, refresh)}))
    num_rollouts = int(cfg.rollout.num_rollouts)
    rng = _rng("rollouts", arm, refresh, shard)
    traj_rows = []
    for row in pool:
        if int(row["round"]) != refresh or int(row["example_index"]) % num_shards != shard:
            continue
        lengths = rng.integers(4, 12, size=num_rollouts)
        truncated = rng.random(num_rollouts) < 0.3
        ids = np.full((num_rollouts, 3 + int(lengths.max())), 7, dtype=np.int32)
        save_npz(rollouts_dir / "tokens" / f"example_{int(row['example_index']):05d}.npz", {
            "input_ids": ids, "prompt_length": 3, "response_lengths": lengths, "truncated": truncated,
            "responses": [f"r{j}" for j in range(num_rollouts)],
            "finish_reasons": ["length" if t else "stop" for t in truncated],
        })
        for j in range(num_rollouts):
            correct = bool(rng.random() < p) and not bool(truncated[j])
            traj_rows.append({
                "example_index": int(row["example_index"]), "rollout_index": j, "id": row["id"],
                "prompt_length": 3, "response": f"r{j}", "response_length": int(lengths[j]),
                "truncated": bool(truncated[j]), "finish_reason": "length" if truncated[j] else "stop",
                "correct": correct, "has_answer": True, "has_boxed": correct, "seed": int(cfg.seed),
            })
    write_jsonl(rollouts_dir / f"trajectories.shard{shard}.jsonl", traj_rows)
    (rollouts_dir / f"done.shard{shard}").touch()


def _model_path(cfg, run_dir: Path, arm: str, refresh: int) -> str:
    if refresh == 0:
        return str(cfg.model.student_id)
    ckpt = paths.checkpoint_dir(run_dir, arm, refresh - 1)
    if not any(ckpt.glob("*.safetensors")):  # the stages' resolve_model_path rule
        raise FileNotFoundError(f"refresh {refresh} expects weights in {ckpt}")
    return str(ckpt)


def _examples(rdir: Path) -> list[tuple[int, Path]]:
    return [(int(p.stem.split("_")[1]), p) for p in sorted((rdir / "rollouts" / "tokens").glob("example_*.npz"))]


def _entropy(rdir: Path, shard: int, num_shards: int) -> None:
    out = rdir / "entropy"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for example_index, npz in _examples(rdir):
        if example_index % num_shards != shard:
            continue
        with np.load(npz, allow_pickle=True) as d:
            n = int(d["input_ids"].shape[0])
        rng = _rng("entropy", rdir, example_index)
        for j in range(n):
            rows.append({"example_index": example_index, "rollout_index": j,
                         "entropy": float(rng.random()), "mean_logprob": -float(rng.random()), "scored_tokens": 8})
    write_jsonl(out / f"entropy.shard{shard}.jsonl", rows)
    (out / f"done.shard{shard}").touch()


def _oracle(rdir: Path, shard: int, num_shards: int, estimator: str) -> None:
    out = rdir / "oracle"
    out.mkdir(parents=True, exist_ok=True)
    stem = "oracle_kl_mc" if estimator == "mc" else "oracle_kl"
    path = out / f"{stem}.shard{shard}.jsonl"
    # scripts/oracle_kl.py shards by POSITION in the sorted npz list, not by
    # example_index, and appends.
    for pos, (example_index, npz) in enumerate(_examples(rdir)):
        if pos % num_shards != shard:
            continue
        with np.load(npz, allow_pickle=True) as d:
            n, lengths, truncated = int(d["input_ids"].shape[0]), d["response_lengths"], d["truncated"]
        rng = _rng("oracle", rdir, example_index)
        for j in range(n):
            kl = float(rng.random())
            scores = {"rkl_mc": kl} if estimator == "mc" else {
                "mean_reverse_kl": kl, "mean_forward_kl": kl * 1.5, "student_entropy": float(rng.random()),
                "teacher_entropy": float(rng.random()), "overlap_ratio_top16": float(rng.random()),
                "overlap_advantage_top16": float(rng.random()),
            }
            append_jsonl(path, {"example_index": example_index, "rollout_index": j, **scores,
                                "response_length": int(lengths[j]), "truncated": bool(truncated[j])})


def _train(cfg, run_dir: Path, rdir: Path, arm: str, refresh: int, args: dict[str, Any]) -> None:
    if "global-step-offset" not in args:
        raise ValueError("train stage launched without --global-step-offset")
    offset = int(args["global-step-offset"])
    train_dir, ckpt = rdir / "train", rdir / "checkpoint"
    if bool(cfg.resume) and (train_dir / "done.shard0").exists() and (ckpt / "config.json").exists():
        return
    selected = read_jsonl(rdir / "selected" / "selected.jsonl")
    if not selected:
        raise RuntimeError(f"no selected trajectories in {rdir / 'selected'}")
    _model_path(cfg, run_dir, arm, refresh)  # the weights must exist
    persist = bool(cfg.train.get("persist_optimizer") or False)
    if persist and refresh > 0:
        prev = paths.checkpoint_dir(run_dir, arm, refresh - 1) / "optimizer_state.pt"
        if not prev.exists():
            raise FileNotFoundError(f"train.persist_optimizer is set but {prev} is missing")
    steps = len(selected) // int(cfg.train.effective_batch)
    rng = _rng("train", arm, refresh)
    losses = [float(0.1 - 0.0005 * (offset + s) + 0.01 * rng.random()) for s in range(steps)]
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text(json.dumps({"dry_run": True, "refresh": refresh}))
    (ckpt / "model.safetensors").write_bytes(b"dry-run weights")
    if persist:
        (ckpt / "optimizer_state.pt").write_bytes(b"dry-run optimizer state")
    train_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_dir / "log_history.jsonl", [{"step": s + 1, "global_step": offset + s + 1, "loss": l} for s, l in enumerate(losses)])
    (train_dir / "summary.json").write_text(json.dumps({
        "num_trajectories": len(selected), "tokens_trained": sum(int(s["response_length"]) for s in selected),
        "tail_truncated_rows": 0, "train_loss_mean": sum(losses) / len(losses), "train_loss_final": losses[-1],
        "wall_clock_s": 0.0, "steps": steps, "global_step_offset": offset,
    }, indent=2))
    (train_dir / "done.shard0").touch()
