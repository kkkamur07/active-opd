"""KL-bucket experiment driver (design LOCKED by user 2026-08-15 late).

Does training on HIGH reverse-KL trajectories drive the gain, actively hurt,
or does LOW divergence contribute nothing? Fresh everything at cap 16384 --
nothing here is comparable to the 8192 apod arms.

Locked design:
- Fresh student rollouts each round: 128 prompts x 12 at 16384 from the arm's
  CURRENT checkpoint; teacher rollouts 128 x 4 at 16384, generated once
  (teacher frozen).
- Score reverse KL, forward KL, student + teacher entropy on BOTH trajectory
  sets, RE-SCORED EVERY ROUND with the current student ("rescore every round
  that is more true").
- Buckets: per-prompt reverse-KL tertiles of the 12 student rollouts ->
  kl_high (ranks 1-4) / kl_mid (5-8) / kl_low (9-12), 512 trajectories each.
  At round 0 all arms are the identical base policy, so the tertiles
  partition ONE shared scored pool; from round 1 each arm generates and
  re-scores its own.
- 3 training rounds per arm (0,1,2) + terminal eval; identical
  hyperparameters, micro-batch 2 across all arms; eval 500x4 at 16384 every
  round; fresh 16384 base eval (the shared round-0 eval) is the baseline.

Usage:
  uv run python scripts/bucket_experiment.py --build   # run dirs + configs
  uv run python scripts/bucket_experiment.py --drive   # everything, resumable
  uv run python scripts/bucket_experiment.py --report
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
SOURCE_RUN = ROOT / "outputs/runs/apod"
RUN_DIR = ROOT / "outputs/runs/oracle16k"
TEACHER_RUN = ROOT / "outputs/runs/oracle16k_teacher"
BUCKETS = ("kl_high", "kl_mid", "kl_low")
TRAIN_ROUNDS = 3  # trains at rounds 0..2; round 3 is the terminal eval-only
NUM_ROLLOUTS = 12
TEACHER_ROLLOUTS = 4


def log(msg: str) -> None:
    print(f"[bucket] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def arm_round(run_dir: Path, arm: str, rnd: int) -> Path:
    return run_dir / "arms" / arm / "rounds" / f"round_{rnd:02d}"


def checkpoint_path(arm: str, rnd: int) -> str:
    """Model that PRODUCES round rnd artifacts (mirrors resolve_model_path)."""
    if rnd == 0:
        return "Qwen/Qwen3.5-2B"
    return str(arm_round(RUN_DIR, arm, rnd - 1) / "checkpoint")


# --- build ------------------------------------------------------------------

def _write_config(dst: Path, *, student_id: str, num_rollouts: int) -> None:
    cfg = OmegaConf.load(SOURCE_RUN / "resolved_config.yaml")
    cfg.model.student_id = student_id
    cfg.rollout.num_rollouts = num_rollouts
    cfg.sampling.max_new_tokens = 16384
    cfg.engine.max_model_len = 24576
    cfg.train.max_length = 24576
    cfg.train.per_device_train_batch_size = 2
    cfg.train.gradient_accumulation_steps = 8
    dst.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=dst / "resolved_config.yaml", resolve=True)
    if not (dst / "pool").exists():
        shutil.copytree(SOURCE_RUN / "pool", dst / "pool")


def build() -> None:
    _write_config(RUN_DIR, student_id="Qwen/Qwen3.5-2B", num_rollouts=NUM_ROLLOUTS)
    _write_config(TEACHER_RUN, student_id="Qwen/Qwen3.5-9B", num_rollouts=TEACHER_ROLLOUTS)
    log(f"configs written: {RUN_DIR}, {TEACHER_RUN} (cap 16384, micro 2 x accum 8)")


# --- stage runners (all resumable via stage-native markers) -----------------

def _shards_done(rdir: Path, sub: str) -> bool:
    return all((rdir / sub / f"done.shard{k}").exists() for k in (0, 1))


def run_rollout_eval(run_dir: Path, arm: str, rnd: int, *, eval_only: bool) -> None:
    rdir = arm_round(run_dir, arm, rnd)
    if _shards_done(rdir, "eval") and (eval_only or _shards_done(rdir, "rollouts")):
        log(f"{arm} r{rnd} rollout_eval already done")
        return
    procs = []
    for shard in (0, 1):
        cmd = [
            sys.executable, "-m", "apod.stages.rollout_eval",
            "--run-dir", str(run_dir), "--arm", arm, "--round", str(rnd),
            "--shard", str(shard), "--num-shards", "2",
        ] + (["--eval-only"] if eval_only else [])
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": str(shard)}
        log("launch: " + " ".join(cmd[3:]))
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"{arm} r{rnd} rollout_eval shards exited {codes}")


def _score_count(out_dir: Path) -> int:
    return sum(len(read_jsonl(p)) for p in out_dir.glob("oracle_kl.shard*.jsonl"))


def run_score(tokens_dir: Path, out_dir: Path, student_path: str, expected: int) -> None:
    if out_dir.exists() and _score_count(out_dir) >= expected:
        log(f"scoring already done: {out_dir}")
        return
    procs = []
    for shard in (0, 1):
        cmd = [
            sys.executable, "scripts/oracle_kl.py",
            "--run-dir", str(RUN_DIR), "--arm", BUCKETS[0], "--round", "0",
            "--shard", str(shard), "--num-shards", "2",
            "--student-path", student_path,
            "--tokens-dir", str(tokens_dir), "--out-dir", str(out_dir),
        ]
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": str(shard)}
        log("launch: score " + str(tokens_dir) + f" shard{shard} (student {student_path})")
        procs.append(subprocess.Popen(cmd, env=env, cwd=ROOT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"scoring {tokens_dir} exited {codes}")
    n = _score_count(out_dir)
    if n < expected:
        raise RuntimeError(f"scoring {out_dir}: {n} rows < expected {expected}")


def run_train(arm: str, rnd: int) -> None:
    rdir = arm_round(RUN_DIR, arm, rnd)
    if (rdir / "train" / "done.shard0").exists() and (rdir / "checkpoint" / "config.json").exists():
        log(f"{arm} r{rnd} train already done")
        return
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", "-m", "apod.stages.train",
        "--run-dir", str(RUN_DIR), "--arm", arm, "--round", str(rnd),
    ]
    log("launch: " + " ".join(cmd[5:]))
    subprocess.run(cmd, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})


def write_selection(arm: str, rnd: int, oracle_dir: Path) -> None:
    rdir = arm_round(RUN_DIR, arm, rnd)
    sel_path = rdir / "selected" / "selected.jsonl"
    if sel_path.exists():
        log(f"{arm} r{rnd} selection already written")
        return
    oracle = []
    for shard in sorted(oracle_dir.glob("oracle_kl.shard*.jsonl")):
        oracle.extend(read_jsonl(shard))
    meta = {}
    for shard in sorted((rdir / "rollouts").glob("trajectories.shard*.jsonl")):
        for r in read_jsonl(shard):
            meta[(r["example_index"], r["rollout_index"])] = r
    by_example: dict[int, list[dict]] = defaultdict(list)
    for r in oracle:
        by_example[r["example_index"]].append(r)
    tertile = BUCKETS.index(arm)
    rows = []
    for example_index in sorted(by_example):
        ranked = sorted(
            by_example[example_index], key=lambda r: r["mean_reverse_kl"], reverse=True
        )
        assert len(ranked) == NUM_ROLLOUTS, (
            f"example {example_index}: {len(ranked)} scored rollouts, expected {NUM_ROLLOUTS}"
        )
        for r in ranked[tertile * 4 : (tertile + 1) * 4]:
            key = (r["example_index"], r["rollout_index"])
            rows.append(
                {
                    "example_index": r["example_index"],
                    "rollout_index": r["rollout_index"],
                    "entropy": r.get("student_entropy"),
                    "mean_reverse_kl": r["mean_reverse_kl"],
                    "mean_forward_kl": r["mean_forward_kl"],
                    "correct": bool(meta[key]["correct"]),
                    "truncated": bool(r["truncated"]),
                    "response_length": r["response_length"],
                }
            )
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    with sel_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    kls = [r["mean_reverse_kl"] for r in rows]
    log(
        f"{arm} r{rnd}: selected {len(rows)} rows, reverse-KL "
        f"min {min(kls):.4f} median {sorted(kls)[len(kls)//2]:.4f} max {max(kls):.4f}"
    )


def _copy_shared_round0(src_arm: str, dst_arm: str) -> None:
    """Round-0 pool is shared: same base policy, ONE scored pool partitioned."""
    src, dst = arm_round(RUN_DIR, src_arm, 0), arm_round(RUN_DIR, dst_arm, 0)
    if _shards_done(dst, "eval") and _shards_done(dst, "rollouts"):
        return
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("eval", "rollouts"):
        if (dst / sub).exists():
            shutil.rmtree(dst / sub)
        shutil.copytree(src / sub, dst / sub, symlinks=True, ignore=shutil.ignore_patterns("tokens"))
    tokens_link = dst / "rollouts" / "tokens"
    if not tokens_link.exists():
        tokens_link.symlink_to(src / "rollouts" / "tokens")
    log(f"round-0 shared pool copied {src_arm} -> {dst_arm}")


# --- micro-batch probe ------------------------------------------------------
# USER 2026-08-15: "if GPU utilization is dipping or memory not being consumed
# then increase the batch size ... do like a smoke run with actual configs and
# just with variable micro batchs". The 8192-cap curve (41.2/45.3/59.2 GiB at
# micro 1/2/4) says micro 4 probably OOMs at 16384, but measure, don't
# extrapolate: trial-train each candidate on the REAL round-0 kl_high
# selection in a throwaway run dir, take the largest that survives with peak
# memory under a safety line. Eff batch stays 32 regardless.

PROBE_DIR = ROOT / "outputs/runs/oracle16k_msmoke"
PROBE_SECONDS = 900   # >= a few optimizer steps at 16k-token sequences
PROBE_MEM_CAP_MIB = 75_000  # accept only if peak < ~73 GiB of 81.9k MiB


def _gpu_peak_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    return max(int(x) for x in out.split())


def _probe_one(micro: int) -> tuple[bool, int]:
    """Trial-train round-0 kl_high at this micro-batch; return (ok, peak MiB)."""
    if PROBE_DIR.exists():
        shutil.rmtree(PROBE_DIR)
    cfg = OmegaConf.load(RUN_DIR / "resolved_config.yaml")
    cfg.train.per_device_train_batch_size = micro
    cfg.train.gradient_accumulation_steps = 32 // (2 * micro)
    PROBE_DIR.mkdir(parents=True)
    OmegaConf.save(config=cfg, f=PROBE_DIR / "resolved_config.yaml", resolve=True)
    (PROBE_DIR / "pool").symlink_to(RUN_DIR / "pool")
    src = arm_round(RUN_DIR, BUCKETS[0], 0)
    rdir = arm_round(PROBE_DIR, BUCKETS[0], 0)
    rdir.mkdir(parents=True)
    (rdir / "rollouts").symlink_to(src / "rollouts")
    (rdir / "selected").symlink_to(src / "selected")
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", "-m", "apod.stages.train",
        "--run-dir", str(PROBE_DIR), "--arm", BUCKETS[0], "--round", "0",
    ]
    log(f"micro-probe: micro {micro} x accum {32 // (2 * micro)} for up to {PROBE_SECONDS}s")
    proc = subprocess.Popen(cmd, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    deadline, peak = time.monotonic() + PROBE_SECONDS, 0
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(10)
        peak = max(peak, _gpu_peak_mib())
    crashed = proc.poll() is not None and proc.returncode != 0
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    time.sleep(10)  # let CUDA contexts release before the next candidate
    ok = not crashed and peak < PROBE_MEM_CAP_MIB
    log(f"micro-probe: micro {micro} -> peak {peak} MiB, "
        + ("crashed" if crashed else "survived") + f", {'ACCEPT' if ok else 'reject'}")
    return ok, peak


def run_micro_probe() -> None:
    marker = RUN_DIR / "micro_probe.json"
    if marker.exists():
        log(f"micro-probe already decided: {marker.read_text().strip()}")
        return
    results = {}
    chosen = 2  # measured-safe fallback (45.3 GiB at 8192; scales to ~fit)
    for micro in (4, 2):
        ok, peak = _probe_one(micro)
        results[micro] = {"ok": ok, "peak_mib": peak}
        if ok:
            chosen = micro
            break
    cfg = OmegaConf.load(RUN_DIR / "resolved_config.yaml")
    cfg.train.per_device_train_batch_size = chosen
    cfg.train.gradient_accumulation_steps = 32 // (2 * chosen)
    OmegaConf.save(config=cfg, f=RUN_DIR / "resolved_config.yaml", resolve=True)
    marker.write_text(json.dumps({"chosen_micro": chosen, "results": results}) + "\n")
    if PROBE_DIR.exists():
        shutil.rmtree(PROBE_DIR)
    log(f"micro-probe: locked micro {chosen} x accum {32 // (2 * chosen)} for all arms")


# --- drive ------------------------------------------------------------------

def drive() -> None:
    build()

    # Teacher block, once: 128 x 4 teacher rollouts + teacher 500x4 eval @16384.
    run_rollout_eval(TEACHER_RUN, "all", 0, eval_only=False)
    teacher_tokens = arm_round(TEACHER_RUN, "all", 0) / "rollouts" / "tokens"

    # Shared round-0 block: one pool, scored once, partitioned three ways.
    run_rollout_eval(RUN_DIR, BUCKETS[0], 0, eval_only=False)
    for arm in BUCKETS[1:]:
        _copy_shared_round0(BUCKETS[0], arm)
    r0_oracle = arm_round(RUN_DIR, BUCKETS[0], 0) / "oracle"
    run_score(
        arm_round(RUN_DIR, BUCKETS[0], 0) / "rollouts" / "tokens",
        r0_oracle, checkpoint_path(BUCKETS[0], 0), expected=128 * NUM_ROLLOUTS,
    )
    run_score(
        teacher_tokens, RUN_DIR / "teacher_scoring" / "round_00_base",
        checkpoint_path(BUCKETS[0], 0), expected=128 * TEACHER_ROLLOUTS,
    )
    for arm in BUCKETS:
        write_selection(arm, 0, r0_oracle)

    # Micro-batch probe on the real 16384 data BEFORE any real train step;
    # locks per_device/accum into resolved_config for every arm and round.
    run_micro_probe()

    # Iterative rounds, re-scored per arm from its current policy.
    for rnd in range(TRAIN_ROUNDS):
        for arm in BUCKETS:
            run_train(arm, rnd)
        for arm in BUCKETS:
            terminal = rnd + 1 == TRAIN_ROUNDS
            run_rollout_eval(RUN_DIR, arm, rnd + 1, eval_only=terminal)
            if not terminal:
                oracle_dir = arm_round(RUN_DIR, arm, rnd + 1) / "oracle"
                run_score(
                    arm_round(RUN_DIR, arm, rnd + 1) / "rollouts" / "tokens",
                    oracle_dir, checkpoint_path(arm, rnd + 1),
                    expected=128 * NUM_ROLLOUTS,
                )
                run_score(
                    teacher_tokens,
                    RUN_DIR / "teacher_scoring" / f"round_{rnd + 1:02d}_{arm}",
                    checkpoint_path(arm, rnd + 1), expected=128 * TEACHER_ROLLOUTS,
                )
                write_selection(arm, rnd + 1, oracle_dir)
    log("drive complete")


def report() -> None:
    subprocess.run(
        [sys.executable, "scripts/eval_table.py", "--run-dir", str(RUN_DIR)],
        check=True, cwd=ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--drive", action="store_true")
    group.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    elif args.drive:
        drive()
    else:
        report()


if __name__ == "__main__":
    main()
