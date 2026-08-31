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
  kl_high (ranks 1-4) / kl_mid (5-8) / kl_low (9-12), 512 trajectories each,
  plus a RANDOM control arm (random 4 of 12 per prompt, re-sampled each
  round; USER 2026-08-15: "there should be random control"). At round 0 all
  arms are the identical base policy, so selections partition/sample ONE
  shared scored pool; from round 1 each KL arm generates and re-scores its
  own. The random arm's later rounds are NOT scored (USER: "drop the scoring
  on random arm") -- it just rolls out, samples 4, trains.
- 2 training rounds per arm (0..1) + terminal eval (USER 2026-08-16:
  "make it 2 rounds please", reduced mid-run from 4);
  identical hyperparameters, micro-batch constant across all arms;
  eval 500x4 at 16384 every round; fresh 16384 base eval (the shared
  round-0 eval) is the baseline.
- No go/no-go gate on tertile separation (USER: "don't agree with gates just
  run the experiment") -- separation and per-bucket truncation/correctness/
  length ARE reported each round (bucket_stats.jsonl + logs), just not
  gated on. The round-0 separation is relayed to the user as soon as the
  shared pool is scored: THEY decide whether to pull the plug.

Usage:
  uv run python scripts/bucket_experiment.py --build   # run dirs + configs
  uv run python scripts/bucket_experiment.py --drive   # everything, resumable
  uv run python scripts/bucket_experiment.py --report
  uv run python scripts/bucket_experiment.py --replicate
      # USER 2026-08-17: different-seed replication of the round-1
      # inverted-U (mid > random > high > low). Fresh run dir at seed 1042:
      # new base pool 128x12 + base eval, scored, tertiled, the four
      # original arms trained ONE round each + terminal evals. No teacher
      # rollouts (reverse KL alone forms buckets), no micro re-probe, no
      # "all" arm.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from apod import paths

ROOT = Path(__file__).resolve().parent.parent
SOURCE_RUN = ROOT / "outputs/runs/apod"
RUN_DIR = ROOT / "outputs/runs/oracle16k"
TEACHER_RUN = ROOT / "outputs/runs/oracle16k_teacher"
BUCKETS = ("kl_high", "kl_mid", "kl_low")
ARMS = BUCKETS + ("random", "all")
# Controls: "random" = 4-of-12 per prompt, re-sampled each round;
# "all" = every rollout, no selection (1536 rows, 48 steps/round -- USER
# 2026-08-16: "add two rounds of training on all, not categorizing on
# divergence"). Neither is KL-scored after the shared round 0.
TRAIN_ROUNDS = 3  # trains at rounds 0..2; round 3 is the terminal eval-only.
                  # History: 4 -> 2 (USER 2026-08-16 "make it 2 rounds") ->
                  # 3 (USER 2026-08-17 "run round 3 as well", after every arm
                  # regressed in round 2: does it continue, plateau, or
                  # recover?). On resume, the banked round-2 eval-only rounds
                  # gain rollouts/scoring/selection/train; round 3 is the new
                  # terminal.
NUM_ROLLOUTS = 12
TEACHER_ROLLOUTS = 4


def log(msg: str) -> None:
    print(f"[bucket] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def arm_round(run_dir: Path, arm: str, rnd: int) -> Path:
    return paths.round_dir(run_dir, arm, rnd)


def checkpoint_path(arm: str, rnd: int) -> str:
    """Model that PRODUCES round rnd artifacts (mirrors resolve_model_path)."""
    if rnd == 0:
        return "Qwen/Qwen3.5-2B"
    return str(paths.checkpoint_dir(RUN_DIR, arm, rnd - 1))


# --- build ------------------------------------------------------------------

LR_OVERRIDE: float | None = None  # --lr: round-3 extension runs at the LR the
                                  # probe verdict picks (USER: "agreed on the
                                  # ordering" -- LR conditional on probe)


def _write_config(dst: Path, *, student_id: str, num_rollouts: int,
                  seed: int | None = None, cap: int = 16384) -> None:
    cfg = OmegaConf.load(SOURCE_RUN / "resolved_config.yaml")
    cfg.model.student_id = student_id
    cfg.rollout.num_rollouts = num_rollouts
    if seed is not None:
        cfg.seed = seed
        cfg.train.seed = seed  # resolved_config stores ${seed} resolved
    if LR_OVERRIDE is not None:
        cfg.train.learning_rate = LR_OVERRIDE
    cfg.sampling.max_new_tokens = cap
    if cap < 16384:
        # Truncated-regime reruns (USER 2026-08-18: caps 8192 and 4096 —
        # a 3-point cap sweep of the selection effect). Micro 4 is the
        # measured-safe setting at 8192 (59.2 GiB peak, apod run) and has
        # ample headroom at 4096.
        cfg.engine.max_model_len = cap * 2
        cfg.train.max_length = cap * 2
        cfg.train.per_device_train_batch_size = 4
        cfg.train.gradient_accumulation_steps = 4
    else:
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


def write_selection(arm: str, rnd: int, oracle_dir: Path | None) -> None:
    """oracle_dir=None: unscored pool (random arm rounds >= 1 -- USER
    2026-08-15: "drop the scoring on random arm"); select from trajectory
    metadata alone."""
    rdir = arm_round(RUN_DIR, arm, rnd)
    sel_path = rdir / "selected" / "selected.jsonl"
    if sel_path.exists():
        log(f"{arm} r{rnd} selection already written")
        return
    meta = {}
    for shard in sorted((rdir / "rollouts").glob("trajectories.shard*.jsonl")):
        for r in read_jsonl(shard):
            meta[(r["example_index"], r["rollout_index"])] = r
    by_example: dict[int, list[dict]] = defaultdict(list)
    if oracle_dir is not None:
        for shard in sorted(oracle_dir.glob("oracle_kl.shard*.jsonl")):
            for r in read_jsonl(shard):
                by_example[r["example_index"]].append(r)
    else:
        assert arm in ("random", "all"), f"{arm} requires a scored pool"
        for (example_index, _), r in sorted(meta.items()):
            by_example[example_index].append(r)
    rows = []
    for example_index in sorted(by_example):
        pool = by_example[example_index]
        assert len(pool) == NUM_ROLLOUTS, (
            f"example {example_index}: {len(pool)} rollouts, expected {NUM_ROLLOUTS}"
        )
        if arm == "all":
            picks = pool
        elif arm == "random":
            # Control: random 4 of 12, re-sampled each round. Deterministic
            # seed per (round, example) so resume replays the same draw.
            rng = random.Random(f"oracle16k-r{rnd}-e{example_index}")
            picks = [pool[i] for i in sorted(rng.sample(range(NUM_ROLLOUTS), 4))]
        else:
            ranked = sorted(pool, key=lambda r: r["mean_reverse_kl"], reverse=True)
            tertile = BUCKETS.index(arm)
            picks = ranked[tertile * 4 : (tertile + 1) * 4]
        for r in picks:
            key = (r["example_index"], r["rollout_index"])
            rows.append(
                {
                    "example_index": r["example_index"],
                    "rollout_index": r["rollout_index"],
                    "entropy": r.get("student_entropy"),
                    "mean_reverse_kl": r.get("mean_reverse_kl"),
                    "mean_forward_kl": r.get("mean_forward_kl"),
                    # USER POLICY 2026-08-18: no \boxed => incorrect (strict).
                    "correct": bool(meta[key]["correct"]) and bool(meta[key].get("has_boxed")),
                    "truncated": bool(meta[key]["truncated"]),
                    "response_length": meta[key]["response_length"],
                }
            )
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: bare existence of selected.jsonl is the resume done-marker, so
    # a kill mid-write must never leave a partial file behind.
    tmp = sel_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, sel_path)
    # Per-bucket per-round composition (USER: report, don't gate): tertile
    # separation, truncation %, correctness %, mean length. KL fields are
    # absent for the unscored random-arm rounds.
    stats = {
        "arm": arm,
        "round": rnd,
        "n": len(rows),
        "truncated_pct": sum(r["truncated"] for r in rows) / len(rows),
        "correct_pct": sum(r["correct"] for r in rows) / len(rows),
        "mean_length": sum(r["response_length"] for r in rows) / len(rows),
    }
    kl_line = "unscored"
    if oracle_dir is not None:
        kls = sorted(r["mean_reverse_kl"] for r in rows)
        stats.update(
            rkl_min=kls[0],
            rkl_median=kls[len(kls) // 2],
            rkl_mean=sum(kls) / len(kls),
            rkl_max=kls[-1],
            fkl_mean=sum(r["mean_forward_kl"] for r in rows) / len(rows),
        )
        kl_line = (
            f"rkl mean {stats['rkl_mean']:.4f} [{stats['rkl_min']:.4f}.."
            f"{stats['rkl_max']:.4f}] | fkl mean {stats['fkl_mean']:.4f}"
        )
    with (RUN_DIR / "bucket_stats.jsonl").open("a") as f:
        f.write(json.dumps(stats) + "\n")
    log(
        f"{arm} r{rnd}: {stats['n']} rows | {kl_line} "
        f"| trunc {stats['truncated_pct']:.1%} | correct {stats['correct_pct']:.1%} "
        f"| len {stats['mean_length']:.0f}"
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

REPLICATE = False  # variant modes (--replicate / --at8192): no teacher/probe
SEED_OVERRIDE: int | None = None
CAP = 16384


def drive() -> None:
    teacher_tokens = None
    if REPLICATE:
        _write_config(RUN_DIR, student_id="Qwen/Qwen3.5-2B",
                      num_rollouts=NUM_ROLLOUTS, seed=SEED_OVERRIDE, cap=CAP)
        log(f"variant config written: {RUN_DIR} (seed {SEED_OVERRIDE or 'base'}, cap {CAP})")
    else:
        build()
        # Teacher block, once: 128x4 teacher rollouts + teacher 500x4 eval.
        run_rollout_eval(TEACHER_RUN, "all", 0, eval_only=False)
        teacher_tokens = arm_round(TEACHER_RUN, "all", 0) / "rollouts" / "tokens"

    # Shared round-0 block: one pool, scored once, partitioned three ways.
    run_rollout_eval(RUN_DIR, ARMS[0], 0, eval_only=False)
    for arm in ARMS[1:]:
        _copy_shared_round0(ARMS[0], arm)
    r0_oracle = arm_round(RUN_DIR, BUCKETS[0], 0) / "oracle"
    run_score(
        arm_round(RUN_DIR, BUCKETS[0], 0) / "rollouts" / "tokens",
        r0_oracle, checkpoint_path(BUCKETS[0], 0), expected=128 * NUM_ROLLOUTS,
    )
    if not REPLICATE:
        run_score(
            teacher_tokens, RUN_DIR / "teacher_scoring" / "round_00_base",
            checkpoint_path(BUCKETS[0], 0), expected=128 * TEACHER_ROLLOUTS,
        )
    for arm in ARMS:
        write_selection(arm, 0, r0_oracle)

    # Micro-batch probe on the real 16384 data BEFORE any real train step;
    # locks per_device/accum into resolved_config for every arm and round.
    # (Replication inherits the main run's measured micro 2 via the config.)
    if not REPLICATE:
        run_micro_probe()

    # Iterative rounds, re-scored per arm from its current policy.
    for rnd in range(TRAIN_ROUNDS):
        for arm in ARMS:
            run_train(arm, rnd)
        for arm in ARMS:
            terminal = rnd + 1 == TRAIN_ROUNDS
            run_rollout_eval(RUN_DIR, arm, rnd + 1, eval_only=terminal)
            if not terminal:
                oracle_dir = None
                if arm in BUCKETS:  # USER: no scoring passes on the controls
                    oracle_dir = arm_round(RUN_DIR, arm, rnd + 1) / "oracle"
                    run_score(
                        arm_round(RUN_DIR, arm, rnd + 1) / "rollouts" / "tokens",
                        oracle_dir, checkpoint_path(arm, rnd + 1),
                        expected=128 * NUM_ROLLOUTS,
                    )
                    if not REPLICATE:
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
    global REPLICATE, RUN_DIR, TRAIN_ROUNDS, ARMS, SEED_OVERRIDE, CAP
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--drive", action="store_true")
    group.add_argument("--report", action="store_true")
    group.add_argument("--replicate", action="store_true",
                       help="seed-1042 one-round rerun of the 4 bucket arms")
    group.add_argument("--at8192", action="store_true",
                       help="truncated-regime rerun: 4 bucket arms at cap 8192, "
                            "1 round + terminal (USER 2026-08-18)")
    group.add_argument("--at4096", action="store_true",
                       help="third cap point: 4 bucket arms at cap 4096, "
                            "1 round + terminal (USER 2026-08-18)")
    parser.add_argument("--lr", type=float, default=None,
                        help="override train LR for this (re)launch, e.g. 5e-6 "
                             "for the round-3 extension per the probe verdict")
    args = parser.parse_args()
    if args.lr is not None:
        global LR_OVERRIDE
        LR_OVERRIDE = args.lr
    if args.replicate:
        REPLICATE = True
        RUN_DIR = ROOT / "outputs/runs/oracle16k_seed2"
        SEED_OVERRIDE = 1042
        TRAIN_ROUNDS = 3  # 1 -> 2 -> 3 (USER 2026-08-19 "do three rounds",
                          # after the main run's round-3 oscillation: the
                          # dip-then-recover pattern needs the third trained
                          # round at the second seed too)
        ARMS = BUCKETS + ("random",)
        drive()
    elif args.at8192 or args.at4096:
        REPLICATE = True  # same skips: no teacher set, no micro re-probe
        CAP = 8192 if args.at8192 else 4096
        RUN_DIR = ROOT / ("outputs/runs/oracle8k" if args.at8192 else "outputs/runs/oracle4k")
        TRAIN_ROUNDS = 3  # 1 -> 2 -> 3 (USER 2026-08-19 "do three rounds"):
                          # the full peak / dip / recovery trajectory at
                          # every cap point, matching the main run.
        ARMS = BUCKETS + ("random",)
        drive()
    elif args.build:
        build()
    elif args.drive:
        drive()
    else:
        report()


if __name__ == "__main__":
    main()
