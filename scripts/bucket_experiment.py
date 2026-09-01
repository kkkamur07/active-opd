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
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from apod import paths

ROOT = Path(__file__).resolve().parent.parent
SOURCE_RUN = ROOT / "outputs/runs/apod"  # prompt pool + MATH-500 eval set only
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
# --eval-num-problems: evaluate only the first N materialized problems in the
# INTERMEDIATE rounds (0 < round < terminal), the apod.main protocol
# (conf/eval intermediate_num_problems). None = the full set every round,
# which is what kl50/kl50w ran. Round 0 (the anchor) and the terminal
# eval-only round always use the full set regardless.
EVAL_NUM_PROBLEMS: int | None = None


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


def _compose_config():
    """conf/ composed the way ``apod.main``'s Hydra app sees it (defaults
    list + interpolations), before any per-run override below."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "conf")):
        return compose(config_name="config")


def _write_config(dst: Path, *, student_id: str, num_rollouts: int,
                  seed: int | None = None, cap: int = 16384,
                  num_prompts: int | None = None,
                  gpu_mem: float | None = None) -> None:
    """Fresh run dir: compose conf/ and stamp this run's overrides.

    An existing run dir keeps the resolved_config.yaml it started with:
    every stage and every resume reads THAT file, and conf/ may have moved
    on since (engine.target_concurrent_sequences is a between-runs setting,
    like a sampling knob). ``--lr`` is the one deliberate relaunch override.
    """
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "pool").exists():
        shutil.copytree(SOURCE_RUN / "pool", dst / "pool")
    path = dst / "resolved_config.yaml"
    if path.exists():
        if LR_OVERRIDE is not None:
            cfg = OmegaConf.load(path)
            cfg.train.learning_rate = LR_OVERRIDE
            OmegaConf.save(config=cfg, f=path, resolve=True)
        return
    cfg = _compose_config()
    cfg.run_name = dst.name
    cfg.output_dir = str(dst)  # absolute, as apod.main stamps it
    cfg.model.student_id = student_id
    cfg.rollout.num_rollouts = num_rollouts
    if num_prompts is not None:
        cfg.rollout.num_prompts = num_prompts
    if gpu_mem is not None:
        cfg.engine.gpu_memory_utilization = gpu_mem
    if seed is not None:
        # The pool is COPIED from SOURCE_RUN, so its seed stays the composed
        # default; only the sampling seed and train.seed (${seed}) move.
        cfg.data.pool_seed = int(cfg.data.pool_seed)
        cfg.seed = seed
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
    OmegaConf.save(config=cfg, f=path, resolve=True)


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
        if EVAL_NUM_PROBLEMS is not None and rnd > 0 and not eval_only:
            cmd += ["--eval-num-problems", str(EVAL_NUM_PROBLEMS)]
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": str(shard)}
        log("launch: " + " ".join(cmd[3:]))
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"{arm} r{rnd} rollout_eval shards exited {codes}")


def _score_count(out_dir: Path) -> int:
    # Distinct trajectories, not raw lines: a resume re-scores rows that
    # predate the Eq. 6-7 fields and APPENDS, so line count can reach
    # `expected` while trajectories are still missing.
    keys = set()
    for p in out_dir.glob("oracle_kl.shard*.jsonl"):
        keys.update((r["example_index"], r["rollout_index"]) for r in read_jsonl(p))
    return len(keys)


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
    if WARMUP_OPT:
        # Run-wide schedule: this pass resumes the LR curve at its global step.
        cmd += ["--global-step-offset", str(rnd * KL50_STEPS_PER_ROUND)]
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
        # Keep the LAST row per trajectory: a resume after the Eq. 6-7
        # fields were added re-scores pre-existing rows and APPENDS, leaving
        # both versions in the shard files (same convention as
        # oracle_kl --analyze; the kl50 banked round 0 hits this).
        dedup: dict[tuple[int, int], dict] = {}
        for shard in sorted(oracle_dir.glob("oracle_kl.shard*.jsonl")):
            for r in read_jsonl(shard):
                dedup[(r["example_index"], r["rollout_index"])] = r
        for r in dedup.values():
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
    """Round-0 pool is shared: same base policy, ONE scored pool partitioned.

    Unconditional recopy on every (re)launch: it is a few MB (tokens are
    symlinked, not copied) and it heals a dst copied before the src gained
    later rows -- e.g. the kl50 top-up to 136 prompts after banking 128."""
    src, dst = arm_round(RUN_DIR, src_arm, 0), arm_round(RUN_DIR, dst_arm, 0)
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

PROBE_SECONDS = 900   # >= a few optimizer steps at 16k-token sequences
PROBE_MEM_CAP_MIB = 75_000  # accept only if peak < ~73 GiB of 81.9k MiB


def _probe_dir() -> Path:
    return RUN_DIR.parent / (RUN_DIR.name + "_msmoke")


def _gpu_peak_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    return max(int(x) for x in out.split())


def _probe_one(micro: int) -> tuple[bool, int]:
    """Trial-train round-0 kl_high at this micro-batch; return (ok, peak MiB)."""
    probe_dir = _probe_dir()
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    cfg = OmegaConf.load(RUN_DIR / "resolved_config.yaml")
    cfg.train.per_device_train_batch_size = micro
    cfg.train.gradient_accumulation_steps = 32 // (2 * micro)
    probe_dir.mkdir(parents=True)
    OmegaConf.save(config=cfg, f=probe_dir / "resolved_config.yaml", resolve=True)
    (probe_dir / "pool").symlink_to(RUN_DIR / "pool")
    src = arm_round(RUN_DIR, BUCKETS[0], 0)
    rdir = arm_round(probe_dir, BUCKETS[0], 0)
    rdir.mkdir(parents=True)
    (rdir / "rollouts").symlink_to(src / "rollouts")
    (rdir / "selected").symlink_to(src / "selected")
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", "-m", "apod.stages.train",
        "--run-dir", str(probe_dir), "--arm", BUCKETS[0], "--round", "0",
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


def _apply_micro(chosen: int) -> None:
    cfg = OmegaConf.load(RUN_DIR / "resolved_config.yaml")
    cfg.train.per_device_train_batch_size = chosen
    cfg.train.gradient_accumulation_steps = 32 // (2 * chosen)
    OmegaConf.save(config=cfg, f=RUN_DIR / "resolved_config.yaml", resolve=True)


def run_micro_probe(candidates: tuple[int, ...] = (4, 2), fallback: int = 2) -> None:
    marker = RUN_DIR / "micro_probe.json"
    if marker.exists():
        # Re-apply on resume (a no-op now that _write_config leaves an
        # existing run's config alone; kept so a hand-edited config cannot
        # silently drop the probe's verdict).
        chosen = json.loads(marker.read_text())["chosen_micro"]
        _apply_micro(chosen)
        log(f"micro-probe already decided: micro {chosen} (re-applied)")
        return
    results = {}
    chosen = fallback
    for micro in candidates:
        ok, peak = _probe_one(micro)
        results[micro] = {"ok": ok, "peak_mib": peak}
        if ok:
            chosen = micro
            break
    _apply_micro(chosen)
    marker.write_text(json.dumps({"chosen_micro": chosen, "results": results}) + "\n")
    if _probe_dir().exists():
        shutil.rmtree(_probe_dir())
    log(f"micro-probe: locked micro {chosen} x accum {32 // (2 * chosen)} for all arms")


# --- peak-LR probe (kl50w) --------------------------------------------------
# OPD/GKD recipes for Qwen-scale students cluster at 5e-6..5e-5 with cosine
# decay + short warmup (Thinking Machines OPD blog; GKD paper appendix).
# The incumbent 2e-5 (all prior runs) is the ceiling candidate; probe each
# candidate for ~PROBE_SECONDS of real steps on the round-0 kl_high selection
# and pick by training-loss progress, preferring the lower LR on a near-tie
# (three rounds of persisted Adam state compound any instability).


def _apply_lr(chosen: float) -> None:
    cfg = OmegaConf.load(RUN_DIR / "resolved_config.yaml")
    cfg.train.learning_rate = chosen
    OmegaConf.save(config=cfg, f=RUN_DIR / "resolved_config.yaml", resolve=True)


def _lr_probe_one(lr: float) -> dict:
    """Trial-train round-0 kl_high at this peak LR under the run's actual
    schedule (warmup 2 + cosine); parse per-step loss/grad_norm from the log."""
    probe_dir = _probe_dir()
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    cfg = OmegaConf.load(RUN_DIR / "resolved_config.yaml")
    cfg.train.learning_rate = lr
    cfg.train.persist_optimizer = False  # throwaway: no state to load or keep
    probe_dir.mkdir(parents=True)
    OmegaConf.save(config=cfg, f=probe_dir / "resolved_config.yaml", resolve=True)
    (probe_dir / "pool").symlink_to(RUN_DIR / "pool")
    src = arm_round(RUN_DIR, BUCKETS[0], 0)
    rdir = arm_round(probe_dir, BUCKETS[0], 0)
    rdir.mkdir(parents=True)
    (rdir / "rollouts").symlink_to(src / "rollouts")
    (rdir / "selected").symlink_to(src / "selected")
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", "-m", "apod.stages.train",
        "--run-dir", str(probe_dir), "--arm", BUCKETS[0], "--round", "0",
    ]
    log(f"lr-probe: lr {lr:g} for up to {PROBE_SECONDS}s")
    plog = RUN_DIR / f"lr_probe_{lr:g}.log"
    with plog.open("w") as f:
        proc = subprocess.Popen(cmd, env={**os.environ, "HF_HUB_OFFLINE": "1"},
                                stdout=f, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + PROBE_SECONDS
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(10)
        crashed = proc.poll() is not None and proc.returncode != 0
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    time.sleep(10)  # let CUDA contexts release before the next candidate
    text = plog.read_text()
    # the trainer prints values as quoted strings: {'loss': '0.06174', ...}
    losses = [float(x) for x in re.findall(r"'loss': '?([0-9.eE+-]+)", text)]
    grads = [float(x) for x in re.findall(r"'grad_norm': '?([0-9.eE+-]+)", text)]
    post_warmup_peak = max(grads[2:], default=0.0)
    ok = not crashed and len(losses) >= 5 and post_warmup_peak < 5.0
    log(f"lr-probe: lr {lr:g} -> {len(losses)} steps, last-3 losses "
        f"{[round(x, 4) for x in losses[-3:]]}, post-warmup grad peak "
        f"{post_warmup_peak:.2f}, " + ("crashed, " if crashed else "")
        + ("ACCEPT" if ok else "reject"))
    return {"lr": lr, "ok": ok, "losses": losses, "grad_norms": grads}


def run_lr_probe(candidates: tuple[float, ...] = (2e-5, 1e-5, 5e-6),
                 fallback: float = 2e-5) -> None:
    marker = RUN_DIR / "lr_probe.json"
    if marker.exists():
        # Re-apply on resume, same reason as run_micro_probe.
        chosen = json.loads(marker.read_text())["chosen_lr"]
        _apply_lr(chosen)
        log(f"lr-probe already decided: lr {chosen:g} (re-applied)")
        return
    results = [_lr_probe_one(lr) for lr in candidates]
    usable = [r for r in results if r["ok"]]
    if usable:
        def score(r: dict) -> float:
            return sum(r["losses"][-3:]) / len(r["losses"][-3:])
        best = score(min(usable, key=score))
        # Near-tie (within 2% relative): the lower LR is the safer choice.
        chosen = min(r["lr"] for r in usable if score(r) <= best * 1.02)
    else:
        chosen = fallback
    _apply_lr(chosen)
    marker.write_text(json.dumps({"chosen_lr": chosen, "results": results}) + "\n")
    if _probe_dir().exists():
        shutil.rmtree(_probe_dir())
    log(f"lr-probe: locked peak lr {chosen:g} for all arms/rounds")


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


def prune_checkpoints(arm: str, rnd: int, keep: int = 2) -> None:
    """Drop weight files of rounds older than the newest `keep` (resume and
    the next round only need the latest; configs/manifests stay)."""
    for old in range(rnd - keep + 1):
        ckpt = paths.checkpoint_dir(RUN_DIR, arm, old)
        for weights in ckpt.glob("*.safetensors"):
            weights.unlink()
            log(f"pruned {weights}")


# --- kl50: kl_high / kl_mid / kl_low vs random at cap 8192, 51 steps --------
# (USER 2026-08-31 late: "total student rollouts being 12 (4 each for kl high
# to low) and random as well ... 50 steps (so expand the sampling of the
# prompts)". 136 prompts x 4 selected = 544 rows = 17 steps/round at eff
# batch 32 -> 51 total; 128 prompts gave only 48. Micro-batch probe 8 then 4;
# vLLM at 0.95 (USER asked 0.99 "if safe": 0.99 leaves <1 GiB for the CUDA
# context + cudagraphs on an 80 GiB card, so 0.95 is the max-safe setting).)

KL50_DIR = ROOT / "outputs/runs/kl50"
KL50W_DIR = ROOT / "outputs/runs/kl50w"
KL50_TEACHER = ROOT / "outputs/runs/kl50_teacher"
ORACLE8K = ROOT / "outputs/runs/oracle8k"
KL50_PROMPTS = 136
KL50_STEPS_PER_ROUND = KL50_PROMPTS * 4 // 32  # 544 selected rows / eff batch 32 = 17
# kl50w (USER 2026-09-01: "restart the run with warmup again - with the
# optimizer state stores as well ... and also we need to hyper parameter
# tune for lr (with cosine decay or something)"): same arms/data/steps as
# kl50, but one run-wide warmup+cosine LR schedule, Adam state persisted
# across rounds (name-mapped, see train.py), and the peak LR picked by a
# probe + per-arm sweep + refine (3.16e-6).
WARMUP_OPT = False   # set by --kl50w
BANK_SRC = ORACLE8K  # kl50 banks from oracle8k; kl50w banks from kl50


def _rebucket_pool_rounds() -> None:
    """Re-bucket the existing prompt pool at KL50_PROMPTS per round.

    The pool rows (a fixed seed-42 sample, 1024 prompts) keep their order and
    example_index; only the `round` field is reassigned as i // 136. The
    banked oracle8k round 0 covered examples 0..127, which stay in round 0;
    examples 128..135 (previously round 1) join round 0 and are generated on
    resume. Rounds 1-2 draw entirely from prompts no kl50 stage has touched.
    Re-sampling the pool at a larger n instead would invalidate the bank:
    load_examples is not prefix-stable across n.
    """
    pool_path = RUN_DIR / "pool" / "prompts.jsonl"
    rows = read_jsonl(pool_path)
    changed = False
    for i, row in enumerate(rows):
        assert row["example_index"] == i, f"pool row {i} has index {row['example_index']}"
        if row["round"] != i // KL50_PROMPTS:
            row["round"] = i // KL50_PROMPTS
            changed = True
    if not changed:
        return
    tmp = pool_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, pool_path)
    log(f"pool re-bucketed: {len(rows)} prompts at {KL50_PROMPTS}/round")


def _bank_round0(src_run: Path) -> None:
    """The source run's shared round-0 block is valid here verbatim: same
    base model, pool, seed 42 and cap-8192 config (all written by
    _write_config), with its eval/rollouts/oracle scoring complete.
    Selections are NOT copied -- write_selection redoes them.

    The done.shard markers are DROPPED from the copy so run_rollout_eval
    still launches the stage and row-level resume decides what is pending
    (kl50 banking oracle8k's 128-prompt round 0 needed a 136-prompt top-up;
    copied markers made the driver skip the stage entirely, which is exactly
    the bug that stalled the first relaunch). The bank itself is guarded by
    its own marker file, written last, so a re-launch never re-copies over
    post-bank work (top-up rows, oracle re-scoring)."""
    src = arm_round(src_run, BUCKETS[0], 0)
    dst = arm_round(RUN_DIR, BUCKETS[0], 0)
    markers = (dst / "banked_round0.json", dst / "banked_from_oracle8k.json")
    if any(m.exists() for m in markers):
        return
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("eval", "rollouts", "oracle"):
        if (dst / sub).exists():
            shutil.rmtree(dst / sub)
        shutil.copytree(src / sub, dst / sub)
    for done in [*(dst / "eval").glob("done.shard*"), *(dst / "rollouts").glob("done.shard*")]:
        done.unlink()
    markers[0].write_text(json.dumps({"src": str(src)}) + "\n")
    log(f"round-0 banked from {src_run.name} (eval + rollouts + oracle scoring)")


def drive_kl50() -> None:
    _write_config(RUN_DIR, student_id="Qwen/Qwen3.5-2B", num_rollouts=NUM_ROLLOUTS,
                  cap=CAP, num_prompts=KL50_PROMPTS, gpu_mem=0.95)
    _write_config(KL50_TEACHER, student_id="Qwen/Qwen3.5-9B",
                  num_rollouts=TEACHER_ROLLOUTS, cap=CAP, gpu_mem=0.95)
    if WARMUP_OPT:
        # ONE LR schedule over the whole run (USER 2026-09-01 ~20:45: "it has
        # to continue, fix that" -- a per-round warmup+cosine restarted the LR
        # at [0, peak] every round while the Adam moments continued): warmup
        # 3 steps at the start only (5% of 51, rounded up; USER: "5% warmup
        # ratio is fair"), then cosine to 10% of peak over the 51 steps. Each
        # round's train pass advances the scheduler by --global-step-offset
        # (run_train) so its 17 steps land at the right point of the curve.
        # persist_optimizer makes each round a continuation of the last
        # (train.py name-maps the saved Adam state onto the new param order).
        cfg = OmegaConf.load(RUN_DIR / "resolved_config.yaml")
        cfg.train.warmup_steps = 3
        cfg.train.total_training_steps = TRAIN_ROUNDS * KL50_STEPS_PER_ROUND
        cfg.train.lr_scheduler_type = "cosine_with_min_lr"
        cfg.train.lr_scheduler_kwargs = {"min_lr_rate": 0.1}
        cfg.train.persist_optimizer = True
        OmegaConf.save(config=cfg, f=RUN_DIR / "resolved_config.yaml", resolve=True)
        log(f"kl50w: warmup {cfg.train.warmup_steps} + "
            f"cosine_with_min_lr(min_lr_rate=0.1) over {cfg.train.total_training_steps} "
            "steps run-wide, Adam state persisted across rounds")
    # Warmup history: a warmup_steps=2 + constant_with_warmup patch ran here
    # for the round-2 trains only (it did kill the Adam-reset restart spike:
    # kl_high r2 opened 0.066/0.065 vs r1's 0.067->0.175, grad_norm 0.53 vs
    # 6.12), but USER 2026-09-01 ~11:25 UTC ordered round 2 RETRAINED without
    # it so all rounds share one optimizer regime ("kill the runs and train
    # out the warmup"). The warmup-trained round_02 checkpoints/train dirs
    # and round_03 evals are preserved in outputs/runs/kl50_warmup_backup/.
    _rebucket_pool_rounds()
    # Teacher ceiling at THIS cap (500x4, same protocol) -- eval only, no
    # teacher rollouts and no teacher-pool rescoring (those were diagnostics).
    run_rollout_eval(KL50_TEACHER, "all", 0, eval_only=True)

    _bank_round0(BANK_SRC)
    run_rollout_eval(RUN_DIR, ARMS[0], 0, eval_only=False)  # no-op when banked
    for arm in ARMS[1:]:
        _copy_shared_round0(ARMS[0], arm)
    r0_oracle = arm_round(RUN_DIR, BUCKETS[0], 0) / "oracle"
    run_score(
        arm_round(RUN_DIR, BUCKETS[0], 0) / "rollouts" / "tokens",
        r0_oracle, checkpoint_path(BUCKETS[0], 0),
        expected=KL50_PROMPTS * NUM_ROLLOUTS,
    )
    for arm in ARMS:
        write_selection(arm, 0, r0_oracle)

    # Probe micro 8 first (USER: "i think it can accomodate 8 easily now");
    # micro 4 is the measured-safe 8192 fallback (59.2 GiB peak, apod run).
    # kl50w inherits kl50's verdict (identical model/cap/data): copy the
    # marker so run_micro_probe just re-applies instead of re-probing.
    if RUN_DIR != KL50_DIR and not (RUN_DIR / "micro_probe.json").exists():
        shutil.copy(KL50_DIR / "micro_probe.json", RUN_DIR / "micro_probe.json")
    run_micro_probe(candidates=(8, 4), fallback=4)
    if WARMUP_OPT:
        run_lr_probe()

    for rnd in range(TRAIN_ROUNDS):
        for arm in ARMS:
            run_train(arm, rnd)
            prune_checkpoints(arm, rnd)
            if WARMUP_OPT and rnd >= 1:
                # The previous round's Adam state was consumed by this train;
                # drop it (bf16 moments ~8 GB per arm) so steady state keeps
                # exactly one optimizer_state.pt per arm on disk.
                stale = paths.checkpoint_dir(RUN_DIR, arm, rnd - 1) / "optimizer_state.pt"
                if stale.exists():
                    stale.unlink()
                    log(f"pruned {stale}")
        for arm in ARMS:
            terminal = rnd + 1 == TRAIN_ROUNDS
            run_rollout_eval(RUN_DIR, arm, rnd + 1, eval_only=terminal)
            if not terminal:
                oracle_dir = None
                if arm in BUCKETS:  # random stays unscored after round 0
                    oracle_dir = arm_round(RUN_DIR, arm, rnd + 1) / "oracle"
                    run_score(
                        arm_round(RUN_DIR, arm, rnd + 1) / "rollouts" / "tokens",
                        oracle_dir, checkpoint_path(arm, rnd + 1),
                        expected=KL50_PROMPTS * NUM_ROLLOUTS,
                    )
                write_selection(arm, rnd + 1, oracle_dir)
    log("kl50 drive complete")


def report() -> None:
    subprocess.run(
        [sys.executable, "scripts/eval_table.py", "--run-dir", str(RUN_DIR)],
        check=True, cwd=ROOT,
    )


def main() -> None:
    global REPLICATE, RUN_DIR, TRAIN_ROUNDS, ARMS, SEED_OVERRIDE, CAP, WARMUP_OPT, BANK_SRC
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
    group.add_argument("--kl50", action="store_true",
                       help="kl_high/kl_mid/kl_low vs random at cap 8192, "
                            "3 rounds x 17 steps = 51 at eff batch 32, 136 "
                            "prompts/round, teacher ceiling eval at 8192, "
                            "micro probed 8 then 4 (USER 2026-08-31)")
    group.add_argument("--kl50w", action="store_true",
                       help="kl50 rerun with per-round warmup+cosine LR "
                            "cycles, Adam state persisted across rounds, and "
                            "a probed peak LR; banks round 0 from kl50 "
                            "(USER 2026-09-01)")
    parser.add_argument("--lr", type=float, default=None,
                        help="override train LR for this (re)launch, e.g. 5e-6 "
                             "for the round-3 extension per the probe verdict")
    parser.add_argument("--eval-num-problems", type=int, default=None,
                        help="intermediate rounds evaluate only the first N "
                             "problems (round 0 and the terminal round always "
                             "run the full set); default: full set every round")
    args = parser.parse_args()
    if args.lr is not None:
        global LR_OVERRIDE
        LR_OVERRIDE = args.lr
    if args.eval_num_problems is not None:
        global EVAL_NUM_PROBLEMS
        EVAL_NUM_PROBLEMS = args.eval_num_problems
    if args.kl50 or args.kl50w:
        RUN_DIR = KL50W_DIR if args.kl50w else KL50_DIR
        CAP = 8192
        TRAIN_ROUNDS = 3
        ARMS = ("kl_high", "kl_mid", "kl_low", "random")
        if args.kl50w:
            WARMUP_OPT = True
            BANK_SRC = KL50_DIR
        drive_kl50()
    elif args.replicate:
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
