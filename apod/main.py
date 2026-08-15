"""Hydra entry point for the APOD experiment (contract: docs/pipeline.md).

    uv run python -m apod.main +experiment=smoke   # smoke profile
    uv run python -m apod.main                     # real 3-arm run

Runs arms x rounds sequentially. Per (arm, round):
  1. apod.stages.rollout_eval  -- one subprocess per GPU (eval, then rollouts)
  2. merge eval shards         -> eval/summary.json
  3. apod.stages.entropy       -- only when the arm needs entropy scores
  4. selection in-process      -> selected/selected.jsonl
  5. apod.stages.train         -- single subprocess on cfg.train.train_gpu
  6. manifest.json + metrics.jsonl row
After the last training round an eval-only pass measures the final checkpoint
(round_{rounds:02d}), then apod.plotting renders the curves.

Stages stay separate subprocesses on purpose: each vLLM engine session must own
its GPU and be torn down before the HF stages, and CUDA_VISIBLE_DEVICES is
pinned per shard so stage code always sees exactly one device as cuda:0.

Resume protocol: a stage is skipped when its done.shard{K} markers (or its
output artifacts, for the in-process stages) already exist. metrics.jsonl is
idempotent -- one row per (arm, round), recomputed from on-disk artifacts and
rewritten wholesale, so re-running a finished round only refreshes its row.
Stage timings are unknowable for stages skipped on resume; those wall-clock
fields keep the values from the previously written metrics row (null if none).

Everything the driver and its children print is also written to
<run_dir>/driver.log, so a crashed overnight run keeps its full trace.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from apod.datasets.io import append_jsonl, read_jsonl, read_shards, write_jsonl
from apod.datasets.load import load_examples

REPO_ROOT = Path(__file__).resolve().parent.parent

WALL_CLOCK_KEYS = ("rollout_eval_s", "entropy_s", "train_s")


# ---------------------------------------------------------------------------
# subprocess plumbing
# ---------------------------------------------------------------------------


def _pump(proc: subprocess.Popen, tag: str) -> None:
    """Forward a child's merged stdout/stderr through the driver log."""
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info("[{}] {}", tag, line.rstrip())


def _map_gpu(logical: str) -> str:
    """Map logical GPU indices through any parent CUDA_VISIBLE_DEVICES.

    A driver launched under CUDA_VISIBLE_DEVICES=2,3 must hand its children
    physical GPUs 2 and 3 -- overwriting with the logical "0"/"1" would
    silently target someone else's cards on a shared box.
    """
    parent = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not parent:
        return logical
    visible = [d.strip() for d in parent.split(",") if d.strip()]
    try:
        return ",".join(visible[int(i)] for i in logical.split(","))
    except IndexError:
        raise RuntimeError(
            f"num_gpus needs logical GPU(s) {logical} but CUDA_VISIBLE_DEVICES="
            f"{parent!r} exposes only {len(visible)} device(s)"
        ) from None


def _run_subprocesses(specs: list[tuple[list[str], str, str]]) -> None:
    """Launch (cmd, gpu, tag) triples concurrently; fail loudly on any nonzero exit.

    Each child gets the driver's environment plus CUDA_VISIBLE_DEVICES pinned
    to its GPU, so stage modules always see exactly one device as cuda:0.
    """
    procs: list[tuple[subprocess.Popen, threading.Thread, list[str], str]] = []
    for cmd, gpu, tag in specs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = _map_gpu(gpu)
        logger.info("[{}] launch (CUDA_VISIBLE_DEVICES={}): {}", tag, gpu, " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        pump = threading.Thread(target=_pump, args=(proc, tag), daemon=True)
        pump.start()
        procs.append((proc, pump, cmd, tag))

    failures = []
    for proc, pump, cmd, tag in procs:
        code = proc.wait()
        pump.join()
        if code != 0:
            failures.append(f"[{tag}] exit {code}: {' '.join(cmd)}")
    if failures:
        raise RuntimeError("stage subprocess(es) failed:\n" + "\n".join(failures))


def _stage_cmd(module: str, run_dir: Path, arm: str, rnd: int, extra: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        f"apod.stages.{module}",
        "--run-dir",
        str(run_dir),
        "--arm",
        arm,
        "--round",
        str(rnd),
        *extra,
    ]


def _banner(arm: str, rnd: int, stage: str, status: str) -> None:
    logger.info("=" * 72)
    logger.info("== [{}] round {:02d} :: {} -- {}", arm, rnd, stage, status)
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
# resume checks
# ---------------------------------------------------------------------------


def _markers_present(stage_dir: Path, num_shards: int) -> bool:
    return all((stage_dir / f"done.shard{k}").exists() for k in range(num_shards))


def _eval_num_problems(cfg: DictConfig, rnd: int, *, eval_only: bool) -> int:
    full = int(cfg.eval.num_problems)
    if rnd == 0 or eval_only:
        return full
    return min(int(cfg.eval.intermediate_num_problems), full)


def _reuse_round0_eval(
    cfg: DictConfig, run_dir: Path, arm: str, round_dir: Path, num_shards: int
) -> None:
    """Round 0 evaluates the IDENTICAL base model in every arm, so the eval
    is computed once and copied (~30 min per duplicate arm at the real
    config). Rows AND markers are copied; the stage's resume path still
    validates the rows and rewrites the markers, so a torn copy cannot pass
    silently. rollouts are NOT shared: they are each arm's training data.
    """
    eval_dir = round_dir / "eval"
    if _markers_present(eval_dir, num_shards):
        return
    for other in cfg.arms:
        if str(other) == arm:
            continue
        src = run_dir / "arms" / str(other) / "rounds" / "round_00" / "eval"
        if not _markers_present(src, num_shards):
            continue
        shard_files = sorted(src.glob("eval.shard*.jsonl"))
        if not shard_files:
            continue
        eval_dir.mkdir(parents=True, exist_ok=True)
        for f in shard_files:
            shutil.copy2(f, eval_dir / f.name)
        for k in range(num_shards):
            shutil.copy2(src / f"done.shard{k}", eval_dir / f"done.shard{k}")
        with open(eval_dir / "reused_from.json", "w") as f:
            json.dump(
                {
                    "arm": str(other),
                    "reason": "round-0 base model is identical across arms",
                },
                f,
            )
        logger.info(
            "[{}] round 00 eval reused from arm '{}' (identical base model)",
            arm,
            other,
        )
        return


def _rollout_eval_done(round_dir: Path, num_shards: int, eval_only: bool) -> bool:
    if not _markers_present(round_dir / "eval", num_shards):
        return False
    return eval_only or _markers_present(round_dir / "rollouts", num_shards)


def _train_done(round_dir: Path) -> bool:
    return (round_dir / "checkpoint").is_dir() and (round_dir / "train" / "summary.json").exists()


# ---------------------------------------------------------------------------
# prompt pool
# ---------------------------------------------------------------------------


def _build_pool(cfg: DictConfig, run_dir: Path) -> None:
    """Create pool/prompts.jsonl, or extend it byte-stably when rounds grew.

    random.Random(seed).sample is not guaranteed to be prefix-stable across
    different n (CPython switches sampling algorithms with k), so an extension
    recomputes the full sample at the new n and verifies every existing row
    still matches before appending only the new tail. A mismatch means the two
    pools are incompatible and the run must use a fresh run_name.
    """
    pool_path = run_dir / "pool" / "prompts.jsonl"
    num_prompts = int(cfg.rollout.num_prompts)
    n = int(cfg.rounds) * num_prompts

    existing = read_jsonl(pool_path) if (cfg.resume and pool_path.exists()) else []
    if len(existing) >= n:
        logger.info("pool: reusing {} ({} rows >= {} needed)", pool_path, len(existing), n)
        return

    examples = load_examples(
        cfg.data.dataset, n=n, seed=int(cfg.data.pool_seed), split=cfg.data.split
    )
    rows = [
        {
            "example_index": i,
            "id": ex["id"],
            "prompt": ex["prompt"],
            "reference": ex["answer"],
            "round": i // num_prompts,
        }
        for i, ex in enumerate(examples)
    ]

    if not existing:
        write_jsonl(pool_path, rows)
        logger.info("pool: wrote {} rows to {}", len(rows), pool_path)
        return

    for i, old in enumerate(existing):
        if old != rows[i]:
            raise RuntimeError(
                f"pool extension mismatch at example_index {i}: the sample at "
                f"n={n} does not begin with the existing pool (sampling is not "
                f"prefix-stable across n). Start a fresh run_name instead of "
                f"extending rounds for this run."
            )
    for row in rows[len(existing) :]:
        append_jsonl(pool_path, row)
    logger.info(
        "pool: extended {} from {} to {} rows (existing rows untouched)",
        pool_path,
        len(existing),
        len(rows),
    )


def _materialize_eval_set(cfg: DictConfig, run_dir: Path) -> None:
    """Pin the eval problems to the run dir, like the prompt pool.

    The eval stage previously re-derived its problems from the live Hub
    dataset every round; an upstream dataset update (or cache re-download)
    mid-run would silently remap problem_index -> problem between rounds,
    making per-round eval curves compare different problem sets. One
    materialization at run start makes the mapping immutable.
    """
    path = run_dir / "pool" / "eval_problems.jsonl"
    if path.exists() and cfg.resume:
        logger.info("eval set: reusing {}", path)
        return
    problems = load_examples(
        cfg.eval.dataset, n=int(cfg.eval.num_problems), seed=int(cfg.seed)
    )
    write_jsonl(path, problems)
    logger.info("eval set: wrote {} problems to {}", len(problems), path)


# ---------------------------------------------------------------------------
# merges / metrics
# ---------------------------------------------------------------------------


def _merge_eval_summary(round_dir: Path, num_samples: int, num_problems: int) -> dict:
    rows = read_shards(round_dir / "eval", "eval.shard*.jsonl")
    if not rows:
        raise RuntimeError(f"no eval rows found under {round_dir / 'eval'}")
    expected = num_problems * num_samples
    if len(rows) != expected:
        # Markers alone are not proof: a lost/corrupted shard file with a
        # surviving marker would otherwise average over half the problems and
        # look completely normal downstream.
        raise RuntimeError(
            f"{round_dir / 'eval'} has {len(rows)} rows, expected "
            f"{num_problems} problems x {num_samples} samples = {expected}; "
            "an eval shard file is missing rows despite its done marker"
        )
    by_problem: dict[int, list[dict]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_index"], []).append(row)
    summary = {
        "avg_at_n": statistics.mean(float(r["correct"]) for r in rows),
        "pass_at_n": statistics.mean(
            float(any(r["correct"] for r in group)) for group in by_problem.values()
        ),
        "num_problems": len(by_problem),
        "num_samples": num_samples,
        # Strict grading requires a \boxed answer (the instructed protocol);
        # loose avg_at_n also credits a matching last expression mid-thinking.
        # Strict is the PRE-REGISTERED PRIMARY endpoint (report §11): a model
        # that learns to terminate converts loose-credits into strict-credits,
        # so loose is nearly blind to the main hypothesized effect. The
        # strict-vs-loose gap is the "knows it but won't stop" diagnostic.
        "strict_avg_at_n": statistics.mean(
            float(r["correct"] and r.get("has_boxed", True)) for r in rows
        ),
        "strict_pass_at_n": statistics.mean(
            float(any(r["correct"] and r.get("has_boxed", True) for r in group))
            for group in by_problem.values()
        ),
        "cap_hit_rate": statistics.mean(float(r["truncated"]) for r in rows),
        "mean_response_length": statistics.mean(float(r["response_length"]) for r in rows),
        # Fraction of samples with no extractable answer: separates "wrong"
        # from "unparseable" (rows written before this field count as parsed).
        "no_answer_rate": statistics.mean(
            1.0 - float(r.get("has_answer", True)) for r in rows
        ),
        # Problems whose GOLD answer failed to parse (scored 0 for every arm
        # and round -- a dataset defect, not a model failure).
        "gold_unparsed_problems": sum(
            1
            for group in by_problem.values()
            if not all(g.get("gold_parsed", True) for g in group)
        ),
    }
    with open(round_dir / "eval" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _run_selection(
    cfg: DictConfig, round_dir: Path, arm: str, *, needs_entropy: bool
) -> list[dict]:
    """Merge entropy into trajectories (when scored) and write selected.jsonl."""
    from apod.selection import select_trajectories

    trajectories = read_shards(round_dir / "rollouts", "trajectories.shard*.jsonl")
    if not trajectories:
        raise RuntimeError(f"no trajectory rows found under {round_dir / 'rollouts'}")

    if needs_entropy:
        entropy_rows = read_shards(round_dir / "entropy", "entropy.shard*.jsonl")
        by_key = {(r["example_index"], r["rollout_index"]): r for r in entropy_rows}
        missing = 0
        for traj in trajectories:
            scored = by_key.get((traj["example_index"], traj["rollout_index"]))
            if scored is None:
                missing += 1
            else:
                traj["entropy"] = scored["entropy"]
        if arm == "entropy_top4" and missing:
            raise RuntimeError(
                f"{missing}/{len(trajectories)} trajectories have no entropy "
                f"score under {round_dir / 'entropy'}; cannot select for "
                "entropy_top4. If the entropy stage logged 'empty response, "
                "skipping', those trajectories are unscorable and the round "
                "needs manual attention -- resume alone cannot clear this."
            )

    selected = select_trajectories(
        arm,
        trajectories,
        k=int(cfg.selection.k),
        num_rollouts=int(cfg.rollout.num_rollouts),
        seed=int(cfg.seed),
    )
    selected = sorted(selected, key=lambda r: (r["example_index"], r["rollout_index"]))
    write_jsonl(round_dir / "selected" / "selected.jsonl", selected)
    return selected


def _mean_or_none(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


class MetricsFile:
    """metrics.jsonl with idempotent upserts keyed by (arm, round)."""

    def __init__(self, path: Path, arm_order: list[str]):
        self.path = path
        self.arm_order = arm_order
        self.rows: dict[tuple[str, int], dict] = {
            (r["arm"], r["round"]): r for r in read_jsonl(path)
        }

    def upsert(self, row: dict) -> None:
        key = (row["arm"], row["round"])
        old = self.rows.get(key)
        if old is not None:
            # Timings are only measurable when the stage actually ran this
            # invocation; keep previously recorded values for skipped stages.
            old_wc = old.get("wall_clock") or {}
            for wc_key in WALL_CLOCK_KEYS:
                if row["wall_clock"].get(wc_key) is None and old_wc.get(wc_key) is not None:
                    row["wall_clock"][wc_key] = old_wc[wc_key]
            if row.get("rollout_throughput_tok_s") is None:
                row["rollout_throughput_tok_s"] = old.get("rollout_throughput_tok_s")
        self.rows[key] = row
        self._write()

    def _write(self) -> None:
        def sort_key(row: dict) -> tuple[int, int]:
            arm = row["arm"]
            arm_rank = self.arm_order.index(arm) if arm in self.arm_order else len(self.arm_order)
            return (arm_rank, row["round"])

        write_jsonl(self.path, sorted(self.rows.values(), key=sort_key))


# ---------------------------------------------------------------------------
# per-round orchestration
# ---------------------------------------------------------------------------


def _model_path(run_dir: Path, arm: str, rnd: int, cfg: DictConfig) -> str:
    """Manifest record of the round's starting model.

    Unconditional for rnd > 0: the round DID start from that checkpoint even
    if retention has since pruned its weights, and a resumed old round must
    neither crash here (the stages' strict resolve_model_path handles actual
    loads) nor misrecord the base model as its start.
    """
    if rnd > 0:
        return str(run_dir / "arms" / arm / "rounds" / f"round_{rnd - 1:02d}" / "checkpoint")
    return str(cfg.model.student_id)


def _run_round(
    cfg: DictConfig,
    run_dir: Path,
    metrics: MetricsFile,
    arm: str,
    rnd: int,
    cumulative_before: int,
    *,
    eval_only: bool,
) -> int:
    """Run (or resume) one (arm, round); returns the new cumulative count."""
    round_dir = run_dir / "arms" / arm / "rounds" / f"round_{rnd:02d}"
    num_shards = int(cfg.num_gpus)
    resume = bool(cfg.resume)
    wall_clock: dict[str, float | None] = {k: None for k in WALL_CLOCK_KEYS}

    # Round 0 and the terminal eval run the full set (the anchor and the
    # pre-registered primary endpoint); intermediate rounds run a prefix
    # subset — curve resolution is all they buy (~29 -> ~6 min at the real
    # config).
    n_eval = _eval_num_problems(cfg, rnd, eval_only=eval_only)

    # --- stage 1: eval + rollouts (sharded across GPUs) ---
    stage_name = "rollout_eval (eval-only)" if eval_only else "rollout_eval"
    if rnd == 0 and not eval_only:
        _reuse_round0_eval(cfg, run_dir, arm, round_dir, num_shards)
    if resume and _rollout_eval_done(round_dir, num_shards, eval_only):
        _banner(arm, rnd, stage_name, "SKIP (done markers present)")
    else:
        _banner(arm, rnd, stage_name, "start")
        extra = [
            "--num-shards", str(num_shards),
            "--eval-num-problems", str(n_eval),
        ] + (["--eval-only"] if eval_only else [])
        t0 = time.time()
        _run_subprocesses(
            [
                (
                    _stage_cmd("rollout_eval", run_dir, arm, rnd, ["--shard", str(k), *extra]),
                    str(k),
                    f"{arm}/r{rnd:02d}/rollout_eval/shard{k}",
                )
                for k in range(num_shards)
            ]
        )
        wall_clock["rollout_eval_s"] = time.time() - t0
        _banner(arm, rnd, stage_name, f"done in {wall_clock['rollout_eval_s']:.1f}s")

    eval_summary = _merge_eval_summary(round_dir, int(cfg.eval.num_samples), n_eval)
    eval_rows = read_shards(round_dir / "eval", "eval.shard*.jsonl")
    eval_tokens = sum(int(r["response_length"]) for r in eval_rows)

    if eval_only:
        row = {
            "arm": arm,
            "round": rnd,
            "trajectories_round": 0,
            "trajectories_cumulative": cumulative_before,
            "tokens_trained": None,
            "avg_at_n": eval_summary["avg_at_n"],
            "pass_at_n": eval_summary["pass_at_n"],
            "strict_avg_at_n": eval_summary["strict_avg_at_n"],
            "strict_pass_at_n": eval_summary["strict_pass_at_n"],
            "eval_num_problems": eval_summary["num_problems"],
            "eval_cap_hit_rate": eval_summary["cap_hit_rate"],
            "rollout_cap_hit_rate": None,
            "rollout_accuracy": None,
            "mean_entropy_selected": None,
            "train_loss_mean": None,
            "train_loss_final": None,
            "wall_clock": wall_clock,
            # None, not eval_tokens/wall: the field means ROLLOUT throughput
            # and an eval-only round has no rollouts to measure.
            "rollout_throughput_tok_s": None,
        }
        metrics.upsert(row)
        _write_manifest(cfg, round_dir, arm, rnd, run_dir, row["wall_clock"], row["rollout_throughput_tok_s"])
        return cumulative_before

    # --- stage 2: entropy scoring (only when the arm needs scores) ---
    # Single source of truth for "this arm needs entropy": _run_selection
    # receives the same flag, so the stage gate and the merge can never
    # disagree (skip-the-stage-but-demand-scores, or score-and-never-merge).
    needs_entropy = arm == "entropy_top4" or bool(cfg.selection.score_all_arms)
    if needs_entropy:
        if resume and _markers_present(round_dir / "entropy", num_shards):
            _banner(arm, rnd, "entropy", "SKIP (done markers present)")
        else:
            _banner(arm, rnd, "entropy", "start")
            t0 = time.time()
            _run_subprocesses(
                [
                    (
                        _stage_cmd(
                            "entropy",
                            run_dir,
                            arm,
                            rnd,
                            ["--shard", str(k), "--num-shards", str(num_shards)],
                        ),
                        str(k),
                        f"{arm}/r{rnd:02d}/entropy/shard{k}",
                    )
                    for k in range(num_shards)
                ]
            )
            wall_clock["entropy_s"] = time.time() - t0
            _banner(arm, rnd, "entropy", f"done in {wall_clock['entropy_s']:.1f}s")

    # --- stage 3: selection (in-process) ---
    selected_path = round_dir / "selected" / "selected.jsonl"
    if resume and selected_path.exists():
        _banner(arm, rnd, "selection", "SKIP (selected.jsonl present)")
        selected = read_jsonl(selected_path)
    else:
        _banner(arm, rnd, "selection", "start")
        selected = _run_selection(cfg, round_dir, arm, needs_entropy=needs_entropy)
        _banner(arm, rnd, "selection", f"done ({len(selected)} trajectories kept)")

    # --- stage 4: training (DDP across all GPUs via torchrun) ---
    if resume and _train_done(round_dir):
        _banner(arm, rnd, "train", "SKIP (checkpoint + summary present)")
    else:
        _banner(arm, rnd, "train", "start")
        t0 = time.time()
        if num_shards > 1:
            # Data-parallel: student + frozen teacher replicated per rank; the
            # only cross-GPU traffic is the student gradient all-reduce.
            # gradient_accumulation_steps in the config is already set for
            # this world size (effective batch = ranks x accum).
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={num_shards}",
                "-m",
                "apod.stages.train",
                "--run-dir", str(run_dir),
                "--arm", arm,
                "--round", str(rnd),
            ]
            gpus = ",".join(str(k) for k in range(num_shards))
        else:
            cmd = _stage_cmd("train", run_dir, arm, rnd, [])
            gpus = str(cfg.train.train_gpu)
        _run_subprocesses([(cmd, gpus, f"{arm}/r{rnd:02d}/train")])
        wall_clock["train_s"] = time.time() - t0
        _banner(arm, rnd, "train", f"done in {wall_clock['train_s']:.1f}s")

    with open(round_dir / "train" / "summary.json") as f:
        train_summary = json.load(f)

    # --- stage 5: bookkeeping ---
    trajectories = read_shards(round_dir / "rollouts", "trajectories.shard*.jsonl")
    rollout_tokens = sum(int(r["response_length"]) for r in trajectories)
    throughput = (
        (eval_tokens + rollout_tokens) / wall_clock["rollout_eval_s"]
        if wall_clock["rollout_eval_s"]
        else None
    )
    cumulative = cumulative_before + len(selected)
    row = {
        "arm": arm,
        "round": rnd,
        "trajectories_round": len(selected),
        "trajectories_cumulative": cumulative,
        "tokens_trained": train_summary["tokens_trained"],
        "avg_at_n": eval_summary["avg_at_n"],
        "pass_at_n": eval_summary["pass_at_n"],
        "strict_avg_at_n": eval_summary["strict_avg_at_n"],
        "strict_pass_at_n": eval_summary["strict_pass_at_n"],
        "eval_num_problems": eval_summary["num_problems"],
        "eval_cap_hit_rate": eval_summary["cap_hit_rate"],
        "rollout_cap_hit_rate": _mean_or_none([float(t["truncated"]) for t in trajectories]),
        "rollout_accuracy": _mean_or_none([float(t["correct"]) for t in trajectories]),
        "mean_entropy_selected": _mean_or_none([s.get("entropy") for s in selected]),
        "train_loss_mean": train_summary["train_loss_mean"],
        "train_loss_final": train_summary["train_loss_final"],
        "wall_clock": wall_clock,
        "rollout_throughput_tok_s": throughput,
    }
    metrics.upsert(row)
    # Manifest reuses the metrics row's wall_clock/throughput, which upsert()
    # has already backfilled from any previous invocation's row.
    stored = metrics.rows[(arm, rnd)]
    _write_manifest(
        cfg, round_dir, arm, rnd, run_dir, stored["wall_clock"], stored["rollout_throughput_tok_s"]
    )

    # Checkpoint retention, deliberately LAST: prune only once this round is
    # fully complete (checkpoint, metrics row, manifest), so a re-run of any
    # of its stages can still find the predecessor weights it resolves.
    # Weights older than the newest keep_checkpoints rounds are never read
    # again -- and keeping all 24 real-run checkpoints would need ~86 GB.
    # Config and tokenizer files stay; check_run knows pruned rounds have no
    # weights, and resolve_model_path names this setting when a re-run
    # reaches past the prune horizon.
    for old in range(rnd - int(cfg.keep_checkpoints) + 1):
        old_ckpt = run_dir / "arms" / arm / "rounds" / f"round_{old:02d}" / "checkpoint"
        for weights in old_ckpt.glob("*.safetensors"):
            weights.unlink()
            logger.info("pruned superseded weights: {}", weights)
    return cumulative


def _write_manifest(
    cfg: DictConfig,
    round_dir: Path,
    arm: str,
    rnd: int,
    run_dir: Path,
    wall_clock: dict,
    throughput: float | None,
) -> None:
    manifest = {
        "arm": arm,
        "round": rnd,
        "model_path": _model_path(run_dir, arm, rnd, cfg),
        "sampling": OmegaConf.to_container(cfg.sampling, resolve=True),
        "engine": OmegaConf.to_container(cfg.engine, resolve=True),
        "train": OmegaConf.to_container(cfg.train, resolve=True),
        "wall_clock": wall_clock,
        "rollout_throughput_tok_s": throughput,
    }
    round_dir.mkdir(parents=True, exist_ok=True)
    with open(round_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _check_fingerprint(cfg: DictConfig, run_dir: Path) -> None:
    """Refuse to resume a run dir under structurally different settings.

    Shard assignment (num_gpus), group completeness (num_rollouts,
    eval.num_samples), index->problem maps (seed, num_prompts, num_problems)
    are all re-derived from the CURRENT config on resume; changing any of
    them reinterprets on-disk artifacts silently (e.g. one surviving shard
    marker "completes" a stage, or 12-row groups count done at group_size 8).
    resolved_config.yaml is overwritten each start, so this is the one record
    that survives. rounds may legitimately grow (the pool extends).
    """
    fields = {
        "num_gpus": int(cfg.num_gpus),
        "seed": int(cfg.seed),
        "rollout.num_prompts": int(cfg.rollout.num_prompts),
        "rollout.num_rollouts": int(cfg.rollout.num_rollouts),
        "eval.num_problems": int(cfg.eval.num_problems),
        "eval.intermediate_num_problems": int(cfg.eval.intermediate_num_problems),
        "eval.num_samples": int(cfg.eval.num_samples),
        "sampling.max_new_tokens": int(cfg.sampling.max_new_tokens),
    }
    path = run_dir / "fingerprint.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        # A key the recorded fingerprint predates is backfilled, not a
        # mismatch: the old run cannot conflict with a knob that did not
        # exist, and the count-mismatch it guards against still fails loudly
        # in _merge_eval_summary. Keys that WERE recorded stay hard failures.
        missing = [k for k in fields if k not in recorded]
        diffs = {k: (recorded[k], v) for k, v in fields.items() if k in recorded and recorded[k] != v}
        if diffs:
            raise RuntimeError(
                f"run dir {run_dir} was created under different structural "
                f"settings; refusing to resume. Changed (recorded -> now): "
                f"{diffs}. Use a fresh run_name, or restore the recorded "
                "values, or delete the run dir for a clean start."
            )
        if missing:
            logger.warning(
                "fingerprint.json predates key(s) {}; backfilling with current values", missing
            )
            recorded.update({k: fields[k] for k in missing})
            path.write_text(json.dumps(recorded, indent=2))
    else:
        path.write_text(json.dumps(fields, indent=2))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.output_dir)
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Everything below -- driver lines and pumped child output alike -- also
    # lands in driver.log, so a run that dies overnight keeps its full trace.
    logger.add(run_dir / "driver.log", enqueue=True)

    # Stages read resolved_config.yaml (never conf/), so stamp the absolute
    # run dir and resolve every interpolation before dumping.
    cfg.output_dir = str(run_dir)
    _check_fingerprint(cfg, run_dir)
    OmegaConf.save(config=cfg, f=run_dir / "resolved_config.yaml", resolve=True)
    logger.info("run dir: {}", run_dir)

    _build_pool(cfg, run_dir)
    _materialize_eval_set(cfg, run_dir)

    arms = [str(a) for a in cfg.arms]
    # Validate BEFORE any GPU work: a typo'd arm would otherwise burn the full
    # rollout+eval stage (hours at real scale) before selection rejects it.
    from apod.selection import ARMS

    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown} in cfg.arms; expected among {ARMS}")
    rounds = int(cfg.rounds)
    metrics = MetricsFile(run_dir / "metrics.jsonl", arms)

    for arm in arms:
        cumulative = 0
        for rnd in range(rounds):
            cumulative = _run_round(
                cfg, run_dir, metrics, arm, rnd, cumulative, eval_only=False
            )
        # Final eval-only round: measure the checkpoint the last round produced.
        _run_round(cfg, run_dir, metrics, arm, rounds, cumulative, eval_only=True)

    logger.info("all arms complete; rendering plots")
    from apod.plotting import plot_results  # matplotlib Agg, no GPU: safe in-process

    out_path = plot_results(run_dir)
    logger.info("done: {}", out_path)


if __name__ == "__main__":
    main()
