"""Step-based experiment driver: arms x refreshes, budget in training steps.

    python -m apod.driver +experiment=r1_correctness_8k
    python -m apod.driver +experiment=r2_trajsel_8k driver.dry_run=true output_dir=/tmp/r2

Hydra app over conf/ (schema: conf/driver.yaml; ADR 0005, 0006). One
training step = one optimizer update at train.effective_batch trajectories.
Per arm (sequential, in cfg.driver.arms order), refresh r = 0..refreshes-1
at step = r * refresh_every:

  1. eval the current weights on cfg.eval (MATH-500, avg@4 / pass@4, all
     500) and on every driver.monitor_sets entry (AIME 2025+2026, avg@16),
     at the run cap, strict (no \\boxed = wrong, cap-hit included)
  2. roll out the refresh's block of questions from those same weights
     (one rollout_eval engine session does 1 and 2: MATH-500 requests, the
     monitor sets', then the rollouts -- --eval-dataset takes the list)
  3. score (entropy stage, or scripts/oracle_kl.py for reverse KL) and
     select trajectories per the arm's rule (apod.selection)
  4. train refresh_every steps (train stage under torchrun, continuing the
     run-level LR schedule at --global-step-offset = step; Adam state
     carried by train.persist_optimizer)
  5. checkpoint; prune weights to keep_checkpoints, drop the consumed
     optimizer state

A final eval at step steps_total (refresh index = refreshes) measures the
last checkpoint; then apod.plotting renders the curves. metrics.jsonl holds
one row per (arm, step), upserted, and apod.tracking (when present) gets
the same row via log_refresh.

On disk a refresh's work lives in ``arms/<arm>/rounds/round_<r>``: the
stages' existing layout (apod.paths) and ``--round`` flag, kept as
identifiers -- round_<r> IS refresh r; nothing is counted in rounds. The
rollout stage reads questions from the fixed path ``pool/prompts.jsonl``
(its field names ``prompt`` and ``round`` mean question text and refresh
index); arms may have different questions, so the driver keeps one file per
arm (``pool/questions_<arm>.jsonl``) and re-points ``pool/prompts.jsonl`` at
the running arm's file before its stages launch, then verifies the rollouts
are that arm's questions.

Resume: every stage leaves a done-marker (or its output artifact); a resumed
run skips finished stages and continues from any refresh boundary. An
existing run dir keeps the resolved_config.yaml it started with.
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

from apod import paths
from apod.datasets.io import read_jsonl, read_shards, write_jsonl
from apod.selection import canonical_rule, needs_entropy, needs_reverse_kl, select_trajectories

try:  # per-step diagnostics + W&B (sibling module); the loop runs without it
    from apod import tracking
except ImportError:  # pragma: no cover - depends on the merge state
    tracking = None

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTION_SOURCES = ("pool_random", "bank_bucket", "bank_top_entropy", "bank_random")
WALL_CLOCK_KEYS = ("rollout_eval_s", "entropy_s", "oracle_s", "train_s")


def _log(msg: str) -> None:
    logger.info(msg)


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """Training-step budget and what it implies per refresh."""

    steps_total: int
    refresh_every: int
    effective_batch: int
    k: int  # trajectories trained per question (selection.k)
    num_rollouts: int

    @property
    def refreshes(self) -> int:
        return self.steps_total // self.refresh_every

    @property
    def trajectories_per_refresh(self) -> int:
        return self.refresh_every * self.effective_batch

    @property
    def questions_per_refresh(self) -> int:
        return self.trajectories_per_refresh // self.k

    @property
    def num_questions(self) -> int:
        return self.refreshes * self.questions_per_refresh


def budget_from(cfg: DictConfig) -> Budget:
    b = Budget(
        steps_total=int(cfg.driver.steps_total),
        refresh_every=int(cfg.driver.refresh_every),
        effective_batch=int(cfg.train.effective_batch),
        k=int(cfg.selection.k),
        num_rollouts=int(cfg.rollout.num_rollouts),
    )
    if b.steps_total % b.refresh_every:
        raise ValueError(f"steps_total {b.steps_total} is not a multiple of refresh_every {b.refresh_every}")
    if b.trajectories_per_refresh % b.k:
        raise ValueError(
            f"{b.trajectories_per_refresh} trajectories per refresh is not a multiple of selection.k={b.k}"
        )
    if b.k > b.num_rollouts:
        raise ValueError(f"selection.k={b.k} exceeds rollout.num_rollouts={b.num_rollouts}")
    return b


# ---------------------------------------------------------------------------
# config / run dir
# ---------------------------------------------------------------------------


def compose_config(overrides: list[str]) -> DictConfig:
    """conf/ composed as ``python -m apod.driver <overrides>`` sees it."""

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf")):
        return compose(config_name="config", overrides=overrides)


def prepare_config(cfg: DictConfig, run_dir: Path) -> DictConfig:
    """The run's resolved_config.yaml: written once, never rewritten.

    A fresh run dir gets the composed conf/ plus the driver's derived
    stamps (absolute output_dir, rollout.num_prompts = questions per
    refresh, eval_sets.<name> for the monitor sets). An existing run dir
    keeps the file it started with -- every stage and every resume reads
    that file, and conf/ may have moved on since.
    """

    path = run_dir / "resolved_config.yaml"
    if path.exists():
        _log(f"existing run: using {path} (CLI composition ignored)")
        return OmegaConf.load(path)
    budget = budget_from(cfg)
    with open_dict(cfg):
        cfg.output_dir = str(run_dir)
        cfg.rollout.num_prompts = budget.questions_per_refresh
        cfg.eval_sets = {}
        for name in cfg.driver.monitor_sets:
            conf = REPO_ROOT / "conf" / "eval" / f"{name}.yaml"
            if not conf.exists():
                raise FileNotFoundError(f"driver.monitor_sets names {name!r} but {conf} does not exist")
            cfg.eval_sets[str(name)] = OmegaConf.load(conf)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=path, resolve=True)
    _log(f"wrote {path}")
    return OmegaConf.load(path)


# ---------------------------------------------------------------------------
# question sources (question selection: which questions get teacher effort)
# ---------------------------------------------------------------------------


def _from_bank(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "question": row["question"],
        "reference": row["reference"],
        "bucket": row["bucket"],
        "question_entropy": row.get("question_entropy"),
        "bank_example_index": row["example_index"],
    }


def parse_source(source: str) -> tuple[str, str]:
    """``(kind, bucket)`` of a question source string; bucket is '' unless bank_bucket."""

    kind, _, bucket = source.partition(":")
    if kind not in QUESTION_SOURCES or (kind == "bank_bucket") != bool(bucket):
        raise ValueError(
            f"unknown question source {source!r}; expected pool_random, bank_bucket:<bucket>, "
            "bank_top_entropy or bank_random"
        )
    return kind, bucket


def select_questions(
    source: str, *, n: int, seed: int, bank_dir: Path, dry_run: bool, data_cfg: DictConfig | None = None
) -> list[dict[str, Any]]:
    """The arm's n questions under ``source`` (conf/driver.yaml).

    pool_random        seeded sample of the OpenThoughts pool
                       (apod.datasets.load.load_examples)
    bank_bucket:<b>    the first n bank questions in bucket b
    bank_top_entropy   the n highest question_entropy bank questions
                       (ties -> lower bank example_index)
    bank_random        the first n bank questions that have a
                       question_entropy -- the same candidates
                       bank_top_entropy ranks, so run 3's arms differ only
                       in the ranking

    The bank (apod.bank, ``bank_dir/questions.jsonl``) lists its questions in
    the pool's seeded order, so "the first n" of a bank source is a seeded
    random sample and every arm of a run reads the same file.
    """

    kind, bucket = parse_source(source)
    if kind == "pool_random":
        if dry_run:
            return [
                {"id": f"dry-{i}", "question": f"Question {i}: what is {i} + {i}?", "reference": str(2 * i)}
                for i in range(n)
            ]
        from apod.datasets.load import load_examples

        examples = load_examples(
            str(data_cfg.dataset), n=n, seed=seed, split=data_cfg.split
        )
        return [{"id": ex["id"], "question": ex["prompt"], "reference": ex["answer"]} for ex in examples]

    from apod import bank

    rows = bank.load_bank(bank_dir)
    if not rows:
        raise FileNotFoundError(f"question bank {bank_dir}/questions.jsonl is missing or empty (python -m apod.bank)")
    if kind == "bank_bucket":
        if bucket not in bank.BUCKETS:
            raise ValueError(f"question source {source!r}: unknown correctness bucket; expected one of {bank.BUCKETS}")
        candidates = bank.bucket_questions(rows, bucket)
    else:
        candidates = [r for r in rows if r.get("question_entropy") is not None]
    if len(candidates) < n:
        raise ValueError(
            f"question source {source!r}: {len(candidates)} candidate questions in {bank_dir} "
            f"(buckets: {dict(bank.bucket_counts(rows))}), need {n}"
        )
    if kind == "bank_top_entropy":
        chosen = sorted(candidates, key=lambda r: (-r["question_entropy"], r["example_index"]))[:n]
    else:
        chosen = candidates[:n]
    return [_from_bank(r) for r in chosen]


# ---------------------------------------------------------------------------
# eval summaries / metrics
# ---------------------------------------------------------------------------


def summarize_eval(rows: list[dict[str, Any]], *, num_problems: int, num_samples: int) -> dict[str, Any]:
    """Strict (and loose) avg@n / pass@n, cap-hit and mean length of one eval."""

    expected = num_problems * num_samples
    if len(rows) != expected:
        # Markers alone are not proof: a lost shard file with a surviving
        # marker would otherwise average over half the questions.
        raise RuntimeError(f"expected {num_problems} x {num_samples} = {expected} eval rows, found {len(rows)}")
    by_problem: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_problem[r["problem_index"]].append(r)

    def strict(r: dict[str, Any]) -> bool:
        return bool(r["correct"]) and bool(r.get("has_boxed", True))

    return {
        "strict_avg_at_n": statistics.mean(float(strict(r)) for r in rows),
        "strict_pass_at_n": statistics.mean(float(any(strict(r) for r in g)) for g in by_problem.values()),
        "avg_at_n": statistics.mean(float(r["correct"]) for r in rows),
        "pass_at_n": statistics.mean(float(any(r["correct"] for r in g)) for g in by_problem.values()),
        "cap_hit_rate": statistics.mean(float(r["truncated"]) for r in rows),
        "mean_response_length": statistics.mean(float(r["response_length"]) for r in rows),
        "num_problems": len(by_problem),
        "num_samples": num_samples,
    }


def _mean_or_none(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


class MetricsFile:
    """metrics.jsonl with idempotent upserts keyed by (arm, step)."""

    def __init__(self, path: Path, arm_order: list[str]):
        self.path = path
        self.arm_order = arm_order
        self.rows: dict[tuple[str, int], dict] = {(r["arm"], r["step"]): r for r in read_jsonl(path)}

    def upsert(self, row: dict) -> dict:
        key = (row["arm"], row["step"])
        old = self.rows.get(key)
        if old is not None:
            # Timings are only measurable when the stage ran this invocation;
            # keep previously recorded values for stages skipped on resume.
            for wc_key, value in (old.get("wall_clock") or {}).items():
                if row["wall_clock"].get(wc_key) is None:
                    row["wall_clock"][wc_key] = value
        self.rows[key] = row
        order = {a: i for i, a in enumerate(self.arm_order)}
        write_jsonl(
            self.path,
            sorted(self.rows.values(), key=lambda r: (order.get(r["arm"], len(order)), r["step"])),
        )
        return row


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


class Driver:
    def __init__(self, cfg: DictConfig, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        self.budget = budget_from(cfg)
        self.num_gpus = int(cfg.num_gpus)
        self.dry_run = bool(cfg.driver.dry_run)
        self.resume = bool(cfg.resume)
        self.arms: dict[str, dict[str, str]] = {
            str(name): {"question_source": str(spec.question_source), "selection": str(spec.selection)}
            for name, spec in cfg.driver.arms.items()
        }
        if not self.arms:
            raise ValueError("cfg.driver.arms is empty; pick an experiment (+experiment=r1_correctness_8k)")
        for spec in self.arms.values():  # a typo must fail before any GPU work
            canonical_rule(spec["selection"])
            parse_source(spec["question_source"])
        self.monitor_sets = [str(s) for s in cfg.driver.monitor_sets]
        self.metrics = MetricsFile(run_dir / "metrics.jsonl", list(self.arms))

    # --- eval protocols -----------------------------------------------------

    def eval_protocol(self, name: str | None) -> tuple[str, str, DictConfig]:
        """(eval subdir, pool file, protocol) for cfg.eval (None) or a monitor set."""

        if name is None:
            return "eval", "eval_problems.jsonl", self.cfg.eval
        return f"eval_{name}", f"eval_problems_{name}.jsonl", self.cfg.eval_sets[name]

    # --- stage launching -----------------------------------------------------

    def launch(self, specs: list[tuple[list[str], str, str]]) -> None:
        """(cmd, gpus, tag) triples run concurrently, CUDA_VISIBLE_DEVICES pinned.

        With driver.dry_run the stage is an in-process stub (apod.dry_run)
        that writes the same files and markers; every launch is recorded in
        dry_run_launches.jsonl either way, so tests can check the commands.
        """

        if self.dry_run:
            from apod.dry_run import run_stub

            for cmd, gpus, tag in specs:
                _log(f"[{tag}] dry-run (CUDA_VISIBLE_DEVICES={gpus}): {' '.join(cmd[1:])}")
                with (self.run_dir / "dry_run_launches.jsonl").open("a") as f:
                    f.write(json.dumps({"tag": tag, "gpus": gpus, "cmd": cmd[1:]}) + "\n")
                run_stub(cmd, self.run_dir)
            return

        procs = []
        for cmd, gpus, tag in specs:
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": _map_gpu(gpus)}
            _log(f"[{tag}] launch (CUDA_VISIBLE_DEVICES={gpus}): {' '.join(cmd[1:])}")
            proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
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

    def stage_cmd(self, module: str, arm: str, refresh: int, extra: list[str]) -> list[str]:
        return [
            sys.executable, "-m", f"apod.stages.{module}",
            "--run-dir", str(self.run_dir), "--arm", arm, "--round", str(refresh), *extra,
        ]

    def sharded(self, build_cmd, tag: str) -> list[tuple[list[str], str, str]]:
        return [(build_cmd(k), str(k), f"{tag}/shard{k}") for k in range(self.num_gpus)]

    # --- questions -----------------------------------------------------------

    def questions_file(self, arm: str) -> Path:
        return self.run_dir / "pool" / f"questions_{arm}.jsonl"

    def write_questions(self, arm: str) -> list[dict[str, Any]]:
        """``pool/questions_<arm>.jsonl`` in the rollout stage's row schema
        (``prompt`` = question text, ``round`` = refresh index), once."""

        path = self.questions_file(arm)
        if path.exists() and self.resume:
            return read_jsonl(path)
        b = self.budget
        questions = select_questions(
            self.arms[arm]["question_source"],
            n=b.num_questions,
            seed=int(self.cfg.seed),
            bank_dir=Path(str(self.cfg.driver.bank_dir)),
            dry_run=self.dry_run,
            data_cfg=self.cfg.data,
        )
        rows = []
        for i, q in enumerate(questions):
            row = {"example_index": i, "id": q["id"], "prompt": q["question"], "reference": q["reference"],
                   "round": i // b.questions_per_refresh}
            row.update({k: v for k, v in q.items() if k not in ("id", "question", "reference")})
            rows.append(row)
        write_jsonl(path, rows)
        _log(f"[{arm}] questions: {len(rows)} from {self.arms[arm]['question_source']} -> {path}")
        return rows

    def point_pool_at(self, arm: str) -> None:
        """Re-point ``pool/prompts.jsonl`` (the stage's fixed path) at the arm's file."""

        link = self.run_dir / "pool" / "prompts.jsonl"
        target = self.questions_file(arm).name
        if link.is_symlink():
            if os.readlink(link) == target:
                return
            link.unlink()
        elif link.exists():
            raise FileExistsError(f"{link} is a regular file; this driver expects a per-arm symlink")
        link.symlink_to(target)

    def materialize_eval_sets(self) -> None:
        """Pin every eval set to pool/ once, so problem_index -> question never drifts."""

        for name in [None, *self.monitor_sets]:
            _, pool_file, protocol = self.eval_protocol(name)
            path = self.run_dir / "pool" / pool_file
            if path.exists() and self.resume:
                continue
            n = int(protocol.num_problems)
            if self.dry_run:
                problems = [{"id": f"{protocol.dataset}-{i}", "prompt": f"eval {i}", "answer": str(i)} for i in range(n)]
            else:
                from apod.datasets.load import load_examples

                problems = load_examples(str(protocol.dataset), n=n, seed=int(self.cfg.seed))
            write_jsonl(path, problems)
            _log(f"eval set {protocol.dataset}: {len(problems)} questions -> {path}")

    # --- resume checks -------------------------------------------------------

    def markers_present(self, stage_dir: Path) -> bool:
        return all((stage_dir / f"done.shard{k}").exists() for k in range(self.num_gpus))

    def train_done(self, rdir: Path) -> bool:
        return (rdir / "train" / "done.shard0").exists() and (rdir / "checkpoint" / "config.json").exists()

    def model_path(self, arm: str, refresh: int) -> str:
        """The weights refresh ``refresh`` evaluates and rolls out from."""

        if refresh == 0:
            return str(self.cfg.model.student_id)
        return str(paths.checkpoint_dir(self.run_dir, arm, refresh - 1))

    # --- stages --------------------------------------------------------------

    def reuse_step0_eval(self, arm: str, subdir: str, expected_rows: int) -> None:
        """Step 0 evaluates the identical base model in every arm: copy a
        finished arm's rows and markers instead of regenerating (~30 min per
        eval set per arm at the real config). Rollouts are never shared."""

        dst = paths.round_dir(self.run_dir, arm, 0) / subdir
        if self.markers_present(dst):
            return
        for other in self.arms:
            src = paths.round_dir(self.run_dir, other, 0) / subdir
            if other == arm or not self.markers_present(src):
                continue
            rows = read_shards(src, "eval.shard*.jsonl")
            if len(rows) != expected_rows:
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for f in [*src.glob("eval.shard*.jsonl"), *src.glob("done.shard*")]:
                shutil.copy2(f, dst / f.name)
            (dst / "reused_from.json").write_text(json.dumps({"arm": other, "reason": "step 0 = base model"}) + "\n")
            _log(f"[{arm}] step-0 {subdir} reused from arm {other}")
            return

    def run_rollout_eval(self, arm: str, refresh: int, *, eval_only: bool) -> float | None:
        """One rollout_eval engine session: cfg.eval and every monitor set
        (``--eval-dataset <cfg.eval.dataset> <monitor>...``, each into its own
        eval dir), plus this refresh's rollouts unless eval_only."""

        rdir = paths.round_dir(self.run_dir, arm, refresh)
        sets = [None, *self.monitor_sets]
        subdirs = []
        for name in sets:
            subdir, _, protocol = self.eval_protocol(name)
            subdirs.append(subdir)
            if refresh == 0:
                self.reuse_step0_eval(arm, subdir, int(protocol.num_problems) * int(protocol.num_samples))
        done = all(self.markers_present(rdir / subdir) for subdir in subdirs) and (
            eval_only or self.markers_present(rdir / "rollouts")
        )
        tag = f"{arm}/step{refresh * self.budget.refresh_every:03d}/eval" + ("" if eval_only else "+rollouts")
        if self.resume and done:
            _log(f"[{tag}] skip (done markers present)")
            return None
        extra = ["--num-shards", str(self.num_gpus)]
        if eval_only:
            extra.append("--eval-only")
        extra += ["--eval-dataset", str(self.cfg.eval.dataset), *self.monitor_sets]
        t0 = time.time()
        self.launch(self.sharded(lambda k: self.stage_cmd("rollout_eval", arm, refresh, ["--shard", str(k), *extra]), tag))
        return time.time() - t0

    def verify_rollouts(self, arm: str, refresh: int, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The refresh's trajectories must be THIS arm's questions (guards the
        pool symlink) with every rollout present."""

        rdir = paths.round_dir(self.run_dir, arm, refresh)
        trajectories = read_shards(rdir / "rollouts", "trajectories.shard*.jsonl")
        block = [q for q in questions if q["round"] == refresh]
        expected_ids = {q["id"] for q in block}
        got_ids = {t["id"] for t in trajectories}
        if got_ids != expected_ids:
            raise RuntimeError(
                f"[{arm}] refresh {refresh}: rollouts cover {len(got_ids)} question ids, "
                f"{len(got_ids - expected_ids)} not in this arm's block of {len(expected_ids)}; "
                "pool/prompts.jsonl pointed at another arm's questions when the stage ran"
            )
        n_expected = len(block) * self.budget.num_rollouts
        if len(trajectories) != n_expected:
            raise RuntimeError(f"[{arm}] refresh {refresh}: {len(trajectories)} trajectories, expected {n_expected}")
        return trajectories

    def run_entropy(self, arm: str, refresh: int) -> float | None:
        rdir = paths.round_dir(self.run_dir, arm, refresh)
        tag = f"{arm}/step{refresh * self.budget.refresh_every:03d}/entropy"
        if self.resume and self.markers_present(rdir / "entropy"):
            _log(f"[{tag}] skip (done markers present)")
            return None
        t0 = time.time()
        self.launch(self.sharded(
            lambda k: self.stage_cmd("entropy", arm, refresh, ["--shard", str(k), "--num-shards", str(self.num_gpus)]), tag,
        ))
        return time.time() - t0

    def oracle_rows(self, rdir: Path) -> dict[tuple[int, int], dict[str, Any]]:
        """Last scored row per trajectory (a resume can append a re-score)."""

        mc = str(self.cfg.driver.kl_estimator) == "mc"
        stem, done_key = ("oracle_kl_mc", "rkl_mc") if mc else ("oracle_kl", "overlap_ratio_top16")
        rows: dict[tuple[int, int], dict[str, Any]] = {}
        for r in read_shards(rdir / "oracle", f"{stem}.shard*.jsonl"):
            if done_key in r:
                rows[(r["example_index"], r["rollout_index"])] = r
        return rows

    def run_oracle(self, arm: str, refresh: int, expected: int) -> float | None:
        """scripts/oracle_kl.py over the refresh's rollouts under the weights that produced them."""

        rdir = paths.round_dir(self.run_dir, arm, refresh)
        tag = f"{arm}/step{refresh * self.budget.refresh_every:03d}/oracle"
        if self.resume and len(self.oracle_rows(rdir)) >= expected:
            _log(f"[{tag}] skip (scored)")
            return None
        estimator = str(self.cfg.driver.kl_estimator)
        t0 = time.time()
        self.launch(self.sharded(
            lambda k: [
                sys.executable, "scripts/oracle_kl.py",
                "--run-dir", str(self.run_dir), "--arm", arm, "--round", str(refresh),
                "--shard", str(k), "--num-shards", str(self.num_gpus),
                "--student-path", self.model_path(arm, refresh), "--estimator", estimator,
            ],
            tag,
        ))
        n = len(self.oracle_rows(rdir))
        if n < expected:
            raise RuntimeError(f"[{tag}]: {n} scored trajectories < expected {expected}")
        return time.time() - t0

    def select(self, arm: str, refresh: int, trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge the rule's scores into the trajectories and write selected.jsonl."""

        rdir = paths.round_dir(self.run_dir, arm, refresh)
        rule = self.arms[arm]["selection"]
        path = rdir / "selected" / "selected.jsonl"
        if self.resume and path.exists():
            return read_jsonl(path)
        if needs_entropy(rule):
            scores = {(r["example_index"], r["rollout_index"]): r for r in read_shards(rdir / "entropy", "entropy.shard*.jsonl")}
            for t in trajectories:
                scored = scores.get((t["example_index"], t["rollout_index"]))
                t["entropy"] = scored["entropy"] if scored else None
        if needs_reverse_kl(rule):
            key = "rkl_mc" if str(self.cfg.driver.kl_estimator) == "mc" else "mean_reverse_kl"
            scores = self.oracle_rows(rdir)
            for t in trajectories:
                scored = scores.get((t["example_index"], t["rollout_index"]))
                t["mean_reverse_kl"] = scored[key] if scored else None
        selected = select_trajectories(
            rule, trajectories, k=self.budget.k, num_rollouts=self.budget.num_rollouts, seed=int(self.cfg.seed)
        )
        if len(selected) != self.budget.trajectories_per_refresh:
            raise RuntimeError(
                f"[{arm}] refresh {refresh}: selected {len(selected)} trajectories, "
                f"expected {self.budget.trajectories_per_refresh} (= {self.budget.refresh_every} steps x "
                f"{self.budget.effective_batch})"
            )
        write_jsonl(path, selected)  # atomic: bare existence is the resume marker
        return selected

    def run_train(self, arm: str, refresh: int) -> float | None:
        rdir = paths.round_dir(self.run_dir, arm, refresh)
        step = refresh * self.budget.refresh_every
        tag = f"{arm}/step{step:03d}/train"
        if self.resume and self.train_done(rdir):
            _log(f"[{tag}] skip (checkpoint present)")
            return None
        # --global-step-offset = training steps already completed for this
        # arm (also after a resume): the train stage continues the run-level
        # LR schedule (train.total_training_steps, warmup once at step 0)
        # from there and puts its per-step diagnostics on the run's step axis.
        args = ["--run-dir", str(self.run_dir), "--arm", arm, "--round", str(refresh), "--global-step-offset", str(step)]
        if self.num_gpus > 1:
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={self.num_gpus}", "-m", "apod.stages.train", *args]
            gpus = ",".join(str(k) for k in range(self.num_gpus))
        else:
            cmd = [sys.executable, "-m", "apod.stages.train", *args]
            gpus = str(self.cfg.train.train_gpu)
        t0 = time.time()
        self.launch([(cmd, gpus, tag)])
        if not self.train_done(rdir):
            raise RuntimeError(f"[{tag}] finished without train/done.shard0 + checkpoint/config.json")
        return time.time() - t0

    def prune(self, arm: str, refresh: int) -> None:
        """Keep weights of the newest keep_checkpoints refreshes and only the
        newest optimizer state (the previous one was consumed by this train)."""

        keep = int(self.cfg.keep_checkpoints)
        for old in range(refresh - keep + 1):
            for weights in paths.checkpoint_dir(self.run_dir, arm, old).glob("*.safetensors"):
                weights.unlink()
                _log(f"[{arm}] pruned {weights}")
        if refresh > 0:
            stale = paths.checkpoint_dir(self.run_dir, arm, refresh - 1) / "optimizer_state.pt"
            if stale.exists():
                stale.unlink()
                _log(f"[{arm}] pruned {stale}")

    def prune_finished(self, arm: str) -> None:
        """A finished arm keeps only its final weights: the optimizer state
        and the penultimate checkpoint exist for resuming a train pass, and
        there is none left (15 GB -> 3.8 GB per arm; r3 filled the disk)."""

        last = self.budget.refreshes - 1  # the final refresh is eval-only
        for old in range(last):
            for weights in paths.checkpoint_dir(self.run_dir, arm, old).glob("*.safetensors"):
                weights.unlink()
                _log(f"[{arm}] pruned {weights}")
        state = paths.checkpoint_dir(self.run_dir, arm, last) / "optimizer_state.pt"
        if state.exists():
            state.unlink()
            _log(f"[{arm}] pruned {state}")

    # --- one refresh -----------------------------------------------------------

    def run_refresh(self, arm: str, refresh: int, questions: list[dict[str, Any]]) -> dict[str, Any]:
        b = self.budget
        step = refresh * b.refresh_every
        final = refresh == b.refreshes
        rdir = paths.round_dir(self.run_dir, arm, refresh)
        wall: dict[str, float | None] = {k: None for k in WALL_CLOCK_KEYS}
        _log(f"== [{arm}] step {step:03d} (refresh {refresh}{', final eval' if final else ''}) :: {self.model_path(arm, refresh)}")

        # 1 + 2: eval on cfg.eval and the monitor sets + this refresh's
        # rollouts, one engine session.
        wall["rollout_eval_s"] = self.run_rollout_eval(arm, refresh, eval_only=final)
        evals = {}
        for name in [None, *self.monitor_sets]:
            subdir, _, protocol = self.eval_protocol(name)
            summary = summarize_eval(
                read_shards(rdir / subdir, "eval.shard*.jsonl"),
                num_problems=int(protocol.num_problems), num_samples=int(protocol.num_samples),
            )
            (rdir / subdir / "summary.json").write_text(json.dumps(summary, indent=2))
            evals[str(protocol.dataset)] = summary
        row: dict[str, Any] = {
            "arm": arm, "step": step, "refresh": refresh, "model_path": self.model_path(arm, refresh),
            "trajectories_trained": step * b.effective_batch,
            "eval": evals, "rollouts": None, "selected": None,
            "train_loss_mean": None, "train_loss_final": None, "tokens_trained": None,
            "wall_clock": wall,
        }
        if final:
            return self.record(row)

        # 3: score + select.
        trajectories = self.verify_rollouts(arm, refresh, questions)
        rule = self.arms[arm]["selection"]
        if needs_entropy(rule):
            wall["entropy_s"] = self.run_entropy(arm, refresh)
        if needs_reverse_kl(rule):
            wall["oracle_s"] = self.run_oracle(arm, refresh, expected=len(trajectories))
        selected = self.select(arm, refresh, trajectories)

        # 4 + 5: train this block of steps, checkpoint, prune.
        wall["train_s"] = self.run_train(arm, refresh)
        train_summary = json.loads((rdir / "train" / "summary.json").read_text())
        self.prune(arm, refresh)

        strict = lambda t: float(bool(t["correct"]) and bool(t.get("has_boxed", True)))  # noqa: E731
        row["rollouts"] = {
            "num_questions": len({t["example_index"] for t in trajectories}),
            "num_trajectories": len(trajectories),
            "cap_hit_rate": statistics.mean(float(t["truncated"]) for t in trajectories),
            "strict_accuracy": statistics.mean(strict(t) for t in trajectories),
            "mean_response_length": statistics.mean(float(t["response_length"]) for t in trajectories),
        }
        by_key = {(t["example_index"], t["rollout_index"]): t for t in trajectories}
        row["selected"] = {
            "num_trajectories": len(selected),
            "mean_entropy": _mean_or_none([s.get("entropy") for s in selected]),
            "mean_reverse_kl": _mean_or_none([s.get("mean_reverse_kl") for s in selected]),
            "cap_hit_rate": statistics.mean(float(s["truncated"]) for s in selected),
            "strict_accuracy": statistics.mean(strict(by_key[(s["example_index"], s["rollout_index"])]) for s in selected),
            "mean_response_length": statistics.mean(float(s["response_length"]) for s in selected),
        }
        # Loss of the block trained AFTER this eval (steps step .. step+refresh_every).
        row["train_loss_mean"] = train_summary.get("train_loss_mean")
        row["train_loss_final"] = train_summary.get("train_loss_final")
        row["tokens_trained"] = train_summary.get("tokens_trained")
        return self.record(row)

    def record(self, row: dict[str, Any]) -> dict[str, Any]:
        row = self.metrics.upsert(row)
        m = row["eval"][str(self.cfg.eval.dataset)]
        _log(
            f"[{row['arm']}] step {row['step']:03d}: {self.cfg.eval.dataset} strict avg@{m['num_samples']} "
            f"{m['strict_avg_at_n']:.4f} pass@{m['num_samples']} {m['strict_pass_at_n']:.3f} cap-hit {m['cap_hit_rate']:.3f}"
            + "".join(
                f" | {name} strict avg@{s['num_samples']} {s['strict_avg_at_n']:.4f} cap-hit {s['cap_hit_rate']:.3f}"
                for name, s in row["eval"].items() if name != str(self.cfg.eval.dataset)
            )
        )
        if tracking is not None:
            # Open the arm's W&B run only for this log call: a run id can be
            # live in one process at a time ("run ID ... is in use"), and the
            # train stage resumes the same id in its own process in between.
            tracking.init(self.cfg, self.run_dir, row["arm"])
            try:
                tracking.log_refresh(row["step"], row)
            finally:
                tracking.finish()
        return row

    # --- whole run -------------------------------------------------------------

    def run(self) -> Path:
        b = self.budget
        _log(
            f"budget: {b.steps_total} steps, refresh every {b.refresh_every} -> {b.refreshes} refreshes x "
            f"{b.trajectories_per_refresh} trajectories ({b.questions_per_refresh} questions x {b.k} of "
            f"{b.num_rollouts} rollouts); {b.num_questions} questions per arm"
        )
        self.materialize_eval_sets()
        for arm in self.arms:
            questions = self.write_questions(arm)
            self.point_pool_at(arm)
            for refresh in range(b.refreshes + 1):
                self.run_refresh(arm, refresh, questions)
            self.prune_finished(arm)
        _log("all arms complete; rendering plots")
        from apod.plotting import plot_refresh_curves  # matplotlib Agg, no GPU

        return plot_refresh_curves(self.run_dir, band_points=float(self.cfg.driver.noise_band_points))


# ---------------------------------------------------------------------------
# subprocess plumbing
# ---------------------------------------------------------------------------


def _pump(proc: subprocess.Popen, tag: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info("[{}] {}", tag, line.rstrip())


def _map_gpu(logical: str) -> str:
    """Logical GPU indices through any parent CUDA_VISIBLE_DEVICES, so a
    driver launched on GPUs 2,3 never hands its children someone else's cards."""

    parent = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not parent:
        return logical
    visible = [d.strip() for d in parent.split(",") if d.strip()]
    try:
        return ",".join(visible[int(i)] for i in logical.split(","))
    except IndexError:
        raise RuntimeError(f"num_gpus needs logical GPU(s) {logical} but CUDA_VISIBLE_DEVICES={parent!r} is smaller") from None


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> Path:
    run_dir = Path(cfg.output_dir)
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.add(run_dir / "driver.log", enqueue=True)
    cfg = prepare_config(cfg, run_dir)
    _log(f"run dir: {run_dir}")
    return Driver(cfg, run_dir).run()


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    out = run(cfg)
    _log(f"done: {out}")


if __name__ == "__main__":
    main()
