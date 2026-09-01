"""The step-based driver's loop, on CPU, with the GPU stages stubbed.

Runs ``apod.driver`` under ``driver.dry_run=true`` for the three experiment
configs against a synthetic question bank and checks: step accounting (100
steps = 10 refreshes x 320 trajectories, --global-step-offset 0..90),
arm / question-source / selection dispatch (entropy stage only for
entropy_top_k, oracle scoring only for kl_*, bank buckets, top-entropy
questions), sharding (CUDA_VISIBLE_DEVICES per shard), checkpoint pruning
(keep-last-2 weights, one optimizer state), metrics rows (one per (arm,
step), both eval sets), the plot, resolved_config.yaml never rewritten,
resume from every refresh boundary and from mid-refresh states (stubs are
deterministic, so a resumed run must reproduce metrics.jsonl exactly), and
the pool-symlink guard.

Runs under pytest or as ``python -m tests.test_driver`` (no pytest needed).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from apod import paths
from apod.bank import BUCKETS
from apod.datasets.io import read_jsonl, write_jsonl
from apod.driver import Driver, budget_from, compose_config, prepare_config, run
from apod.selection import KL_TERTILES

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("r1_correctness_8k", "r2_trajsel_8k", "r3_qentropy_8k")
NUM_GPUS = 2


# --- fixtures ---------------------------------------------------------------


def _tmp(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"apod-{name}-"))


def _write_bank(bank_dir: Path, per_bucket: int = 850) -> list[dict]:
    """A synthetic apod/bank.py output (bank_dir/questions.jsonl, rows in the
    pool's seeded order): per_bucket questions per bucket."""

    rng = random.Random(7)
    rows = []
    for i in range(per_bucket * len(BUCKETS)):
        bucket = BUCKETS[i % len(BUCKETS)]
        rows.append({
            "example_index": i, "id": f"bank-{i}", "question": f"Bank question {i}?", "reference": str(i),
            "bucket": bucket,
            "question_entropy": None if bucket == "unlabelled" and i % 2 else rng.random(),
            "student_grades": [bool(rng.random() < 0.5) for _ in range(4)],
            "student_truncated": [bool(rng.random() < 0.3) for _ in range(4)],
            "student_lengths": [rng.randint(100, 8192) for _ in range(4)],
        })
    write_jsonl(bank_dir / "questions.jsonl", rows)
    return rows


def _run(experiment: str, out: Path, bank: Path, extra: list[str] = ()) -> Path:
    cfg = compose_config([
        f"+experiment={experiment}", "driver.dry_run=true", f"output_dir={out}",
        f"driver.bank_dir={bank}", f"num_gpus={NUM_GPUS}", *extra,
    ])
    return run(cfg)


def _metrics(out: Path) -> list[dict]:
    """metrics.jsonl rows without the timings (the one field a resume may change)."""

    return [{k: v for k, v in r.items() if k != "wall_clock"} for r in read_jsonl(out / "metrics.jsonl")]


def _launches(out: Path) -> list[dict]:
    return read_jsonl(out / "dry_run_launches.jsonl")


def _clear_launches(out: Path) -> None:
    (out / "dry_run_launches.jsonl").unlink(missing_ok=True)


def _flag(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _stage(cmd: list[str]) -> str:
    if "scripts/oracle_kl.py" in cmd:
        return "oracle"
    last_m = len(cmd) - 1 - cmd[::-1].index("-m")  # torchrun: -m torch.distributed.run ... -m apod.stages.train
    return cmd[last_m + 1].rsplit(".", 1)[-1]


# --- tests ------------------------------------------------------------------


def test_budget() -> None:
    for experiment in EXPERIMENTS:
        b = budget_from(compose_config([f"+experiment={experiment}"]))
        assert (b.steps_total, b.refresh_every, b.effective_batch) == (100, 10, 32)
        assert (b.refreshes, b.trajectories_per_refresh, b.questions_per_refresh, b.num_questions) == (10, 320, 80, 800)
    for bad in ("driver.steps_total=95", "selection.k=3", "selection.k=13"):
        try:
            budget_from(compose_config(["+experiment=r2_trajsel_8k", bad]))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} must be rejected")


def _check_full_run(experiment: str, out: Path, bank_rows: list[dict]) -> None:
    cfg = OmegaConf.load(out / "resolved_config.yaml")
    arms = list(cfg.driver.arms)
    b = budget_from(cfg)
    assert int(cfg.rollout.num_prompts) == 80, "questions per refresh stamped for the rollout stage"
    assert "aime2526" in cfg.eval_sets and int(cfg.eval_sets.aime2526.num_samples) == 16
    assert float(cfg.train.learning_rate) == 3.1623e-06 and bool(cfg.train.persist_optimizer)
    assert int(cfg.train.total_training_steps) == 100 and int(cfg.train.warmup_steps) == 5
    assert str(cfg.train.lr_scheduler_type) == "cosine_with_min_lr"

    # metrics: one row per (arm, step) at every refresh + the final eval.
    rows = read_jsonl(out / "metrics.jsonl")
    assert [(r["arm"], r["step"]) for r in rows] == [(a, s) for a in arms for s in range(0, 101, 10)]
    for r in rows:
        assert set(r["eval"]) == {"math500", "aime2526"}
        for name, n in (("math500", 4), ("aime2526", 16)):
            s = r["eval"][name]
            assert s["num_samples"] == n and s["num_problems"] == {"math500": 500, "aime2526": 60}[name]
            assert {"strict_avg_at_n", "strict_pass_at_n", "cap_hit_rate", "mean_response_length"} <= set(s)
        assert r["trajectories_trained"] == r["step"] * 32
        if r["step"] == 100:
            assert r["rollouts"] is None and r["selected"] is None and r["train_loss_mean"] is None
        else:
            assert r["rollouts"]["num_questions"] == 80
            assert r["rollouts"]["num_trajectories"] == 80 * b.num_rollouts
            assert r["selected"]["num_trajectories"] == 320
            assert r["train_loss_mean"] is not None

    # launches: sharding, dispatch, global step offsets.
    launches = _launches(out)
    for arm in arms:
        rule = str(cfg.driver.arms[arm].selection)
        mine = [l for l in launches if _flag(l["cmd"], "--arm") == arm]
        trains = [l for l in mine if _stage(l["cmd"]) == "train"]
        assert [int(_flag(l["cmd"], "--global-step-offset")) for l in trains] == list(range(0, 100, 10))
        assert all(l["gpus"] == "0,1" and "--nproc_per_node=2" in l["cmd"] for l in trains)
        for stage in ("rollout_eval", "entropy", "oracle"):
            for l in (l for l in mine if _stage(l["cmd"]) == stage):
                assert l["gpus"] == _flag(l["cmd"], "--shard") and _flag(l["cmd"], "--num-shards") == "2"
        entropies = [l for l in mine if _stage(l["cmd"]) == "entropy"]
        oracles = [l for l in mine if _stage(l["cmd"]) == "oracle"]
        assert bool(entropies) == (rule == "entropy_top_k"), (arm, rule)
        assert bool(oracles) == (rule in KL_TERTILES), (arm, rule)
        assert all("--estimator" in l["cmd"] and _flag(l["cmd"], "--estimator") == "exact" for l in oracles)
        for l in oracles:
            r = int(_flag(l["cmd"], "--round"))
            expected = str(cfg.model.student_id) if r == 0 else str(paths.checkpoint_dir(out, arm, r - 1))
            assert _flag(l["cmd"], "--student-path") == expected
        sessions = [l for l in mine if _stage(l["cmd"]) == "rollout_eval"]
        rollouts = [l for l in sessions if "--eval-only" not in l["cmd"]]
        finals = [l for l in sessions if "--eval-only" in l["cmd"]]
        assert len(rollouts) == 10 * NUM_GPUS and len(finals) == 1 * NUM_GPUS
        # MATH-500 and AIME in the SAME session at every refresh incl. the
        # final (never a second launch); step-0 evals reused from the first arm.
        for l in sessions:
            i = l["cmd"].index("--eval-dataset")
            assert l["cmd"][i + 1:i + 3] == ["math500", "aime2526"], l["cmd"]
        if arm != arms[0]:
            for subdir in ("eval", "eval_aime2526"):
                assert (paths.round_dir(out, arm, 0) / subdir / "reused_from.json").exists()

        # per refresh: 320 selected, the rule's scores merged.
        for r in range(10):
            rdir = paths.round_dir(out, arm, r)
            selected = read_jsonl(rdir / "selected" / "selected.jsonl")
            assert len(selected) == 320
            if rule == "entropy_top_k":
                assert all(s["entropy"] is not None for s in selected)
            if rule in KL_TERTILES:
                assert all(s["mean_reverse_kl"] is not None for s in selected)
            summary = json.loads((rdir / "train" / "summary.json").read_text())
            assert summary["global_step_offset"] == r * 10 and summary["steps"] == 10
        # pruning: weights only for the newest 2 refreshes, one optimizer state.
        with_weights = [r for r in range(10) if any(paths.checkpoint_dir(out, arm, r).glob("*.safetensors"))]
        assert with_weights == [8, 9], with_weights
        with_opt = [r for r in range(10) if (paths.checkpoint_dir(out, arm, r) / "optimizer_state.pt").exists()]
        assert with_opt == [9], with_opt
        assert all((paths.checkpoint_dir(out, arm, r) / "config.json").exists() for r in range(10))

    # questions per arm.
    ids_by_arm = {}
    for arm in arms:
        questions = read_jsonl(out / "pool" / f"questions_{arm}.jsonl")
        assert len(questions) == 800
        assert [q["example_index"] for q in questions] == list(range(800))
        assert all(q["round"] == q["example_index"] // 80 for q in questions)
        ids_by_arm[arm] = [q["id"] for q in questions]
        source = str(cfg.driver.arms[arm].question_source)
        if source.startswith("bank_bucket:"):
            assert all(q["bucket"] == source.split(":")[1] for q in questions)
        if source == "bank_top_entropy":
            chosen = {q["bank_example_index"] for q in questions}
            rest = [r["question_entropy"] for r in bank_rows if r["example_index"] not in chosen and r["question_entropy"] is not None]
            assert min(q["question_entropy"] for q in questions) >= max(rest)
        if source == "bank_random":
            assert all(q["question_entropy"] is not None for q in questions)
    if experiment == "r2_trajsel_8k":
        assert len({tuple(v) for v in ids_by_arm.values()}) == 1, "pool_random arms share the questions"
    if experiment == "r1_correctness_8k":
        assert len({tuple(v) for v in ids_by_arm.values()}) == len(arms)
    link = out / "pool" / "prompts.jsonl"
    assert link.is_symlink() and os.readlink(link) == f"questions_{arms[-1]}.jsonl"
    assert (out / "plots" / "accuracy_vs_steps.png").exists()


def test_kl_tertiles_partition_each_question() -> None:
    """kl_high/mid/low arms at the same refresh: ranks by KL are disjoint slices."""

    out, bank = _tmp("r2"), _tmp("bank")
    _write_bank(bank)
    _run("r2_trajsel_8k", out, bank)
    arm = "kl_high"
    rdir = paths.round_dir(out, arm, 2)
    oracle = {(r["example_index"], r["rollout_index"]): r["mean_reverse_kl"] for r in read_jsonl(rdir / "oracle" / "oracle_kl.shard0.jsonl") + read_jsonl(rdir / "oracle" / "oracle_kl.shard1.jsonl")}
    assert len(oracle) == 80 * 12
    selected = read_jsonl(rdir / "selected" / "selected.jsonl")
    by_q: dict[int, list[float]] = {}
    for s in selected:
        by_q.setdefault(s["example_index"], []).append(oracle[(s["example_index"], s["rollout_index"])])
    for q, kls in by_q.items():
        all_kls = sorted((v for (e, _), v in oracle.items() if e == q), reverse=True)
        assert sorted(kls, reverse=True) == all_kls[:4], "kl_high keeps the top 4 by reverse KL"
    shutil.rmtree(out)
    shutil.rmtree(bank)


def test_full_runs_and_never_rewrite_config() -> None:
    bank = _tmp("bank")
    bank_rows = _write_bank(bank)
    for experiment in EXPERIMENTS:
        out = _tmp(experiment)
        _run(experiment, out, bank)
        _check_full_run(experiment, out, bank_rows)
        # A second invocation (even with a different composition) is a no-op
        # that keeps the stored config byte-for-byte.
        before = (out / "resolved_config.yaml").read_bytes()
        metrics_before = (out / "metrics.jsonl").read_text()
        _clear_launches(out)
        _run(experiment, out, bank, extra=["driver.steps_total=50"])
        assert (out / "resolved_config.yaml").read_bytes() == before
        assert _launches(out) == []
        assert (out / "metrics.jsonl").read_text() == metrics_before
        shutil.rmtree(out)
    shutil.rmtree(bank)


def _restore_crash_state(out: Path, arm: str, boundary: int) -> None:
    """The disk as it was right after refresh ``boundary - 1`` completed:
    refreshes >= boundary absent, weights for the two newest refreshes and
    the newest optimizer state present (as before pruning)."""

    for r in range(boundary, 11):
        shutil.rmtree(paths.round_dir(out, arm, r), ignore_errors=True)
    for r in (boundary - 2, boundary - 1):
        if r >= 0:
            ckpt = paths.checkpoint_dir(out, arm, r)
            (ckpt / "model.safetensors").write_bytes(b"dry-run weights")
    if boundary >= 1:
        (paths.checkpoint_dir(out, arm, boundary - 1) / "optimizer_state.pt").write_bytes(b"dry-run optimizer state")
    rows = [r for r in read_jsonl(out / "metrics.jsonl") if not (r["arm"] == arm and r["step"] >= boundary * 10)]
    write_jsonl(out / "metrics.jsonl", rows)


def test_resume_from_every_boundary() -> None:
    bank = _tmp("bank")
    _write_bank(bank)
    out = _tmp("resume")
    _run("r3_qentropy_8k", out, bank)
    reference = _metrics(out)
    arm = list(OmegaConf.load(out / "resolved_config.yaml").driver.arms)[-1]
    for boundary in range(0, 11):
        _restore_crash_state(out, arm, boundary)
        _clear_launches(out)
        _run("r3_qentropy_8k", out, bank)
        launches = [l for l in _launches(out) if _flag(l["cmd"], "--arm") == arm]
        assert launches and all(_flag(l["cmd"], "--arm") == arm for l in _launches(out)), "only the unfinished arm runs"
        assert min(int(_flag(l["cmd"], "--round")) for l in launches) == boundary
        trains = [int(_flag(l["cmd"], "--global-step-offset")) for l in launches if _stage(l["cmd"]) == "train"]
        assert trains == list(range(boundary * 10, 100, 10)), (boundary, trains)
        assert _metrics(out) == reference, f"boundary {boundary}: metrics differ after resume"

    # Mid-refresh states at refresh 4: (a) rollouts + evals done, nothing
    # after; (b) selection done, train not; (c) train done, metrics row not.
    rdir = paths.round_dir(out, arm, 4)

    def mid(remove: list[str], expect: set[str]) -> None:
        _restore_crash_state(out, arm, 5)
        for sub in remove:
            shutil.rmtree(rdir / sub, ignore_errors=True)
        (paths.checkpoint_dir(out, arm, 3) / "optimizer_state.pt").write_bytes(b"x")
        rows = [r for r in read_jsonl(out / "metrics.jsonl") if not (r["arm"] == arm and r["step"] >= 40)]
        write_jsonl(out / "metrics.jsonl", rows)
        _clear_launches(out)
        _run("r3_qentropy_8k", out, bank)
        at4 = {_stage(l["cmd"]) for l in _launches(out) if int(_flag(l["cmd"], "--round")) == 4}
        assert at4 == expect, (remove, at4)
        assert _metrics(out) == reference

    mid(["selected", "train", "checkpoint"], {"train"})
    mid(["train", "checkpoint"], {"train"})
    mid([], set())
    # A lost monitor eval re-enters the session; the finished MATH-500 rows
    # and rollouts of that refresh are kept (deterministic stubs: same rows).
    mid(["eval_aime2526", "selected", "train", "checkpoint"], {"rollout_eval", "train"})
    shutil.rmtree(out)
    shutil.rmtree(bank)


def test_pool_link_guard() -> None:
    """Rollouts generated under another arm's pool file are refused."""

    bank = _tmp("bank")
    _write_bank(bank)
    out = _tmp("guard")
    cfg = compose_config(["+experiment=r1_correctness_8k", "driver.dry_run=true", f"output_dir={out}", f"driver.bank_dir={bank}"])
    cfg = prepare_config(cfg, out)
    driver = Driver(cfg, out)
    driver.materialize_eval_sets()
    questions = {arm: driver.write_questions(arm) for arm in ("both_right", "mixed")}
    driver.point_pool_at("both_right")
    driver.run_rollout_eval("mixed", 0, eval_only=False)  # stage ran under the wrong link
    try:
        driver.verify_rollouts("mixed", 0, questions["mixed"])
    except RuntimeError as e:
        assert "another arm" in str(e)
    else:
        raise AssertionError("rollouts of another arm's questions must be refused")
    driver.run_rollout_eval("both_right", 0, eval_only=False)  # the right link
    assert len(driver.verify_rollouts("both_right", 0, questions["both_right"])) == 320
    shutil.rmtree(out)
    shutil.rmtree(bank)


def test_plot_cli() -> None:
    bank = _tmp("bank")
    _write_bank(bank)
    out = _tmp("plot")
    _run("r3_qentropy_8k", out, bank)
    (out / "plots" / "accuracy_vs_steps.png").unlink()
    subprocess.run([sys.executable, "-m", "apod.plotting", "--run-dir", str(out), "--band-points", "7"], check=True, cwd=ROOT)
    assert (out / "plots" / "accuracy_vs_steps.png").exists()
    shutil.rmtree(out)
    shutil.rmtree(bank)


def test_selection_self_test() -> None:
    subprocess.run([sys.executable, str(ROOT / "apod" / "selection.py")], check=True, cwd=ROOT)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name} ...", flush=True)
            fn()
            print(f"{name} ok", flush=True)
    print("tests/test_driver.py passed")
