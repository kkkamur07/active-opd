"""End-to-end tests of the final program's pipeline (apod.pipeline) in dry-run mode.

Dry-run replaces every GPU stage with a deterministic stub (apod.dry_run), so these
tests exercise the real orchestration: budgets, question sources, launches and
sharding, step-0 sharing, selection, checkpoint retention, resume and guards.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from apod import paths
from apod.data.io import read_jsonl, write_jsonl
from apod.pipeline import Driver, budget_from, compose_config, prepare_config, run, select_questions, used_question_ids
from apod.question_bank import BUCKETS

ROOT = Path(__file__).resolve().parents[1]
NUM_GPUS = 2
REFRESHES = 4
STEPS = (0, 15, 30, 45, 60)


@dataclass(frozen=True)
class Expected:
    arms: tuple[str, ...]
    questions_per_refresh: int
    answers_sampled_per_refresh: int
    shares_step0: bool  # arms of the run draw the same step-0 questions


EXPERIMENTS: dict[str, Expected] = {
    "answer_level": Expected(("random", "kl_mid", "kl_high", "kl_low", "ent_top4"), 120, 1440, True),
    "question_kl": Expected(("qkl_mid", "qkl_high", "qkl_low"), 360, 1440, True),
    "question_entropy": Expected(("qent1_high",), 360, 1440, False),
    "correctness": Expected(("t_right_s_wrong", "both_right", "both_wrong", "mixed"), 120, 480, False),
}


# --- fixtures ---------------------------------------------------------------


def _tmp(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"apod-{name}-"))


def _write_bank(bank_dir: Path, per_bucket: int = 2000) -> list[dict]:
    """A synthetic question bank (questions.jsonl in pool order), per_bucket questions per bucket."""

    rng = random.Random(7)
    rows = []
    for i in range(per_bucket * len(BUCKETS)):
        bucket = BUCKETS[i % len(BUCKETS)]
        rows.append({
            "example_index": i, "id": f"bank-{i}", "question": f"Bank question {i}?", "reference": str(i),
            "bucket": bucket, "question_entropy": rng.random(),
            "student_grades": [bool(rng.random() < 0.5) for _ in range(4)],
            "student_truncated": [bool(rng.random() < 0.3) for _ in range(4)],
            "student_lengths": [rng.randint(100, 8192) for _ in range(4)],
            "teacher_grades": None if bucket == "unlabelled" else [bool(rng.random() < 0.5) for _ in range(4)],
            "teacher_label": None if bucket == "unlabelled" else "C",
        })
    write_jsonl(bank_dir / "questions.jsonl", rows)
    return rows


def _compose(experiment: str, out: Path, bank: Path, extra: tuple[str, ...] = ()) -> DictConfig:
    return compose_config([
        f"+experiment={experiment}", "driver.dry_run=true", f"output_dir={out}", f"driver.bank_dir={bank}",
        f"num_gpus={NUM_GPUS}", "tracking.mode=disabled", "driver.step0_eval_from=null",
        "driver.step0_rollouts_from=null", *extra,
    ])  # tracking disabled: a CPU test must never open a real W&B run


def _run(experiment: str, out: Path, bank: Path, extra: tuple[str, ...] = ()) -> Path:
    return run(_compose(experiment, out, bank, extra))


def _launches(out: Path) -> list[dict]:
    return read_jsonl(out / "dry_run_launches.jsonl")


def _clear_launches(out: Path) -> None:
    (out / "dry_run_launches.jsonl").unlink(missing_ok=True)


def _metrics(out: Path) -> list[dict]:
    """metrics.jsonl rows without timings (the one field a resume may change)."""

    return [{k: v for k, v in r.items() if k != "wall_clock"} for r in read_jsonl(out / "metrics.jsonl")]


def _flag(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _stage(cmd: list[str]) -> str:
    last_m = len(cmd) - 1 - cmd[::-1].index("-m")  # torchrun: -m torch.distributed.run ... -m apod.stages.train
    return cmd[last_m + 1].rsplit(".", 1)[-1]


def _ids(out: Path, arm: str, refresh: int) -> list[tuple[str, int]]:
    return [(q["id"], q["example_index"]) for q in read_jsonl(out / "pool" / f"questions_{arm}.jsonl") if q["round"] == refresh]


def _shared_bank() -> Path:
    bank = _tmp("bank")
    _write_bank(bank)
    return bank


# --- checks used by the full-run test ----------------------------------------


def _check_config(cfg: DictConfig) -> None:
    assert float(cfg.train.learning_rate) == 3.1623e-06 and bool(cfg.train.persist_optimizer)
    assert int(cfg.train.total_training_steps) == 60 and int(cfg.train.warmup_steps) == 3
    assert str(cfg.train.lr_scheduler_type) == "cosine_with_min_lr"
    assert str(cfg.model.student_id) == "Qwen/Qwen3.5-2B" and int(cfg.sampling.max_new_tokens) == 8192


def _check_metrics(out: Path, arms: tuple[str, ...], exp: Expected) -> None:
    rows = read_jsonl(out / "metrics.jsonl")
    assert sorted((r["arm"], r["step"]) for r in rows) == sorted((a, s) for a in arms for s in STEPS)
    for r in rows:
        s = r["eval"]["math500"]
        assert set(r["eval"]) == {"math500"} and s["num_samples"] == 4 and s["num_problems"] == 500
        assert r["trajectories_trained"] == r["step"] * 32
        if r["step"] == STEPS[-1]:
            assert r["rollouts"] is None and r["selected"] is None
            continue
        assert r["rollouts"]["num_questions"] == exp.questions_per_refresh
        assert r["rollouts"]["num_trajectories"] == exp.answers_sampled_per_refresh
        assert r["selected"]["num_trajectories"] == 480


def _check_launches(out: Path, arm: str, first_arm: bool, exp: Expected) -> None:
    mine = [l for l in _launches(out) if _flag(l["cmd"], "--arm") == arm]
    trains = [l for l in mine if _stage(l["cmd"]) == "train"]
    assert [int(_flag(l["cmd"], "--global-step-offset")) for l in trains] == [0, 15, 30, 45]
    assert all(l["gpus"] == "0,1" and "--nproc_per_node=2" in l["cmd"] for l in trains)
    for l in (l for l in mine if _stage(l["cmd"]) in ("generate", "entropy", "teacher_kl")):
        assert l["gpus"] == _flag(l["cmd"], "--shard")
    reused = exp.shares_step0 and not first_arm
    scored_refreshes = REFRESHES - 1 if reused else REFRESHES
    for stage in ("entropy", "teacher_kl"):  # score_all: every answer of every rule is scored
        assert sum(_stage(l["cmd"]) == stage for l in mine) == scored_refreshes * NUM_GPUS, (arm, stage)
    sessions = [l for l in mine if _stage(l["cmd"]) == "generate"]
    finals = [l for l in sessions if "--eval-only" in l["cmd"]]
    assert len(finals) == NUM_GPUS and len(sessions) - len(finals) == scored_refreshes * NUM_GPUS


def _check_refreshes(out: Path, arm: str) -> None:
    for r in range(REFRESHES):
        rdir = paths.round_dir(out, arm, r)
        selected = read_jsonl(rdir / "selected" / "selected.jsonl")
        assert len(selected) == 480
        assert all(s["entropy"] is not None and s["mean_reverse_kl"] is not None for s in selected)
        summary = json.loads((rdir / "train" / "summary.json").read_text())
        assert summary["global_step_offset"] == r * 15 and summary["steps"] == 15
        ckpt = paths.checkpoint_dir(out, arm, r)
        assert any(ckpt.glob("*.safetensors")), "keep_all_weights: every refresh keeps its weights"
        assert not (ckpt / "optimizer_state.pt").exists(), "a finished arm keeps no optimizer state"


def _check_questions(out: Path, cfg: DictConfig, arms: tuple[str, ...], exp: Expected) -> None:
    per_arm = {}
    for arm in arms:
        questions = read_jsonl(out / "pool" / f"questions_{arm}.jsonl")
        assert len(questions) == exp.questions_per_refresh * REFRESHES
        source = str(cfg.driver.arms[arm].question_source)
        if source.startswith("bank_bucket:"):
            assert all(q["bucket"] == source.split(":")[1] for q in questions)
        per_arm[arm] = [q["id"] for q in questions]
    distinct = len({tuple(v) for v in per_arm.values()})
    assert distinct == (1 if exp.shares_step0 else len(arms))
    if exp.shares_step0:
        for arm in arms[1:]:
            reused = paths.round_dir(out, arm, 0) / "rollouts" / "reused_from.json"
            assert reused.exists() and _ids(out, json.loads(reused.read_text())["arm"], 0) == _ids(out, arm, 0)
    assert (out / "plots" / "accuracy_vs_steps.png").exists()


# --- tests ------------------------------------------------------------------


def test_budget() -> None:
    for experiment, exp in EXPERIMENTS.items():
        b = budget_from(compose_config([f"+experiment={experiment}"]))
        assert (b.steps_total, b.refresh_every, b.effective_batch, b.refreshes) == (60, 15, 32, REFRESHES)
        assert b.trajectories_per_refresh == 480 and b.questions_per_refresh == exp.questions_per_refresh
    for bad in ("driver.steps_total=50", "selection.k=7", "selection.k=13"):
        try:
            budget_from(compose_config(["+experiment=answer_level", bad]))
        except ValueError:
            continue
        raise AssertionError(f"{bad} must be rejected")


def test_full_runs_and_never_rewrite_config() -> None:
    bank = _shared_bank()
    for experiment, exp in EXPERIMENTS.items():
        out = _tmp(experiment)
        _run(experiment, out, bank)
        cfg = OmegaConf.load(out / "resolved_config.yaml")
        arms = tuple(cfg.driver.arms)
        assert set(arms) == set(exp.arms)
        _check_config(cfg)
        _check_metrics(out, arms, exp)
        for i, arm in enumerate(arms):
            _check_launches(out, arm, i == 0, exp)
            _check_refreshes(out, arm)
        _check_questions(out, cfg, arms, exp)
        # A second invocation is a no-op that keeps the stored config byte for byte.
        before, metrics_before = (out / "resolved_config.yaml").read_bytes(), (out / "metrics.jsonl").read_text()
        _clear_launches(out)
        _run(experiment, out, bank, extra=("driver.steps_total=30",))
        assert (out / "resolved_config.yaml").read_bytes() == before and _launches(out) == []
        assert (out / "metrics.jsonl").read_text() == metrics_before
        shutil.rmtree(out)
    shutil.rmtree(bank)


def _scores(rdir: Path, field: str, stage_dir: str, stem: str) -> dict[tuple[int, int], float]:
    rows = [r for k in range(NUM_GPUS) for r in read_jsonl(rdir / stage_dir / f"{stem}.shard{k}.jsonl")]
    return {(r["example_index"], r["rollout_index"]): r[field] for r in rows}


def test_answer_kl_bands_partition_each_question() -> None:
    """Answer level: kl_high keeps each question's 4 answers with the highest reverse KL."""

    bank, out = _shared_bank(), _tmp("answer")
    _run("answer_level", out, bank)
    rdir = paths.round_dir(out, "kl_high", 2)
    kl = _scores(rdir, "mean_reverse_kl", "oracle", "oracle_kl")
    assert len(kl) == 120 * 12
    by_q: dict[int, list[float]] = {}
    for s in read_jsonl(rdir / "selected" / "selected.jsonl"):
        by_q.setdefault(s["example_index"], []).append(kl[(s["example_index"], s["rollout_index"])])
    for q, kept in by_q.items():
        ranked = sorted((v for (e, _), v in kl.items() if e == q), reverse=True)
        assert sorted(kept, reverse=True) == ranked[:4]
    shutil.rmtree(out)
    shutil.rmtree(bank)


def _top_third(scores: dict[int, float]) -> set[int]:
    ranked = sorted(scores, key=lambda q: -scores[q])
    return set(ranked[: len(ranked) // 3])


def test_question_rules_keep_the_top_third() -> None:
    """Question level: qkl_high keeps the third with the highest mean reverse KL;
    qent1_high the third with the highest entropy of rollout 0, all answers kept."""

    bank = _shared_bank()
    for experiment, arm, field, stage_dir, stem, one_answer in (
        ("question_kl", "qkl_high", "mean_reverse_kl", "oracle", "oracle_kl", False),
        ("question_entropy", "qent1_high", "entropy", "entropy", "entropy", True),
    ):
        out = _tmp(experiment)
        _run(experiment, out, bank)
        rdir = paths.round_dir(out, arm, 1)
        scores = _scores(rdir, field, stage_dir, stem)
        per_q: dict[int, list[float]] = {}
        for (q, i), v in scores.items():
            if not one_answer or i == 0:
                per_q.setdefault(q, []).append(v)
        expected = _top_third({q: sum(v) / len(v) for q, v in per_q.items()})
        selected = read_jsonl(rdir / "selected" / "selected.jsonl")
        assert {s["example_index"] for s in selected} == expected and len(selected) == 480
        shutil.rmtree(out)
    shutil.rmtree(bank)


def _restore_crash_state(out: Path, arm: str, boundary: int) -> None:
    """The disk right after refresh ``boundary - 1`` finished: later refreshes absent,
    the newest optimizer state present."""

    for r in range(boundary, REFRESHES + 1):
        shutil.rmtree(paths.round_dir(out, arm, r), ignore_errors=True)
    if boundary >= 1:
        (paths.checkpoint_dir(out, arm, boundary - 1) / "optimizer_state.pt").write_bytes(b"dry-run optimizer state")
    rows = [r for r in read_jsonl(out / "metrics.jsonl") if not (r["arm"] == arm and r["step"] >= boundary * 15)]
    write_jsonl(out / "metrics.jsonl", rows)


def test_resume_from_every_boundary() -> None:
    bank, out = _shared_bank(), _tmp("resume")
    _run("correctness", out, bank)
    reference = _metrics(out)
    arm = list(OmegaConf.load(out / "resolved_config.yaml").driver.arms)[-1]
    for boundary in range(REFRESHES + 1):
        _restore_crash_state(out, arm, boundary)
        _clear_launches(out)
        _run("correctness", out, bank)
        launches = _launches(out)
        assert launches and all(_flag(l["cmd"], "--arm") == arm for l in launches), "only the unfinished arm runs"
        assert min(int(_flag(l["cmd"], "--round")) for l in launches) == boundary
        trains = [int(_flag(l["cmd"], "--global-step-offset")) for l in launches if _stage(l["cmd"]) == "train"]
        assert trains == list(range(boundary * 15, 60, 15)), (boundary, trains)
        assert _metrics(out) == reference, f"boundary {boundary}: metrics differ after resume"
    shutil.rmtree(out)
    shutil.rmtree(bank)


def test_resume_mid_refresh() -> None:
    """Crash inside refresh 2: only the missing stages are relaunched."""

    bank, out = _shared_bank(), _tmp("mid")
    _run("correctness", out, bank)
    reference = _metrics(out)
    arm = list(OmegaConf.load(out / "resolved_config.yaml").driver.arms)[-1]
    rdir = paths.round_dir(out, arm, 2)
    for remove, expect in ((["selected", "train", "checkpoint"], {"train"}), (["train", "checkpoint"], {"train"}), ([], set())):
        _restore_crash_state(out, arm, 3)
        for sub in remove:
            shutil.rmtree(rdir / sub, ignore_errors=True)
        (paths.checkpoint_dir(out, arm, 1) / "optimizer_state.pt").write_bytes(b"x")
        write_jsonl(out / "metrics.jsonl", [r for r in read_jsonl(out / "metrics.jsonl") if not (r["arm"] == arm and r["step"] >= 30)])
        _clear_launches(out)
        _run("correctness", out, bank)
        at2 = {_stage(l["cmd"]) for l in _launches(out) if int(_flag(l["cmd"], "--round")) == 2}
        assert at2 == expect, (remove, at2)
        assert _metrics(out) == reference
    shutil.rmtree(out)
    shutil.rmtree(bank)


def test_pool_link_guard() -> None:
    """Answers generated under another arm's question file are refused."""

    bank, out = _shared_bank(), _tmp("guard")
    driver = Driver(prepare_config(_compose("correctness", out, bank), out), out)
    driver.materialize_eval_sets()
    questions = {arm: driver.write_questions(arm) for arm in ("both_right", "mixed")}
    driver.point_pool_at("both_right")
    driver.run_rollout_eval("mixed", 0, eval_only=False)  # stage ran under the wrong link
    try:
        driver.verify_rollouts("mixed", 0, questions["mixed"])
    except RuntimeError as e:
        assert "another arm" in str(e)
    else:
        raise AssertionError("answers to another arm's questions must be refused")
    driver.run_rollout_eval("both_right", 0, eval_only=False)
    assert len(driver.verify_rollouts("both_right", 0, questions["both_right"])) == 480
    shutil.rmtree(out)
    shutil.rmtree(bank)


def test_plot_cli() -> None:
    bank, out = _shared_bank(), _tmp("plot")
    _run("question_entropy", out, bank)
    (out / "plots" / "accuracy_vs_steps.png").unlink()
    subprocess.run([sys.executable, "-m", "apod.plotting", "--run-dir", str(out), "--band-points", "7"], check=True, cwd=ROOT)
    assert (out / "plots" / "accuracy_vs_steps.png").exists()
    shutil.rmtree(out)
    shutil.rmtree(bank)


def test_exclude_questions_from() -> None:
    """A later seed drops every question an earlier run used, even when the bank has
    new bucket members inserted among the old ones (widened bank)."""

    bank = _tmp("excl-bank")
    rows = _write_bank(bank, per_bucket=40)
    bucket = "both_right"
    members = [r for r in rows if r["bucket"] == bucket]
    first = select_questions(f"bank_bucket:{bucket}", n=10, seed=0, bank_dir=bank, dry_run=True)
    run_dir = _tmp("excl-run")
    write_jsonl(run_dir / "pool" / f"questions_{bucket}.jsonl", [{"id": q["id"], "example_index": i} for i, q in enumerate(first)])
    write_jsonl(bank / "questions.jsonl", [{**members[0], "id": "new-0", "example_index": -1}, *rows])
    used = used_question_ids([run_dir])
    assert used == {q["id"] for q in first}
    second = select_questions(f"bank_bucket:{bucket}", n=10, seed=0, bank_dir=bank, dry_run=True, exclude_ids=used)
    assert not {q["id"] for q in second} & used
    assert [q["id"] for q in second] == ["new-0"] + [r["id"] for r in members[10:19]]
    try:
        used_question_ids([bank / "missing"])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a run dir without pool files must raise")
    shutil.rmtree(bank)
    shutil.rmtree(run_dir)


def test_selection_self_test() -> None:
    subprocess.run([sys.executable, str(ROOT / "apod" / "selection.py")], check=True, cwd=ROOT)
