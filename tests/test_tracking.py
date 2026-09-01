"""``apod.tracking``: no-op behaviour without wandb / with mode=disabled,
deterministic run ids, and -- when wandb is importable -- an offline run
that two processes' worth of init/log/finish can share.

Runs under pytest or as ``python -m tests.test_tracking``.
"""

from __future__ import annotations

import builtins
import importlib
import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from apod import tracking

CFG = OmegaConf.create(
    {
        "sampling": {"max_new_tokens": 8192},
        "tracking": {"mode": "offline", "project": "apod-test", "entity": None, "experiment": None, "dir": None},
    }
)


def _fresh():
    tracking.finish()
    return importlib.reload(tracking)


def test_run_id_is_deterministic_and_arm_specific():
    a = tracking.run_id("/x/outputs/runs/r1-correctness-8k", "teacher_right_student_wrong")
    b = tracking.run_id("/other/place/r1-correctness-8k", "teacher_right_student_wrong")
    c = tracking.run_id("/x/outputs/runs/r1-correctness-8k", "mixed")
    assert a == b and a != c
    assert a.startswith("r1-correctness-8k-teacher_right_student_wrong-")
    assert len(a) <= 64


def test_disabled_mode_is_a_noop():
    t = _fresh()
    cfg = OmegaConf.merge(CFG, {"tracking": {"mode": "disabled"}})
    assert t.init(cfg, "/tmp/run-x", "arm") is False
    t.log_step(3, {"loss": 1.0})
    t.log_refresh(10, {"eval/math500/strict_avg4": 0.4})
    t.finish()


def test_missing_wandb_is_a_noop():
    t = _fresh()
    real_import = builtins.__import__

    def no_wandb(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb."):
            raise ImportError("simulated: wandb not installed")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = no_wandb
    try:
        assert t.init(CFG, "/tmp/run-y", "arm") is False
        t.log_step(1, {"loss": 0.5})
        t.finish()
    finally:
        builtins.__import__ = real_import


def test_non_zero_rank_is_a_noop():
    t = _fresh()
    os.environ["RANK"] = "1"
    try:
        assert t.init(CFG, "/tmp/run-z", "arm") is False
    finally:
        del os.environ["RANK"]


def test_bad_mode_is_rejected():
    t = _fresh()
    cfg = OmegaConf.merge(CFG, {"tracking": {"mode": "sometimes"}})
    try:
        t.init(cfg, "/tmp/run-w", "arm")
    except ValueError:
        return
    raise AssertionError("tracking.mode='sometimes' was accepted")


def test_offline_run_when_wandb_is_installed():
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("wandb not installed: offline run test skipped")
        return
    t = _fresh()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "r1-correctness-8k"
        run_dir.mkdir()
        # Two processes' worth: the driver logs a refresh, then a train stage
        # logs steps, both resuming the same deterministic id.
        assert t.init(CFG, run_dir, "mixed") is True
        t.log_refresh(0, {"eval/math500/strict_avg4": 0.41, "eval/aime/avg16": 0.1})
        t.finish()
        assert t.init(CFG, run_dir, "mixed") is True
        for step in range(1, 4):
            t.log_step(step, {"loss": 1.0 / step, "overlap_ratio_top16": 0.5, "response_tokens": 100})
        t.finish()
        offline_runs = sorted((run_dir / "wandb").glob("offline-run-*"))
        assert len(offline_runs) == 2, offline_runs
        assert all(t.run_id(run_dir, "mixed") in p.name for p in offline_runs), offline_runs


if __name__ == "__main__":
    test_run_id_is_deterministic_and_arm_specific()
    test_disabled_mode_is_a_noop()
    test_missing_wandb_is_a_noop()
    test_non_zero_rank_is_a_noop()
    test_bad_mode_is_rejected()
    test_offline_run_when_wandb_is_installed()
    print("tests/test_tracking.py passed")
