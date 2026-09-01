"""Peak-LR sweep for kl50w, PER ARM (USER 2026-09-01: "proper lr probe from
1e-6 to 1e-3 i would say or maybe 1e-2 just run this with hydra optuna for i
would say 5-10 steps each and 10 experiments and (with early stopping)" +
"we need to do this for all right ? kl high to kl low and all" -- "this lr
sweep would help us understand what do selections prefer").

Per arm: optuna TPE over log-uniform [1e-6, 1e-2], 6 enqueued log-spaced
anchors then 4 model-suggested trials. Each trial trains the arm's REAL
round-0 selection under the run's actual warmup(1)+cosine schedule for up to
MAX_STEPS optimizer steps, killed early on divergence (non-finite loss, or
post-warmup loss/grad blowup). Objective: mean of the last 3 step losses.

The RUN still uses ONE common LR for all arms (per-arm LRs would confound
the selection comparison): the anchor with the lowest mean objective across
arms. Per-arm winners are recorded as the "what do selections prefer"
analysis. Verdict lands in outputs/runs/kl50w/lr_probe.json, which the kl50w
driver re-applies on relaunch. Storage is SQLite so an interrupted sweep
resumes (completed trials are never re-run).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import optuna
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "outputs/runs/kl50w"
SWEEP_DIR = RUN_DIR.parent / "kl50w_lrsweep"
ARMS = ("kl_high", "kl_mid", "kl_low", "random")

MAX_STEPS = 8          # 1 warmup + 7 near the cosine peak
TRIAL_TIMEOUT = 1500   # hard wall per trial (load ~4 min + 8 steps ~12 min)
N_TRIALS = 10          # per arm
ANCHORS = (1e-6, 5e-6, 2e-5, 1e-4, 5e-4, 2e-3)
DIVERGED = 1e3         # objective for killed/failed trials (TPE only ranks)

LOSS_RE = re.compile(r"'loss': '?([0-9.eE+-]+)")
GRAD_RE = re.compile(r"'grad_norm': '?([0-9.eE+-]+)")


def log(msg: str) -> None:
    print(f"[lr-sweep {datetime.now():%H:%M:%S}] {msg}", flush=True)


def parse(text: str) -> tuple[list[float], list[float]]:
    return ([float(x) for x in LOSS_RE.findall(text)],
            [float(x) for x in GRAD_RE.findall(text)])


def diverged(losses: list[float], grads: list[float]) -> bool:
    if any(not math.isfinite(x) for x in [*losses, *grads]):
        return True
    if len(losses) <= 2:  # warmup + first full-LR step: no verdict yet
        return False
    baseline = max(losses[:2])
    return losses[-1] > 3 * baseline or grads[-1] > 10


def run_trial(arm: str, lr: float) -> dict:
    if SWEEP_DIR.exists():
        shutil.rmtree(SWEEP_DIR)
    cfg = OmegaConf.load(RUN_DIR / "resolved_config.yaml")
    cfg.train.learning_rate = lr
    cfg.train.persist_optimizer = False
    SWEEP_DIR.mkdir(parents=True)
    OmegaConf.save(config=cfg, f=SWEEP_DIR / "resolved_config.yaml", resolve=True)
    (SWEEP_DIR / "pool").symlink_to(RUN_DIR / "pool")
    src = RUN_DIR / "arms" / arm / "rounds" / "round_00"
    rdir = SWEEP_DIR / "arms" / arm / "rounds" / "round_00"
    rdir.mkdir(parents=True)
    (rdir / "rollouts").symlink_to(src / "rollouts")
    (rdir / "selected").symlink_to(src / "selected")
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", "-m", "apod.stages.train",
        "--run-dir", str(SWEEP_DIR), "--arm", arm, "--round", "0",
    ]
    plog = RUN_DIR / f"lr_sweep_{arm}_{lr:.3e}.log"
    reason = "max_steps"
    with plog.open("w") as f:
        proc = subprocess.Popen(cmd, env={**os.environ, "HF_HUB_OFFLINE": "1"},
                                stdout=f, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + TRIAL_TIMEOUT
        while proc.poll() is None:
            time.sleep(10)
            losses, grads = parse(plog.read_text())
            if len(losses) >= MAX_STEPS:
                break
            if diverged(losses, grads):
                reason = "diverged"
                break
            if time.monotonic() > deadline:
                reason = "timeout"
                break
        if proc.poll() is not None and proc.returncode != 0:
            reason = "crashed"
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    time.sleep(10)  # CUDA teardown before the next trial
    losses, grads = parse(plog.read_text())
    ok = reason == "max_steps" and len(losses) >= 5
    objective = sum(losses[-3:]) / 3 if ok else DIVERGED
    log(f"{arm} lr {lr:.3e}: {reason}, {len(losses)} steps, losses "
        f"{[round(x, 4) for x in losses]}, grad tail "
        f"{[round(x, 2) for x in grads[-3:]]}, objective {objective:.5f}")
    return {"reason": reason, "objective": objective,
            "losses": losses, "grad_norms": grads}


def sweep_arm(arm: str) -> optuna.Study:
    study = optuna.create_study(
        study_name=f"kl50w_lr_{arm}",
        storage=f"sqlite:///{RUN_DIR / 'lr_sweep.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=len(ANCHORS)),
        direction="minimize",
    )
    done = {t.params["lr"] for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE}
    for lr in ANCHORS:
        if lr not in done:
            study.enqueue_trial({"lr": lr})

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        result = run_trial(arm, lr)
        trial.set_user_attr("reason", result["reason"])
        trial.set_user_attr("losses", result["losses"])
        trial.set_user_attr("grad_norms", result["grad_norms"])
        return result["objective"]

    remaining = N_TRIALS - len([t for t in study.trials
                                if t.state == optuna.trial.TrialState.COMPLETE])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    log(f"{arm}: best lr {study.best_trial.params['lr']:.3e} "
        f"(objective {study.best_value:.5f})")
    return study


def main() -> None:
    studies = {arm: sweep_arm(arm) for arm in ARMS}

    # Common run LR: the anchor with the best mean objective across arms
    # (anchors are the only LRs every arm evaluated, so means are comparable).
    def anchor_obj(study: optuna.Study, lr: float) -> float:
        vals = [t.value for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.params["lr"] == lr]
        return min(vals) if vals else DIVERGED

    anchor_means = {
        lr: sum(anchor_obj(s, lr) for s in studies.values()) / len(studies)
        for lr in ANCHORS
    }
    chosen = min(anchor_means, key=anchor_means.get)
    verdict = {
        "chosen_lr": chosen,
        "note": f"per-arm optuna sweep {datetime.now():%Y-%m-%d %H:%M} UTC: "
                f"{N_TRIALS} trials/arm over [1e-6, 1e-2], objective "
                f"mean(last-3 step losses) at {MAX_STEPS} steps under "
                "warmup(1)+cosine; run LR = anchor with best cross-arm mean "
                "(one common LR keeps the arm comparison unconfounded)",
        "anchor_means": {f"{lr:g}": v for lr, v in anchor_means.items()},
        "per_arm": {
            arm: {
                "best_lr": s.best_trial.params["lr"],
                "best_objective": s.best_value,
                "trials": [
                    {"lr": t.params["lr"], "value": t.value,
                     "reason": t.user_attrs.get("reason", "?")}
                    for t in s.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ],
            }
            for arm, s in studies.items()
        },
    }
    (RUN_DIR / "lr_probe.json").write_text(json.dumps(verdict) + "\n")
    if SWEEP_DIR.exists():
        shutil.rmtree(SWEEP_DIR)
    per_arm = ", ".join(f"{a} {v['best_lr']:.1e}" for a, v in verdict["per_arm"].items())
    log(f"SWEEP DONE: common lr {chosen:.3e}; per-arm preferences: {per_arm}")


if __name__ == "__main__":
    main()
