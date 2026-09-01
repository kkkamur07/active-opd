"""Refinement pass for the kl50w peak LR (USER 2026-09-01: "lets do the lr
sweep between 1e-6 to 1e-5 ... maybe do 5 trials").

The wide per-arm sweep (lr_sweep.py) found the same shape for every arm:
minimum at 5e-6, shallow toward 1e-5, rising toward 1e-6, unstable at 2e-5+.
Since the arms agree, this refinement runs on ONE arm (kl_mid, the winner)
as a plain 5-point log-grid over [1e-6, 1e-5] -- a grid, not TPE, because
in a narrow range the grid maps the valley's curvature directly. Reuses
run_trial from lr_sweep.py (same 8-step protocol, warmup(1)+cosine, early
divergence stop). Points already measured by the wide sweep within 3% are
skipped and their objectives reused.

Writes lr_refine.json and, if the refined winner differs from lr_probe.json's
chosen_lr, updates chosen_lr in place (the kl50w driver re-applies it on
launch). Resumable: completed trials are read back from lr_refine.json.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import optuna

from lr_sweep import RUN_DIR, run_trial, log

ARM = "kl_mid"
# 5 log-spaced points (USER: "refine one arm and maybe do 5 trials"):
# 1e-6, 1.78e-6, 3.16e-6, 5.62e-6, 1e-5. The 1e-6 and 1e-5 endpoints reuse
# wide-sweep measurements, so only ~3 fresh trials run.
GRID = [10 ** (-6 + i / 4) for i in range(5)]


def main() -> None:
    out_path = RUN_DIR / "lr_refine.json"
    results: dict[str, dict] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())["results"]

    # Reuse wide-sweep measurements for grid points within 3% (relative),
    # read from the sweep's optuna study (the sweep was stopped before it
    # wrote its verdict, so lr_probe.json carries only the 3-point probe).
    probe = json.loads((RUN_DIR / "lr_probe.json").read_text())
    study = optuna.load_study(study_name=f"kl50w_lr_{ARM}",
                              storage=f"sqlite:///{RUN_DIR / 'lr_sweep.db'}")
    prior = {t.params["lr"]: t.value for t in study.trials
             if t.state == optuna.trial.TrialState.COMPLETE and t.value < 999}

    for lr in GRID:
        key = f"{lr:.4e}"
        if key in results:
            continue
        reused = next((v for plr, v in prior.items()
                       if abs(math.log(plr / lr)) < math.log(1.03)), None)
        if reused is not None:
            results[key] = {"objective": reused, "reason": "reused-wide-sweep"}
            log(f"{ARM} lr {lr:.3e}: reused wide-sweep objective {reused:.5f}")
        else:
            r = run_trial(ARM, lr)
            results[key] = {"objective": r["objective"], "reason": r["reason"],
                            "losses": r["losses"]}
        out_path.write_text(json.dumps(
            {"arm": ARM, "grid": GRID, "results": results}) + "\n")

    ok = {k: v for k, v in results.items() if v["objective"] < 999}
    best_key = min(ok, key=lambda k: ok[k]["objective"])
    best_lr, best_obj = float(best_key), ok[best_key]["objective"]
    # Near-ties (within 2% relative) go to the LOWER lr, same rule as before.
    for k in sorted(ok, key=lambda k: float(k)):
        if ok[k]["objective"] <= best_obj * 1.02:
            best_lr, best_obj = float(k), ok[k]["objective"]
            break
    out_path.write_text(json.dumps(
        {"arm": ARM, "grid": GRID, "results": results,
         "chosen_lr": best_lr,
         "note": f"refine grid {datetime.now():%Y-%m-%d %H:%M} UTC over "
                 "[1e-6, 1e-5]; near-ties within 2% go to the lower lr"}) + "\n")
    if abs(math.log(best_lr / probe["chosen_lr"])) > 1e-9:
        probe["chosen_lr"] = best_lr
        probe["refined_by"] = "lr_refine.json"
        (RUN_DIR / "lr_probe.json").write_text(json.dumps(probe) + "\n")
        log(f"lr_probe.json chosen_lr updated to {best_lr:.3e}")
    log(f"REFINE DONE: best lr {best_lr:.3e} (objective {best_obj:.5f})")


if __name__ == "__main__":
    main()
