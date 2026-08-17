"""LR probe for the universal round-2 regression (USER 2026-08-17: "run the
lr probe as well because we killed the engine and re-built it that could be
a problem").

Retrains kl_mid's round 1 -- the biggest surviving-arm regression (0.5815 ->
0.5010, t=-7.30) -- from the SAME round-0 checkpoint on the SAME round-1
selection with the ONLY change being LR 2e-5 -> 5e-6, then terminal-evals.
Readout against the originals:
  - probe ~= 0.58 (r1 level): the regression was LR overshoot; later rounds
    need a lower LR, the data was fine.
  - probe ~= 0.50 (original r2): regression is inherent to a second identical
    pass on this data (or the driver kill-rebuild mattered); LR exonerated.
Everything (checkpoint, rollouts, selection, pool) is symlinked from the main
oracle16k run, so the probe adds one 16-step train + one 500x4 eval (~1.5 h).

Run AFTER the all-arm append finishes (GPUs must be free):
  uv run python scripts/lr_probe.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/runs/oracle16k"
PROBE = ROOT / "outputs/runs/oracle16k_lr5e6"
ARM = "kl_mid"
LR = 5.0e-6


def link(target: Path, dst: Path) -> None:
    if not dst.exists():
        dst.symlink_to(target)


def main() -> None:
    cfg = OmegaConf.load(SRC / "resolved_config.yaml")
    assert float(cfg.train.learning_rate) == 2.0e-5, "unexpected base LR"
    cfg.train.learning_rate = LR
    PROBE.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=PROBE / "resolved_config.yaml", resolve=True)
    link(SRC / "pool", PROBE / "pool")

    src_rounds = SRC / "arms" / ARM / "rounds"
    r0 = PROBE / "arms" / ARM / "rounds" / "round_00"
    r1 = PROBE / "arms" / ARM / "rounds" / "round_01"
    r0.mkdir(parents=True, exist_ok=True)
    r1.mkdir(parents=True, exist_ok=True)
    link(src_rounds / "round_00" / "checkpoint", r0 / "checkpoint")
    link(src_rounds / "round_01" / "rollouts", r1 / "rollouts")
    link(src_rounds / "round_01" / "selected", r1 / "selected")

    if not (r1 / "checkpoint" / "config.json").exists():
        subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=2", "-m", "apod.stages.train",
                "--run-dir", str(PROBE), "--arm", ARM, "--round", "1",
            ],
            check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"}, cwd=ROOT,
        )

    procs = []
    for shard in (0, 1):
        procs.append(subprocess.Popen(
            [
                sys.executable, "-m", "apod.stages.rollout_eval",
                "--run-dir", str(PROBE), "--arm", ARM, "--round", "2",
                "--shard", str(shard), "--num-shards", "2", "--eval-only",
            ],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": str(shard)},
            cwd=ROOT,
        ))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"probe eval shards exited {codes}")

    subprocess.run(
        [sys.executable, "scripts/eval_table.py", "--run-dir", str(PROBE)],
        check=True, cwd=ROOT,
    )


if __name__ == "__main__":
    main()
