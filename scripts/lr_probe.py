"""Two-cell probe of the universal round-2 regression (USER 2026-08-17:
"run the lr probe ... we killed the engine and re-built it that could be a
problem"; cell choice "same and low").

The regression under test: kl_mid's round-1 train (16 steps on its re-scored
round-1 selection) took the best checkpoint of the experiment, 0.5815, down
to 0.5010 (t=-7.30) -- and every other arm regressed in the same round.

Both cells retrain that exact train from the SAME round-0 checkpoint on the
SAME banked round-1 selection, then terminal-eval (500x4, cap 16384):
  - cell "lr2e5" (LR 2e-5, identical to the original): reproducibility /
    restart-artifact control. ~0.50 -> the original r1 train was not
    anomalous; ~0.58 -> nondeterminism or the driver kill-rebuild mattered.
  - cell "lr5e6" (LR 5e-6, only the LR changed): overshoot test. ~0.58 with
    lr2e5 at ~0.50 -> LR overshoot confirmed, later rounds need decay;
    both ~0.50 -> a second identical pass is inherently harmful here.

Run when GPUs are free (after the all-arm append):
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
ARM = "kl_mid"
CELLS = (("lr2e5", 2.0e-5), ("lr5e6", 5.0e-6))


def link(target: Path, dst: Path) -> None:
    if not dst.exists():
        dst.symlink_to(target)


def run_cell(name: str, lr: float) -> None:
    probe = ROOT / "outputs/runs" / f"oracle16k_{name}"
    cfg = OmegaConf.load(SRC / "resolved_config.yaml")
    assert float(cfg.train.learning_rate) == 2.0e-5, "unexpected base LR"
    cfg.train.learning_rate = lr
    probe.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=probe / "resolved_config.yaml", resolve=True)
    link(SRC / "pool", probe / "pool")

    src_rounds = SRC / "arms" / ARM / "rounds"
    r0 = probe / "arms" / ARM / "rounds" / "round_00"
    r1 = probe / "arms" / ARM / "rounds" / "round_01"
    r0.mkdir(parents=True, exist_ok=True)
    r1.mkdir(parents=True, exist_ok=True)
    link(src_rounds / "round_00" / "checkpoint", r0 / "checkpoint")
    link(src_rounds / "round_01" / "rollouts", r1 / "rollouts")
    link(src_rounds / "round_01" / "selected", r1 / "selected")

    if not (r1 / "checkpoint" / "config.json").exists():
        print(f"[probe] {name}: train r1 at LR {lr}", flush=True)
        subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=2", "-m", "apod.stages.train",
                "--run-dir", str(probe), "--arm", ARM, "--round", "1",
            ],
            check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"}, cwd=ROOT,
        )

    if not all((probe / "arms" / ARM / "rounds" / "round_02" / "eval" / f"done.shard{k}").exists() for k in (0, 1)):
        print(f"[probe] {name}: terminal eval", flush=True)
        procs = []
        for shard in (0, 1):
            procs.append(subprocess.Popen(
                [
                    sys.executable, "-m", "apod.stages.rollout_eval",
                    "--run-dir", str(probe), "--arm", ARM, "--round", "2",
                    "--shard", str(shard), "--num-shards", "2", "--eval-only",
                ],
                env={**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": str(shard)},
                cwd=ROOT,
            ))
        codes = [p.wait() for p in procs]
        if any(codes):
            raise RuntimeError(f"{name} eval shards exited {codes}")

    print(f"[probe] {name} table:", flush=True)
    subprocess.run(
        [sys.executable, "scripts/eval_table.py", "--run-dir", str(probe)],
        check=True, cwd=ROOT,
    )


def main() -> None:
    for name, lr in CELLS:
        run_cell(name, lr)
    print("[probe] originals for comparison: r1 (r0-trained) 0.5815, "
          "r2 (r1-trained, LR 2e-5) 0.5010", flush=True)


if __name__ == "__main__":
    main()
