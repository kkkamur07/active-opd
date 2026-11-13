"""Training memory/speed probe for a new student (2026-11-13: Qwen3.5-4B + 9B teacher).

Synthetic WORST CASE batch: every rollout is a full cap-length response
(8192 tokens, truncated, as ~88% of real 8k rollouts are) behind a 200-token
prompt. For each micro-batch candidate, runs the real train stage (torchrun,
2 ranks, GKD reverse KL, fused kernels, per-step diagnostics on) for STEPS
optimizer steps at effective batch 32, samples nvidia-smi peak memory, and
records seconds per step from the trainer's own runtime.

  python scripts/train_probe.py --experiment p4b_rollout --micro 8 4 2 --steps 6
Writes outputs/runs/train-probe-4b/probe.json; prints a table.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from apod import paths  # noqa: E402
from apod.datasets.io import write_jsonl  # noqa: E402
from apod.driver import compose_config  # noqa: E402

PROMPT, RESPONSE, K = 200, 8192, 4


def make_run(root: Path, experiment: str, micro: int, steps: int) -> Path:
    run = root / f"micro{micro}"
    if run.exists():
        shutil.rmtree(run)
    cfg = compose_config([f"+experiment={experiment}", "tracking.mode=disabled"])
    world = 2
    cfg.train.per_device_train_batch_size = micro
    cfg.train.gradient_accumulation_steps = int(cfg.train.effective_batch) // (world * micro)
    cfg.train.persist_optimizer = False
    cfg.output_dir = str(run)
    cfg.run_name = run.name
    run.mkdir(parents=True)
    OmegaConf.save(config=cfg, f=run / "resolved_config.yaml", resolve=True)
    rdir = paths.round_dir(run, "probe", 0)
    tok = rdir / "rollouts" / "tokens"
    tok.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n_q = steps * int(cfg.train.effective_batch) // K
    selected = []
    for e in range(n_q):
        ids = rng.integers(1000, 150000, size=(K, PROMPT + RESPONSE), dtype=np.int64)
        np.savez(tok / f"example_{e:05d}.npz", input_ids=ids, prompt_length=PROMPT,
                 response_lengths=np.full(K, RESPONSE), truncated=np.ones(K, dtype=bool))
        for j in range(K):
            selected.append({"example_index": e, "rollout_index": j, "entropy": None, "mean_reverse_kl": None,
                             "correct": False, "truncated": True, "response_length": RESPONSE})
    write_jsonl(rdir / "selected" / "selected.jsonl", selected)
    return run


def gpu_peak(stop: threading.Event, out: list[int]) -> None:
    while not stop.is_set():
        try:
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout.split()
            out.append(max(int(x) for x in used))
        except Exception:
            pass
        time.sleep(2)


def probe(run: Path) -> dict:
    stop, peaks = threading.Event(), []
    th = threading.Thread(target=gpu_peak, args=(stop, peaks), daemon=True)
    th.start()
    t0 = time.time()
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
           "-m", "apod.stages.train", "--run-dir", str(run), "--arm", "probe", "--round", "0"]
    log = run / "train.log"
    with log.open("w") as f:
        code = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ, "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled"}).returncode
    stop.set()
    th.join()
    hist_path = paths.round_dir(run, "probe", 0) / "train" / "log_history.jsonl"
    rt = None
    if hist_path.exists():
        for line in hist_path.read_text().splitlines():
            row = json.loads(line)
            if "train_runtime" in row:
                rt = row
    oom = "out of memory" in log.read_text().lower()
    return {"exit": code, "oom": oom, "peak_mib": max(peaks or [0]), "wall_s": round(time.time() - t0, 1),
            "train_runtime_s": rt and rt.get("train_runtime"), "steps_per_s": rt and rt.get("train_steps_per_second")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="p4b_rollout")
    p.add_argument("--micro", type=int, nargs="+", default=[8, 4, 2])
    p.add_argument("--steps", type=int, default=6)
    a = p.parse_args()
    root = REPO_ROOT / "outputs" / "runs" / "train-probe-4b"
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    for m in a.micro:
        run = make_run(root, a.experiment, m, a.steps)
        r = probe(run)
        r["s_per_step"] = round(r["train_runtime_s"] / a.steps, 1) if r["train_runtime_s"] else None
        results[m] = r
        print(f"micro {m}: {r}", flush=True)
        shutil.rmtree(run / "arms" / "probe" / "rounds" / "round_00" / "checkpoint", ignore_errors=True)
        (root / "probe.json").write_text(json.dumps(results, indent=2))
    ok = {m: r for m, r in results.items() if r["exit"] == 0 and r["peak_mib"] < 75000}
    best = min(ok, key=lambda m: ok[m]["s_per_step"] or 1e9) if ok else None
    print(f"fits under 75,000 MiB: {sorted(ok)}; fastest: micro {best}", flush=True)
    (root / "probe.json").write_text(json.dumps({"results": results, "chosen_micro": best}, indent=2))


if __name__ == "__main__":
    main()
