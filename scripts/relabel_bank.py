"""Relabel a question bank's teacher-labelled questions with a new student.

The correctness buckets (teacher_right_student_wrong, both_right, both_wrong,
mixed) depend on the student. For a new student (Qwen3.5-4B) we keep the
existing teacher sweep (same 9B teacher, same cap 8192, same sampling) and
re-run only the student sweep, and only on questions the teacher already
labelled, so every relabelled question gets a bucket.

Layout: <dst>/pool (copied), <dst>/teacher -> <src>/teacher (symlink),
<dst>/student (new, 4 rollouts per question at the bank cap),
<dst>/questions.jsonl (rebuilt by apod.bank.label_questions; question_entropy
stays None: no H(q) arms use this bank).

One vLLM session per GPU over ALL pending questions (no per-chunk engine
reload). Seeds: the bank's RolloutWriter with round_index=0, per position.

  python scripts/relabel_bank.py --src outputs/runs/bank-8k --dst outputs/runs/bank-8k-4b \
      --student Qwen/Qwen3.5-4B --gpus 0,1          # launches one worker per GPU
  python scripts/relabel_bank.py ... --worker --shard K --num-shards N  # (internal)
  python scripts/relabel_bank.py ... --label        # rebuild questions.jsonl + print buckets
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apod import bank  # noqa: E402


def prepare(src: Path, dst: Path, student: str) -> list[int]:
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "pool").exists():
        shutil.copytree(src / "pool", dst / "pool")
    if not (dst / "teacher").exists():
        (dst / "teacher").symlink_to((src / "teacher").resolve())
    path = dst / "resolved_config.yaml"
    if not path.exists():
        cfg = OmegaConf.load(src / "resolved_config.yaml")
        cfg.model.student_id = student
        cfg.bank.name = dst.name
        cfg.bank.dir = str(dst.resolve())
        cfg.run_name = dst.name
        cfg.output_dir = str(dst.resolve())
        cfg.engine.target_concurrent_sequences = 256
        cfg.engine.chunk_sequences = 4096   # one long generate stream (no per-chunk drain)
        OmegaConf.save(config=cfg, f=path, resolve=True)
    rows = bank.load_bank(src)
    indices = sorted(int(r["example_index"]) for r in rows if r["teacher_label"] is not None)
    (dst / "relabel_indices.json").write_text(json.dumps({"src": str(src), "student": student, "example_indices": indices}))
    return indices


def worker(dst: Path, shard: int, num_shards: int) -> None:
    cfg = OmegaConf.load(dst / "resolved_config.yaml")
    indices = json.loads((dst / "relabel_indices.json").read_text())["example_indices"]
    # Every relabel question as one "chunk 0" so the whole shard is one engine session.
    bank.chunk_indices = lambda bank_dir, cfg, model, chunk, pool_size: list(indices)
    bank.run_sweep(dst, cfg, "student", 0, shard, num_shards)


def label(dst: Path) -> None:
    cfg = OmegaConf.load(dst / "resolved_config.yaml")
    rows = bank.label_questions(dst, cfg)
    print(f"{dst}: {len(rows)} questions relabelled with {cfg.model.student_id}")
    print("student labels", dict(Counter(r["student_label"] for r in rows)))
    print("buckets", dict(bank.bucket_counts(rows)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--student", default="Qwen/Qwen3.5-4B")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--label", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args()
    if a.worker:
        return worker(a.dst, a.shard, a.num_shards)
    if a.label:
        return label(a.dst)
    indices = prepare(a.src, a.dst, a.student)
    gpus = [g.strip() for g in a.gpus.split(",")]
    print(f"relabel {len(indices)} questions with {a.student} on GPUs {gpus}", flush=True)
    procs = []
    for k, g in enumerate(gpus):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": g, "HF_HUB_OFFLINE": "1"}
        cmd = [sys.executable, __file__, "--src", str(a.src), "--dst", str(a.dst), "--worker",
               "--shard", str(k), "--num-shards", str(len(gpus))]
        procs.append(subprocess.Popen(cmd, env=env, cwd=REPO_ROOT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"relabel workers exited {codes}")
    label(a.dst)


if __name__ == "__main__":
    main()
