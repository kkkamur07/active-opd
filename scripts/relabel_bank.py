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


def prepare(src: Path, dst: Path, student: str, limit: int | None = None) -> list[int]:
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
        cfg.engine.target_concurrent_sequences = 192  # 1.76M-token KV per GPU: 192 x (8.4k + 0.5k state page) fits worst case at the 8k cap
        cfg.engine.chunk_sequences = 4096   # one long generate stream (no per-chunk drain)
        OmegaConf.save(config=cfg, f=path, resolve=True)
    rows = bank.load_bank(src)
    labelled = [r for r in rows if r["teacher_label"] is not None]
    if limit is None:
        indices = sorted(int(r["example_index"]) for r in labelled)
    else:
        # Every teacher-CORRECT question (the only source of the rarest bucket,
        # teacher_right_student_wrong, and of both_right) plus the first
        # `limit` other teacher-labelled questions in pool order (both_wrong
        # and mixed fill from far fewer). Raise `limit` later if a bucket is
        # short: resume keeps every finished question.
        c = [int(r["example_index"]) for r in labelled if r["teacher_label"] == "C"]
        other = [int(r["example_index"]) for r in labelled if r["teacher_label"] != "C"][:limit]
        indices = sorted(c + other)
    (dst / "relabel_indices.json").write_text(json.dumps({"src": str(src), "student": student, "example_indices": indices}))
    return indices


def worker(dst: Path, shard: int, num_shards: int, model: str = "student", plan: str = "relabel_indices.json",
           chunk: int = 0) -> None:
    cfg = OmegaConf.load(dst / "resolved_config.yaml")
    indices = json.loads((dst / plan).read_text())["example_indices"]
    # Every planned question as one "chunk" so the whole shard is one engine
    # session; the chunk number sets the seed base (distinct per pass).
    bank.chunk_indices = lambda bank_dir, cfg, model, chunk, pool_size: list(indices)
    bank.run_sweep(dst, cfg, model, chunk, shard, num_shards)


def launch(a, gpus: list[str], model: str, plan: str, chunk: int) -> None:
    procs = []
    for k, g in enumerate(gpus):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": g, "HF_HUB_OFFLINE": "1"}
        cmd = [sys.executable, __file__, "--src", str(a.src), "--dst", str(a.dst), "--worker",
               "--model", model, "--plan", plan, "--chunk", str(chunk),
               "--shard", str(k), "--num-shards", str(len(gpus))]
        procs.append(subprocess.Popen(cmd, env=env, cwd=REPO_ROOT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"{model} workers exited {codes}")


def widen(a, gpus: list[str]) -> None:
    """Grow the 4B bank (2026-11-13: only 187 teacher_right_student_wrong).

    1. student (4B) sweep on the first `--widen` pool questions the source bank
       never teacher-labelled (pool positions < 14,500, so never otval200),
       skipping any the 4B already swept;
    2. teacher (9B) sweep ONLY on those the 4B labels W: the only questions
       that can land in teacher_right_student_wrong (or both_wrong);
    3. relabel. Teacher rows go to <dst>/teacher, a real copy of the source
       bank's teacher dir (the source bank is never written).
    """

    t = a.dst / "teacher"
    if t.is_symlink():
        target = t.resolve()
        t.unlink()
        shutil.copytree(target, t)
        print(f"teacher dir copied from {target} (no longer a symlink)", flush=True)
    src_rows = bank.load_bank(a.src)
    labelled = {int(r["example_index"]) for r in src_rows if r["teacher_label"] is not None}
    cfg = OmegaConf.load(a.dst / "resolved_config.yaml")
    n = int(cfg.bank.num_rollouts)
    swept = set(bank.complete_groups(a.dst / "student", n, bank.ROLLOUT_FIELDS))
    plan = a.dst / "widen_student.json"
    if not plan.exists():
        cand = [i for i in range(int(cfg.bank.student_questions)) if i not in labelled and i not in swept][: a.widen]
        plan.write_text(json.dumps({"example_indices": cand}))
    cand = json.loads(plan.read_text())["example_indices"]
    print(f"widen: 4B sweep of {len(cand)} new questions", flush=True)
    launch(a, gpus, "student", plan.name, chunk=30)
    student = bank.complete_groups(a.dst / "student", n, bank.ROLLOUT_FIELDS)
    th = {"correct_min": int(cfg.bank.correct_min), "wrong_max": int(cfg.bank.wrong_max)}
    wrong = [i for i in cand if i in student and bank.label([bank.strict_correct(r) for r in student[i]], **th) == "W"]
    tplan = a.dst / "widen_teacher.json"
    tplan.write_text(json.dumps({"example_indices": wrong}))
    print(f"widen: 9B teacher sweep of the {len(wrong)} questions the 4B fails", flush=True)
    launch(a, gpus, "teacher", tplan.name, chunk=60)
    label(a.dst)


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
    p.add_argument("--limit", type=int, default=None, help="relabel every teacher-correct question + the first N other teacher-labelled ones")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--widen", type=int, default=None, help="grow the bank: 4B-sweep N new questions, teacher on the 4B-wrong ones")
    p.add_argument("--model", default="student")
    p.add_argument("--plan", default="relabel_indices.json")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--label", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args()
    if a.worker:
        return worker(a.dst, a.shard, a.num_shards, a.model, a.plan, a.chunk)
    if a.label:
        return label(a.dst)
    if a.widen is not None:
        return widen(a, [g.strip() for g in a.gpus.split(",")])
    indices = prepare(a.src, a.dst, a.student, a.limit)
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
