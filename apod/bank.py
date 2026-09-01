"""Question bank: the labelled question pool for the correctness experiment.

ADR 0005 (docs/adr/0005-correctness-bucket-experiment.md). Every swept
question gets ``num_rollouts`` student rollouts at the bank's cap, a strict
label (C/W/M from >= correct_min / <= wrong_max strictly correct rollouts),
its question entropy H(q) (mean trajectory entropy of its student rollouts),
and -- only where that label can still fill a bucket -- ``num_rollouts``
teacher rollouts and a teacher label. Buckets (CONTEXT.md):
teacher_right_student_wrong, both_right, both_wrong, mixed (any labelled
question in none of the three, student-M included, without a teacher sweep)
and unlabelled (student swept, teacher not yet).

Layout (``conf/bank.yaml``; ``outputs/runs/<bank.name>/``):

  resolved_config.yaml            conf/ composed + the bank's rollout/sampling stamp
  pool/questions.jsonl            the whole usable dataset in one seeded order:
                                  {example_index, id, prompt, reference}
  questions.jsonl                 one row per student-swept question (schema in
                                  docs/pipeline.md, "Question bank"); rewritten
                                  by every build step and by --report
  student/  teacher/              the rollout layout of a round: trajectories
                                  .shard{K}.jsonl + tokens/example_XXXXX.npz +
                                  done.chunk{C}.shard{K}
  student/entropy/                entropy.shard{K}.jsonl + done.chunk{C}.shard{K}
  teacher/chunks/chunk_{C}.json   the persisted plan of teacher chunk C

The build is a loop of steps, each a subprocess per GPU (one vLLM engine per
process, torn down before the next step's engine is built): student chunk C
(generate) -> entropy chunk C -> ... until student_questions are swept, then
teacher chunks until every bucket holds target_per_bucket questions or no
unswept question can still fill an unfilled bucket. Every step is row-level
resumable (``load_complete_rows``) and the coordinator recomputes the next
step from the files, so a rerun after a crash issues no duplicate requests.

  python -m apod.bank [--gpus 0,1] [bank.student_questions=N ...]   build / resume
  python -m apod.bank --report                                       counts + cost
  python -m apod.bank --bank-dir D --sweep student|teacher --chunk C --shard K --num-shards N
  python -m apod.bank --bank-dir D --entropy --chunk C --shard K --num-shards N
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
from omegaconf import OmegaConf

from apod.datasets import append_jsonl, read_jsonl, write_jsonl
from apod.datasets.load import _load_rows, usable_examples
from apod.models import build_llm
from apod.stages.rollout_eval import GpuMemoryProbe, RolloutWriter, load_complete_rows, run_session
from apod.verification import grade

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("student", "teacher")
BUCKETS = ("teacher_right_student_wrong", "both_right", "both_wrong", "mixed", "unlabelled")
# Buckets a teacher sweep can still put a question in, by student label. A
# student-M question is mixed whatever the teacher says, so it is never swept.
REACHABLE = {
    "C": ("both_right", "mixed"),
    "W": ("teacher_right_student_wrong", "both_wrong", "mixed"),
    "M": (),
}
# The only settings that may change on an existing bank: loop limits, not the
# sampling regime the stored labels were produced under.
MUTABLE_OVERRIDES = ("bank.student_questions", "bank.target_per_bucket")


# --- labels -------------------------------------------------------------------


def strict_correct(row: dict[str, Any]) -> bool:
    """Boxed answer, Math-Verify accepted, and not cap-hit (CONTEXT.md, strict)."""

    return bool(row["correct"]) and bool(row["has_boxed"]) and not bool(row["truncated"])


def label(grades: list[bool], *, correct_min: int, wrong_max: int) -> str:
    n = sum(bool(g) for g in grades)
    return "C" if n >= correct_min else "W" if n <= wrong_max else "M"


def bucket_of(student_label: str, teacher_label: str | None) -> str:
    if student_label == "M":
        return "mixed"
    if teacher_label is None:
        return "unlabelled"
    if teacher_label == "C":
        return "teacher_right_student_wrong" if student_label == "W" else "both_right"
    if teacher_label == "W" and student_label == "W":
        return "both_wrong"
    return "mixed"


# --- reading the raw shards ---------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream rows without holding the file (8k-token responses) in memory.

    Same torn-tail policy as ``read_jsonl(drop_torn_tail=True)``: only a broken
    LAST line (a kill mid-append) is forgiven.
    """

    if not path.exists():
        return
    torn = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if torn:
                raise ValueError(f"{path}: corrupt line before the final one")
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                torn = True


def complete_groups(directory: Path, num_rollouts: int, fields: tuple[str, ...]) -> dict[int, list[dict]]:
    """example_index -> its ``num_rollouts`` rows, from the shard file holding the whole group.

    Groups are complete within ONE shard file (the writer appends a question's
    rows consecutively), which is also what ``load_complete_rows`` counts as
    done; a partial group in one file and a fresh full group in another (GPU
    count changed between runs) resolves to the full one.
    """

    groups: dict[int, list[dict]] = {}
    for path in sorted(directory.glob("trajectories.shard*.jsonl")):
        per_file: dict[int, list[dict]] = defaultdict(list)
        for row in iter_jsonl(path):
            per_file[int(row["example_index"])].append({key: row[key] for key in fields})
        for example_index, rows in per_file.items():
            if len(rows) >= num_rollouts:
                groups[example_index] = sorted(rows[-num_rollouts:], key=lambda r: r["rollout_index"])
    return groups


def complete_indices(path: Path, num_rollouts: int) -> set[int]:
    counts = Counter(int(row["example_index"]) for row in iter_jsonl(path))
    return {index for index, count in counts.items() if count >= num_rollouts}


def entropy_by_question(directory: Path) -> dict[int, dict[int, float]]:
    scores: dict[int, dict[int, float]] = defaultdict(dict)
    for path in sorted(directory.glob("entropy.shard*.jsonl")):
        for row in iter_jsonl(path):
            scores[int(row["example_index"])][int(row["rollout_index"])] = float(row["entropy"])
    return scores


# --- the bank -----------------------------------------------------------------


ROLLOUT_FIELDS = ("example_index", "rollout_index", "correct", "has_boxed", "truncated", "response_length")


def label_questions(bank_dir: Path, cfg) -> list[dict[str, Any]]:
    """Rebuild ``questions.jsonl`` from the raw shards; returns its rows in pool order."""

    bank = cfg.bank
    n = int(bank.num_rollouts)
    thresholds = {"correct_min": int(bank.correct_min), "wrong_max": int(bank.wrong_max)}
    student = complete_groups(bank_dir / "student", n, ROLLOUT_FIELDS)
    teacher = complete_groups(bank_dir / "teacher", n, ROLLOUT_FIELDS)
    entropies = entropy_by_question(bank_dir / "student" / "entropy")

    rows = []
    for question in read_jsonl(bank_dir / "pool" / "questions.jsonl"):
        index = int(question["example_index"])
        s = student.get(index)
        if s is None:
            continue
        t = teacher.get(index)
        student_grades = [strict_correct(r) for r in s]
        teacher_grades = [strict_correct(r) for r in t] if t else None
        student_label = label(student_grades, **thresholds)
        teacher_label = label(teacher_grades, **thresholds) if t else None
        scored = entropies.get(index, {})
        # The entropy worker skips empty responses like the stage does; H(q)
        # is defined once every non-empty rollout is scored.
        scorable = [r["rollout_index"] for r in s if int(r["response_length"]) > 0]
        complete = bool(scorable) and all(j in scored for j in scorable)
        rows.append(
            {
                "example_index": index,
                "id": question["id"],
                "question": question["prompt"],
                "reference": question["reference"],
                "chunk": index // int(bank.chunk_questions),
                "student_grades": student_grades,
                "student_truncated": [bool(r["truncated"]) for r in s],
                "student_lengths": [int(r["response_length"]) for r in s],
                "teacher_grades": teacher_grades,
                "teacher_truncated": [bool(r["truncated"]) for r in t] if t else None,
                "teacher_lengths": [int(r["response_length"]) for r in t] if t else None,
                "student_label": student_label,
                "teacher_label": teacher_label,
                "bucket": bucket_of(student_label, teacher_label),
                "question_entropy": float(np.mean([scored[j] for j in scorable])) if complete else None,
            }
        )
    write_jsonl(bank_dir / "questions.jsonl", rows)
    return rows


def load_bank(bank_dir: Path) -> list[dict[str, Any]]:
    """The labelled questions (``questions.jsonl``) in pool (seeded) order."""

    return read_jsonl(Path(bank_dir) / "questions.jsonl")


def bucket_questions(rows: list[dict[str, Any]], bucket: str) -> list[dict[str, Any]]:
    if bucket not in BUCKETS:
        raise ValueError(f"unknown bucket {bucket!r}; choose from {BUCKETS}")
    return [row for row in rows if row["bucket"] == bucket]


def bucket_counts(rows: list[dict[str, Any]]) -> Counter:
    counts = Counter(row["bucket"] for row in rows)
    return Counter({bucket: counts[bucket] for bucket in BUCKETS})


# --- planning -----------------------------------------------------------------


def student_chunk(cfg, chunk: int, pool_size: int) -> list[int]:
    size = int(cfg.bank.chunk_questions)
    limit = min(int(cfg.bank.student_questions), pool_size)
    return list(range(chunk * size, min((chunk + 1) * size, limit)))


def teacher_plans(bank_dir: Path) -> list[list[int]]:
    plans = []
    for path in sorted((bank_dir / "teacher" / "chunks").glob("chunk_*.json")):
        plans.append([int(i) for i in json.loads(path.read_text())["example_indices"]])
    return plans


def next_step(rows: list[dict[str, Any]], pool_size: int, plans: list[list[int]], cfg) -> tuple[str, int, list[int]] | None:
    """The next (kind, chunk, example_indices) to run, or None when the bank is done.

    Student chunks first (generate, then entropy, per chunk) up to
    student_questions; then any persisted teacher chunk that is not complete;
    then a new teacher chunk of the next chunk_questions eligible questions in
    pool order. Eligible: not teacher-swept and some bucket its teacher label
    could put it in (REACHABLE) is still below target_per_bucket.
    """

    bank = cfg.bank
    by_index = {int(row["example_index"]): row for row in rows}
    size = int(bank.chunk_questions)
    limit = min(int(bank.student_questions), pool_size)
    for chunk in range(math.ceil(limit / size)):
        indices = student_chunk(cfg, chunk, pool_size)
        if any(i not in by_index for i in indices):
            return "student", chunk, indices
        if any(by_index[i]["question_entropy"] is None for i in indices):
            return "entropy", chunk, indices

    for chunk, plan in enumerate(plans):
        if any(i not in by_index or by_index[i]["teacher_label"] is None for i in plan):
            return "teacher", chunk, plan

    counts = bucket_counts(rows)
    full = {bucket: counts[bucket] >= int(bank.target_per_bucket) for bucket in BUCKETS[:4]}
    if all(full.values()):
        return None
    eligible = [
        int(row["example_index"])
        for row in rows
        if row["teacher_label"] is None and any(not full[b] for b in REACHABLE[row["student_label"]])
    ]
    if not eligible:
        return None
    return "teacher", len(plans), eligible[:size]


# --- workers (one GPU each) ---------------------------------------------------


def chunk_indices(bank_dir: Path, cfg, model: str, chunk: int, pool_size: int) -> list[int]:
    if model == "student":
        return student_chunk(cfg, chunk, pool_size)
    plan = bank_dir / "teacher" / "chunks" / f"chunk_{chunk:03d}.json"
    return [int(i) for i in json.loads(plan.read_text())["example_indices"]]


def run_sweep(bank_dir: Path, cfg, model: str, chunk: int, shard: int, num_shards: int) -> None:
    """``num_rollouts`` rollouts of one model for this shard's share of a chunk.

    The rollout stage's writer does the work (request packing and seeds,
    grading pool overlapped with generation, npz + EOS repair, jsonl rows,
    marker); the chunk index plays the stage's round for the seed base.
    """

    pool = {int(q["example_index"]): q for q in read_jsonl(bank_dir / "pool" / "questions.jsonl")}
    out_dir = bank_dir / model
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(cfg.rollout.num_rollouts)
    path = out_dir / f"trajectories.shard{shard}.jsonl"
    marker = out_dir / f"done.chunk{chunk:03d}.shard{shard}"
    marker.unlink(missing_ok=True)

    done = load_complete_rows(path, "example_index", n, resume=True)
    for other in out_dir.glob("trajectories.shard*.jsonl"):
        if other != path:  # a previous run with a different GPU count
            done |= complete_indices(other, n)
    indices = chunk_indices(bank_dir, cfg, model, chunk, len(pool))
    pending = [pool[i] for i in indices if i % num_shards == shard and i not in done]
    print(
        f"bank {model} chunk {chunk} shard {shard}/{num_shards}: "
        f"{len(pending)} questions pending of {len(indices)} ({len(done)} done in this dir)",
        flush=True,
    )
    if not pending:
        marker.touch()
        return
    if float(cfg.sampling.presence_penalty) != 0.0 and not bool(cfg.sampling.fast_presence_penalty):
        raise RuntimeError("a nonzero presence_penalty requires sampling.fast_presence_penalty=true")

    model_path = str(cfg.model.student_id if model == "student" else cfg.model.teacher_id)
    memory = GpuMemoryProbe()
    memory.sample()
    # Fork the graders before the engine exists (no inherited CUDA context).
    graders = ProcessPoolExecutor(max_workers=8)
    list(graders.map(grade, [""] * 8, ["0"] * 8))
    llm = build_llm(
        model_path,
        max_model_len=int(cfg.engine.max_model_len),
        gpu_memory_utilization=float(cfg.engine.gpu_memory_utilization),
        seed=int(cfg.seed),
        fast_presence_penalty=bool(cfg.sampling.fast_presence_penalty),
    )
    tokenizer = llm.get_tokenizer()
    args = SimpleNamespace(shard=shard, round_index=chunk)
    writer = RolloutWriter(cfg, args, out_dir, pending, marker, model_path, tokenizer)
    run_session(llm, tokenizer, cfg, [writer], graders, memory)
    graders.shutdown()
    memory.sample()
    if memory.gib():
        print(f"peak GPU memory: {memory.gib():.2f} GiB", flush=True)


def load_entropy_scorer(model_path: str, logit_chunk: int):
    """HF student + the entropy stage's scorer: ``(ids, prompt_length, response_length) -> scores``."""

    from apod.models import load_lm
    from apod.stages.entropy import _decoder_and_head, trace_scores

    _, model = load_lm(model_path, frozen=True)
    body, head = _decoder_and_head(model)
    return lambda ids, prompt_length, response_length: trace_scores(
        model, body, head, ids, prompt_length, response_length, logit_chunk
    )


def run_entropy(bank_dir: Path, cfg, chunk: int, shard: int, num_shards: int) -> None:
    """Question entropy inputs: H(tau) of every student rollout of the chunk (entropy stage machinery)."""

    pool_size = sum(1 for _ in iter_jsonl(bank_dir / "pool" / "questions.jsonl"))
    tokens_dir = bank_dir / "student" / "tokens"
    entropy_dir = bank_dir / "student" / "entropy"
    entropy_dir.mkdir(parents=True, exist_ok=True)
    path = entropy_dir / f"entropy.shard{shard}.jsonl"
    marker = entropy_dir / f"done.chunk{chunk:03d}.shard{shard}"
    marker.unlink(missing_ok=True)

    scored = read_jsonl(path, drop_torn_tail=True)
    if scored:
        write_jsonl(path, scored)  # atomic rewrite: no torn line before new appends
    done = {(int(r["example_index"]), int(r["rollout_index"])) for r in scored}
    for other in entropy_dir.glob("entropy.shard*.jsonl"):
        if other != path:
            done |= {(int(r["example_index"]), int(r["rollout_index"])) for r in iter_jsonl(other)}
    examples = [
        (i, tokens_dir / f"example_{i:05d}.npz")
        for i in student_chunk(cfg, chunk, pool_size)
        if i % num_shards == shard and (tokens_dir / f"example_{i:05d}.npz").exists()
    ]

    model_path = str(cfg.model.student_id)
    with open(entropy_dir / "meta.json", "w") as f:
        json.dump({"model_path": model_path}, f)
    scorer = None
    scored_trajectories = 0
    for example_index, npz_path in examples:
        with np.load(npz_path, allow_pickle=True) as batch:
            input_ids = batch["input_ids"]
            prompt_length = int(batch["prompt_length"])
            response_lengths = batch["response_lengths"]
        for rollout_index in range(input_ids.shape[0]):
            response_length = int(response_lengths[rollout_index])
            if (example_index, rollout_index) in done or response_length <= 0:
                continue
            if scorer is None:
                scorer = load_entropy_scorer(model_path, int(cfg.selection.logit_chunk))
            scores = scorer(input_ids[rollout_index], prompt_length, response_length)
            append_jsonl(path, {"example_index": example_index, "rollout_index": rollout_index, **scores})
            scored_trajectories += 1
    marker.touch()
    print(
        f"bank entropy chunk {chunk} shard {shard}/{num_shards}: scored {scored_trajectories} "
        f"trajectories over {len(examples)} questions",
        flush=True,
    )


# --- coordinator --------------------------------------------------------------


def pool_examples(cfg) -> list[dict[str, Any]]:
    """Every usable question of the dataset in one seeded order (drawn without replacement)."""

    dataset = str(cfg.data.dataset)
    examples = usable_examples(_load_rows(dataset, cfg.data.split), dataset)
    random.Random(int(cfg.data.pool_seed)).shuffle(examples)
    return examples


def ensure_pool(bank_dir: Path, cfg) -> list[dict[str, Any]]:
    path = bank_dir / "pool" / "questions.jsonl"
    if path.exists():
        return read_jsonl(path)
    rows = [
        {"example_index": i, "id": ex["id"], "prompt": ex["prompt"], "reference": ex["answer"]}
        for i, ex in enumerate(pool_examples(cfg))
    ]
    write_jsonl(path, rows)
    print(f"pool: {len(rows)} questions in seed-{cfg.data.pool_seed} order -> {path}", flush=True)
    return rows


def launch_shards(bank_dir: Path, worker_argv: list[str], gpus: list[int]) -> None:
    """One ``apod.bank`` worker per GPU (CUDA_VISIBLE_DEVICES pinned); fail on any nonzero exit."""

    procs = []
    for shard, gpu in enumerate(gpus):
        cmd = [
            sys.executable, "-m", "apod.bank", "--bank-dir", str(bank_dir), *worker_argv,
            "--shard", str(shard), "--num-shards", str(len(gpus)),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        env.setdefault("HF_HUB_OFFLINE", "1")
        print(f"launch (CUDA_VISIBLE_DEVICES={gpu}): {' '.join(cmd[2:])}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=REPO_ROOT, env=env))
    codes = [proc.wait() for proc in procs]
    if any(codes):
        raise RuntimeError(f"bank workers {worker_argv} exited {codes}")


def progress_line(rows: list[dict[str, Any]], cfg) -> str:
    counts = bucket_counts(rows)
    target = int(cfg.bank.target_per_bucket)
    swept_teacher = sum(row["teacher_label"] is not None for row in rows)
    return (
        f"bank: student {len(rows)} swept, teacher {swept_teacher} swept | "
        + "  ".join(f"{b} {counts[b]}/{target}" for b in BUCKETS[:4])
        + f"  unlabelled {counts['unlabelled']}"
    )


def build(bank_dir: Path, cfg, gpus: list[int]) -> list[dict[str, Any]]:
    pool = ensure_pool(bank_dir, cfg)
    last: tuple[str, int] | None = None
    while True:
        rows = label_questions(bank_dir, cfg)
        print(progress_line(rows, cfg), flush=True)
        plans = teacher_plans(bank_dir)
        step = next_step(rows, len(pool), plans, cfg)
        if step is None:
            return rows
        kind, chunk, indices = step
        if (kind, chunk) == last:
            raise RuntimeError(f"{kind} chunk {chunk} made no progress; see the worker output above")
        last = (kind, chunk)
        if kind == "teacher" and chunk == len(plans):
            plan = bank_dir / "teacher" / "chunks" / f"chunk_{chunk:03d}.json"
            plan.parent.mkdir(parents=True, exist_ok=True)
            plan.write_text(json.dumps({
                "chunk": chunk, "example_indices": indices, "bucket_counts": dict(bucket_counts(rows)),
            }) + "\n")
        print(f"step: {kind} chunk {chunk} ({len(indices)} questions)", flush=True)
        argv = ["--entropy"] if kind == "entropy" else ["--sweep", kind]
        launch_shards(bank_dir, [*argv, "--chunk", str(chunk)], gpus)


# --- report -------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def remaining_generation(rows: list[dict[str, Any]], pool_size: int, cfg) -> dict[str, Any]:
    """Trajectories and hours still to generate, from the bank's own outcome rates.

    Teacher need per bucket = deficit / P(teacher label | student label) among
    the questions already teacher-swept (no data yet: every eligible question
    counts); questions the remaining student sweep will add are credited at
    the observed student-label rates. An estimate, not a schedule: the loop
    itself only ever sweeps what is eligible right now.
    """

    bank = cfg.bank
    n = int(bank.num_rollouts)
    target = int(bank.target_per_bucket)
    counts = bucket_counts(rows)
    limit = min(int(bank.student_questions), pool_size)
    student_left = max(0, limit - len(rows))

    student_labels = Counter(row["student_label"] for row in rows)
    p_student = {s: _rate(student_labels[s], len(rows)) or 0.0 for s in ("C", "W", "M")}
    swept = [row for row in rows if row["teacher_label"] is not None]
    by_student = Counter(row["student_label"] for row in swept)
    outcome = Counter((row["student_label"], row["teacher_label"]) for row in swept)

    def need(student_label: str, buckets: dict[str, str]) -> float | None:
        """Questions of ``student_label`` to teacher-sweep so each bucket reaches target (None: no rates yet)."""

        needs = []
        for bucket, teacher_label in buckets.items():
            deficit = max(0, target - counts[bucket])
            if deficit == 0:
                continue
            rate = _rate(outcome[(student_label, teacher_label)], by_student[student_label])
            if not rate:  # no teacher data for this label yet, or the bucket has never been hit
                return None
            needs.append(deficit / rate)
        return max(needs, default=0.0)

    need_c = need("C", {"both_right": "C"})
    need_w = need("W", {"teacher_right_student_wrong": "C", "both_wrong": "W", "mixed": "M"})
    unswept = Counter(row["student_label"] for row in rows if row["teacher_label"] is None)
    avail = {s: unswept[s] + student_left * p_student[s] for s in ("C", "W")}
    teacher_questions = 0.0
    short_student = 0.0
    for s, needed in (("C", need_c), ("W", need_w)):
        if needed is None:
            teacher_questions += avail[s]
            continue
        teacher_questions += min(needed, avail[s])
        if needed > avail[s] and p_student[s] > 0:
            short_student = max(short_student, (needed - avail[s]) / p_student[s])

    per_min = float(bank.student_trajectories_per_min) * int(cfg.num_gpus) / int(bank.throughput_gpus)
    student_traj = student_left * n
    teacher_traj = teacher_questions * n
    return {
        "student_questions": student_left,
        "student_trajectories": student_traj,
        "student_hours": student_traj / per_min / 60,
        "teacher_questions": teacher_questions,
        "teacher_trajectories": teacher_traj,
        "teacher_hours": teacher_traj / (per_min / float(bank.teacher_slowdown)) / 60,
        "extra_student_questions": short_student,
        "trajectories_per_min": per_min,
    }


def _pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):5.1f}%" if values else "    --"


def report(bank_dir: Path, cfg) -> str:
    bank = cfg.bank
    pool_size = sum(1 for _ in iter_jsonl(bank_dir / "pool" / "questions.jsonl"))
    rows = label_questions(bank_dir, cfg) if pool_size else []
    n = int(bank.num_rollouts)
    target = int(bank.target_per_bucket)
    counts = bucket_counts(rows)
    swept = [row for row in rows if row["teacher_label"] is not None]
    lines = [
        f"{bank.name}: {bank_dir}",
        f"pool {pool_size} questions (seed {cfg.data.pool_seed}), student_questions {bank.student_questions}, "
        f"chunk {bank.chunk_questions}, cap {bank.max_new_tokens}, {n} rollouts, target {target}/bucket, "
        f"labels C >= {bank.correct_min}/{n}, W <= {bank.wrong_max}/{n}",
        f"student swept {len(rows)} questions ({len(rows) * n} trajectories), "
        f"H(q) scored {sum(row['question_entropy'] is not None for row in rows)}; "
        f"teacher swept {len(swept)} ({len(swept) * n} trajectories)",
        "student labels " + "  ".join(f"{s} {c}" for s, c in sorted(Counter(r['student_label'] for r in rows).items()))
        + " | teacher labels "
        + "  ".join(f"{s} {c}" for s, c in sorted(Counter(r['teacher_label'] for r in swept).items())),
        "",
        f"{'bucket':<28}{'questions':>10}{'target':>8}  {'student cap-hit':>16}  {'teacher cap-hit':>16}  {'teacher 4/4 cap-hit':>20}",
    ]
    for bucket in BUCKETS:
        members = bucket_questions(rows, bucket)
        s_trunc = [t for row in members for t in row["student_truncated"]]
        t_rows = [row for row in members if row["teacher_truncated"] is not None]
        t_trunc = [t for row in t_rows for t in row["teacher_truncated"]]
        t_all = [all(row["teacher_truncated"]) for row in t_rows]
        status = "" if bucket == "unlabelled" else ("  full" if counts[bucket] >= target else "")
        lines.append(
            f"{bucket:<28}{counts[bucket]:>10}{'' if bucket == 'unlabelled' else target:>8}  "
            f"{_pct(s_trunc):>16}  {_pct(t_trunc):>16}  {_pct(t_all):>20}{status}"
        )
    est = remaining_generation(rows, pool_size, cfg)
    lines += [
        "",
        f"remaining generation (at {est['trajectories_per_min']:.0f} student trajectories/min on "
        f"{cfg.num_gpus} GPUs, teacher {bank.teacher_slowdown}x slower):",
        f"  student: {est['student_questions']} questions = {est['student_trajectories']:.0f} trajectories "
        f"~ {est['student_hours']:.1f} h",
        f"  teacher: ~{est['teacher_questions']:.0f} questions = {est['teacher_trajectories']:.0f} trajectories "
        f"~ {est['teacher_hours']:.1f} h",
    ]
    unfilled = [b for b in BUCKETS[:4] if counts[b] < target]
    if unfilled and est["extra_student_questions"] > 0:
        lines.append(
            f"  {', '.join(unfilled)} will not fill from student_questions={bank.student_questions}: "
            f"raise it by ~{est['extra_student_questions']:.0f} (bank.student_questions=N), pool has {pool_size}"
        )
    elif unfilled and est["student_questions"] == 0 and est["teacher_questions"] == 0:
        lines.append(
            f"  {', '.join(unfilled)} short with no eligible question left at "
            f"student_questions={bank.student_questions}: raise it (bank.student_questions=N), pool has {pool_size}"
        )
    return "\n".join(lines)


# --- config + CLI -------------------------------------------------------------


def compose_config(overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def load_config(bank_dir: Path | None, overrides: list[str], *, create: bool = True) -> tuple[Path, Any]:
    """The bank's resolved config: composed once from conf/, then read from the bank dir.

    A fresh bank stamps its rollout/sampling regime (num_rollouts, cap, and
    chunk_questions as the seed stride per chunk) into the config every stage
    helper reads. On an existing bank only MUTABLE_OVERRIDES may change.
    """

    if bank_dir is None:
        composed = compose_config(overrides)
        bank_dir = Path(composed.bank.dir)
        if not bank_dir.is_absolute():
            bank_dir = REPO_ROOT / bank_dir
    bank_dir = bank_dir.resolve()
    path = bank_dir / "resolved_config.yaml"
    if path.exists():
        cfg = OmegaConf.load(path)
        if overrides:
            keys = [item.partition("=")[0] for item in overrides]
            illegal = [key for key in keys if key not in MUTABLE_OVERRIDES]
            if illegal:
                raise SystemExit(
                    f"{illegal} cannot change on an existing bank (only {list(MUTABLE_OVERRIDES)}); "
                    "a different regime is a different bank: set bank.name"
                )
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
            OmegaConf.save(config=cfg, f=path, resolve=True)
        return bank_dir, cfg

    if not create:
        raise SystemExit(f"no bank at {bank_dir} (no resolved_config.yaml); build one with python -m apod.bank")
    cfg = compose_config(overrides)
    cfg.run_name = cfg.bank.name
    cfg.output_dir = str(bank_dir)
    cfg.bank.dir = str(bank_dir)
    cfg.rollout.num_rollouts = int(cfg.bank.num_rollouts)
    cfg.rollout.num_prompts = int(cfg.bank.chunk_questions)  # RolloutWriter's per-chunk seed stride
    cfg.sampling.max_new_tokens = int(cfg.bank.max_new_tokens)
    cfg.resume = True
    bank_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=path, resolve=True)
    return bank_dir, OmegaConf.load(path)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank-dir", type=Path, default=None, help="default: conf bank.dir (outputs/runs/<bank.name>)")
    parser.add_argument("--gpus", default=None, help="GPUs for the build, e.g. 0,1 (default: 0..num_gpus-1)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--report", action="store_true", help="relabel and print bucket counts + remaining cost")
    mode.add_argument("--sweep", choices=MODELS, metavar="MODEL", help="worker: rollouts of one chunk shard")
    mode.add_argument("--entropy", action="store_true", help="worker: H(tau) of one student chunk shard")
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. bank.student_questions=20000")
    args = parser.parse_args(argv)
    if args.sweep or args.entropy:
        if None in (args.chunk, args.shard, args.num_shards):
            parser.error("workers need --chunk, --shard and --num-shards")
        if not 0 <= args.shard < args.num_shards:
            parser.error(f"--shard must be in [0, {args.num_shards})")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bank_dir, cfg = load_config(args.bank_dir, args.overrides, create=not args.report)
    if args.sweep:
        run_sweep(bank_dir, cfg, args.sweep, args.chunk, args.shard, args.num_shards)
        return 0
    if args.entropy:
        run_entropy(bank_dir, cfg, args.chunk, args.shard, args.num_shards)
        return 0
    if args.report:
        print(report(bank_dir, cfg))
        return 0
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(int(cfg.num_gpus)))
    build(bank_dir, cfg, gpus)
    print(report(bank_dir, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
