"""Select math problems and sample K student traces per problem with vLLM.

One process per GPU (``CUDA_VISIBLE_DEVICES=0`` / ``=1``, ``--shard 0`` /
``--shard 1``). Every shard builds the same example list from the same seed and
keeps ``example_index % num_shards == shard``, then writes its own
``trajectories.shard{k}.jsonl``. The npz files are named by example index, so
both shards can share one ``student/`` directory without colliding.

The teacher is off by default for the pre-filter pass; ``--with-teacher`` runs
the HF teacher path, which needs the vLLM engine's memory and so is only sane
on its own run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from apod.datasets import DATASETS, append_jsonl, load_examples, read_jsonl, save_npz, write_jsonl
from apod.models import (
    STUDENT_ID,
    TEACHER_ID,
    build_llm,
    generate_teacher,
    generate_trajectories_vllm,
    load_teacher,
)
from apod.verification import verify_answer


def parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="openthoughts")
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-examples", type=int, default=128)
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=8, help="prompts per llm.generate call")
    parser.add_argument("--output-dir", default="outputs/trajectories")
    parser.add_argument("--student-id", default=STUDENT_ID)
    parser.add_argument("--teacher-id", default=TEACHER_ID)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--select-only", action="store_true", help="write examples.jsonl and stop (CPU only)")
    parser.add_argument("--with-teacher", action="store_true", help="also run the HF teacher generation")
    parser.add_argument("--resume", action="store_true", help="skip example indices already in a shard file")
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.num_shards:
        parser.error(f"--shard must be in [0, {args.num_shards}); got {args.shard}")
    if args.max_new_tokens > args.max_model_len:
        parser.error(f"--max-new-tokens {args.max_new_tokens} exceeds --max-model-len {args.max_model_len}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args.dataset, n=args.num_examples, seed=args.seed, split=args.split)

    if args.shard == 0:
        # One writer for the shared files, so two processes never race on them.
        write_jsonl(out / "examples.jsonl", examples)
        (out / "manifest.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"{len(examples)} examples selected (seed {args.seed})")
    if args.select_only:
        print(f"Wrote {out / 'examples.jsonl'}")
        return 0

    traj_path = out / f"trajectories.shard{args.shard}.jsonl"
    done = {row["example_index"] for row in read_jsonl(traj_path)} if args.resume else set()
    if not args.resume:
        traj_path.unlink(missing_ok=True)

    pending = [
        (i, ex)
        for i, ex in enumerate(examples)
        if i % args.num_shards == args.shard and i not in done
    ]
    print(f"shard {args.shard}/{args.num_shards}: {len(pending)} examples pending ({len(done)} already done)")
    if not pending:
        return 0

    llm = build_llm(
        args.student_id,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    teacher_tok, teacher = (None, None)
    if args.with_teacher:
        teacher_tok, teacher = load_teacher(args.teacher_id)

    started = time.monotonic()
    n_correct = 0
    n_total = 0
    n_truncated = 0

    batches = generate_trajectories_vllm(
        llm,
        tokenizer,
        [example["prompt"] for _, example in pending],
        n=args.num_rollouts,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    for position, batch in batches:
        example_index, example = pending[position]
        save_npz(out / "student" / f"{example_index:04d}.npz", batch)

        teacher_rel = None
        if teacher is not None:
            teacher_batch = generate_teacher(
                teacher, teacher_tok, example["prompt"], max_new_tokens=args.max_new_tokens
            )
            npz = save_npz(out / "teacher" / f"{example_index:04d}.npz", teacher_batch)
            teacher_rel = str(npz.relative_to(out))

        correct = [verify_answer(str(response), example["answer"]) for response in batch["responses"]]
        n_correct += sum(correct)
        n_total += len(correct)
        n_truncated += int(batch["truncated"].sum())

        for rollout_index in range(len(correct)):
            append_jsonl(
                traj_path,
                {
                    "example_index": example_index,
                    "rollout_index": rollout_index,
                    "id": example["id"],
                    "prompt": example["prompt"],
                    "response": str(batch["responses"][rollout_index]),
                    "answer": example["answer"],
                    "correct": bool(correct[rollout_index]),
                    "truncated": bool(batch["truncated"][rollout_index]),
                    "finish_reason": str(batch["finish_reasons"][rollout_index]),
                    "prompt_length": int(batch["prompt_length"]),
                    "response_length": int(batch["response_lengths"][rollout_index]),
                    "npz": f"student/{example_index:04d}.npz",
                    "teacher_path": teacher_rel,
                },
            )

        done_count = position + 1
        elapsed = time.monotonic() - started
        print(
            f"[{done_count}/{len(pending)}] example {example_index}  "
            f"correct {sum(correct)}/{len(correct)}  truncated {int(batch['truncated'].sum())}  "
            f"{elapsed / done_count:.1f}s/prompt",
            flush=True,
        )

    summary = {
        "shard": args.shard,
        "examples": len(pending),
        "correct": n_correct,
        "total": n_total,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "truncated": n_truncated,
        "seconds": round(time.monotonic() - started, 1),
    }
    (out / f"collection_summary.shard{args.shard}.json").write_text(json.dumps(summary, indent=2))
    print(f"Verification: {summary}")
    print(f"Stored under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
