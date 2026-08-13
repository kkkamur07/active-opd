"""Select math problems, sample student traces, generate one teacher completion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from apod.datasets import DATASETS, append_jsonl, load_examples, read_jsonl, save_npz, write_jsonl
from apod.models import STUDENT_ID, TEACHER_ID, generate_teacher, generate_trajectories, load_student, load_teacher
from apod.verification import verify_answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="openthoughts")
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-examples", type=int, default=512)
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", default="outputs/trajectories")
    parser.add_argument("--student-id", default=STUDENT_ID)
    parser.add_argument("--teacher-id", default=TEACHER_ID)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args.dataset, n=args.num_examples, seed=args.seed, split=args.split)
    write_jsonl(out / "examples.jsonl", examples)
    (out / "manifest.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"Wrote {len(examples)} examples to {out / 'examples.jsonl'}")
    if args.select_only:
        return 0

    traj_path = out / "trajectories.jsonl"
    done = {row["example_index"] for row in read_jsonl(traj_path)} if args.resume else set()

    student_tok, student = load_student(args.student_id, device_map=args.device_map)
    teacher_tok, teacher = (None, None)
    if not args.skip_teacher:
        teacher_tok, teacher = load_teacher(args.teacher_id, device_map=args.device_map)

    n_correct = 0
    n_total = 0
    pending = [(i, ex) for i, ex in enumerate(examples) if i not in done]
    for example_index, example in tqdm(pending, desc="examples"):
        batch = generate_trajectories(
            student,
            student_tok,
            example["prompt"],
            n=args.num_rollouts,
            max_new_tokens=args.max_new_tokens,
        )
        save_npz(out / "student" / f"{example_index:04d}.npz", batch)

        teacher_rel = None
        if teacher is not None:
            teacher_batch = generate_teacher(
                teacher,
                teacher_tok,
                example["prompt"],
                max_new_tokens=args.max_new_tokens,
            )
            npz = save_npz(out / "teacher" / f"{example_index:04d}.npz", teacher_batch)
            teacher_rel = str(npz.relative_to(out))

        correct = [verify_answer(str(response), example["answer"]) for response in batch["responses"]]
        n_correct += sum(correct)
        n_total += len(correct)
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
                    "correct": correct[rollout_index],
                    "truncated": bool(batch["truncated"][rollout_index]),
                    "prompt_length": int(batch["prompt_length"]),
                    "response_length": int(batch["response_lengths"][rollout_index]),
                    "teacher_path": teacher_rel,
                },
            )

    summary = {"correct": n_correct, "total": n_total}
    (out / "verification_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Verification: {summary}")
    print(f"Stored under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
