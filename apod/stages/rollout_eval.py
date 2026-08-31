"""Rollout + eval stage: one vLLM engine session per (arm, round, shard).

Runs the MATH-500 eval of the round's starting model FIRST, then the round's
rollouts from that same engine (skipped with ``--eval-only`` for the final
eval-only round). Written against ``docs/pipeline.md``; loads
``<run-dir>/resolved_config.yaml`` with ``OmegaConf.load`` (stages are plain
scripts, not Hydra apps). The driver launches one process per shard with
``CUDA_VISIBLE_DEVICES`` set to that shard's GPU.

Seeds:
  eval     -- deterministic per (round, problem):
              ``seed + eval_seed_offset + round * num_problems + problem_index``
              so re-runs of a round reproduce and rounds differ, independent of
              sharding and resume state.
  rollouts -- base ``seed + round * num_prompts`` handed to
              ``generate_trajectories_vllm``, which derives ``base + chunk_start``
              per chunk; the per-row ``seed`` field records that chunk seed.
              With 8 rounds x 128 prompts the rollout seeds stay far below
              ``eval_seed_offset``, so the two streams never collide.

Token storage: TRL's GKD trainer feeds the stored token ids to the student
verbatim, so every non-truncated row must end with an EOS the model actually
emitted. vLLM 0.26.0 includes the terminating EOS in
``CompletionOutput.token_ids`` for ``finish_reason == "stop"`` (verified in the
installed source: the scheduler trims only tokens after the stop token and the
detokenizer re-appends the skipped stop token to its id list), so the
conditional append below is a safety net -- it only fires for stop-*string*
terminations, which match on text and carry no EOS id, or on a regression.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from apod import paths
from apod.datasets import append_jsonl, read_jsonl, save_npz, write_jsonl
from apod.models import build_llm, generate_trajectories_vllm, render_prompt
from apod.models.generate_vllm import build_sampling_params
from apod.stages.common import parse_stage_args, stage_parser
from apod.verification import grade


def _visible_device_index() -> int | None:
    """NVML indexes physical GPUs; ``CUDA_VISIBLE_DEVICES`` renumbers them."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return 0
    first = visible.split(",")[0].strip()
    return int(first) if first.isdigit() else None


class GpuMemoryProbe:
    """Peak device memory on this process's GPU, sampled through NVML.

    ``torch.cuda`` counters are the wrong instrument here: vLLM V1 runs the
    model in a separate EngineCore process, so this process's allocator never
    sees the weights or the KV cache. NVML reads the device instead. Best
    effort: if NVML is missing or the device cannot be resolved, the peak is
    simply None.
    """

    def __init__(self) -> None:
        self.peak_bytes: int | None = None
        self._nvml = None
        self._handle = None
        index = _visible_device_index()
        if index is None:
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._nvml = pynvml
        except Exception:
            self._nvml = None
            self._handle = None

    def sample(self) -> None:
        if self._handle is None:
            return
        try:
            used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
        except Exception:
            self._handle = None
            return
        self.peak_bytes = used if self.peak_bytes is None else max(self.peak_bytes, used)

    def gib(self) -> float | None:
        return round(self.peak_bytes / 2**30, 2) if self.peak_bytes else None


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def parse_args(argv: list[str] | None):
    parser = stage_parser(description=__doc__)
    parser.add_argument(
        "--eval-only", action="store_true", help="run only the eval (the final measurement round)"
    )
    parser.add_argument(
        "--eval-num-problems",
        type=int,
        default=None,
        help="evaluate only the first N problems of the materialized set "
        "(driver sends the intermediate-round subset; default: all)",
    )
    return parse_stage_args(parser, argv)


def resolve_model_path(run_dir: Path, arm: str, round_index: int, cfg) -> str:
    """``round_{X-1}/checkpoint`` for X >= 1, the base student for X = 0.

    A missing checkpoint at X >= 1 is a driver-ordering or corruption bug;
    silently falling back to the base student would measure the wrong model,
    which is a wrong experiment rather than a failed one. Fail instead.
    """

    if round_index == 0:
        return str(cfg.model.student_id)
    checkpoint = paths.checkpoint_dir(run_dir, arm, round_index - 1)
    if not any(checkpoint.glob("*.safetensors")):
        raise FileNotFoundError(
            f"round {round_index} expects weights in {checkpoint}; either the "
            "previous round's train stage did not finish, or retention pruned "
            "them (keep_checkpoints in conf/config.yaml) and this round is "
            "being re-run past its prune horizon"
        )
    return str(checkpoint)


def load_complete_rows(path: Path, key: str, group_size: int, resume: bool) -> set[int]:
    """Done-set of ``key`` values with all ``group_size`` rows present.

    Rows for one problem/example are appended consecutively, so a crash can
    leave a partial group at the tail. Partial groups are dropped from the file
    (their work reruns) rather than counted done, so resume never duplicates or
    under-fills a group.
    """

    if not resume:
        path.unlink(missing_ok=True)
        return set()
    rows = read_jsonl(path, drop_torn_tail=True)
    counts = Counter(row[key] for row in rows)
    done = {value for value, count in counts.items() if count >= group_size}
    if rows:
        # Unconditional atomic rewrite: drops partial tail groups AND any
        # torn final line, so the file is clean before appends resume.
        write_jsonl(path, [row for row in rows if row[key] in done])
    return done


def ensure_trailing_eos(batch: dict[str, Any], eos_ids: set[int], eos_id: int, pad_id: int) -> int:
    """Append EOS to any ``stop`` row whose stored ids do not already end in one.

    Normally a no-op on vLLM 0.26.0 (see module docstring); covers stop-string
    terminations and regressions. Returns the number of rows repaired.
    """

    input_ids = batch["input_ids"]
    prompt_length = int(batch["prompt_length"])
    lengths = batch["response_lengths"]
    repaired = 0
    for row, reason in enumerate(batch["finish_reasons"]):
        if str(reason) != "stop":
            continue
        length = int(lengths[row])
        if length > 0 and int(input_ids[row, prompt_length + length - 1]) in eos_ids:
            continue
        if prompt_length + length >= input_ids.shape[1]:
            pad = np.full((input_ids.shape[0], 1), pad_id, dtype=input_ids.dtype)
            input_ids = np.concatenate([input_ids, pad], axis=1)
            batch["input_ids"] = input_ids
        input_ids[row, prompt_length + length] = eos_id
        lengths[row] = length + 1
        repaired += 1
    return repaired


def collect_eos_ids(tokenizer, model_path: str) -> set[int]:
    """All ids that legitimately terminate a completion.

    Qwen3 declares two (``<|im_end|>`` and ``<|endoftext|>``) in its generation
    config; the tokenizer only exposes one of them as ``eos_token_id``. Best
    effort on the generation config -- a checkpoint without one still leaves
    the tokenizer's EOS.
    """

    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    # The base 2B repo ships NO generation_config.json, and its config.json
    # declares a DIFFERENT eos (<|endoftext|>) than the tokenizer
    # (<|im_end|>); trainer-saved checkpoints declare both. Union all three
    # sources so the terminator set is identical at round 0 and later rounds.
    # Best effort per source, as before: a missing or broken config leaves
    # the other sources' ids in place.
    try:
        from transformers import AutoConfig, GenerationConfig
    except Exception:
        return ids
    for load in (GenerationConfig.from_pretrained, AutoConfig.from_pretrained):
        try:
            declared = load(model_path).eos_token_id
        except Exception:
            continue
        if isinstance(declared, int):
            ids.add(declared)
        elif declared is not None:
            ids.update(int(token) for token in declared)
    return ids


def _sampling_kwargs(cfg) -> dict[str, Any]:
    """The distribution settings shared by eval and rollouts.

    One source, so the two stages provably sample from the same distribution
    (per-call knobs like n and seed stay with the callers).
    """

    return {
        "max_tokens": int(cfg.sampling.max_new_tokens),
        "temperature": float(cfg.sampling.temperature),
        "top_p": float(cfg.sampling.top_p),
        "top_k": int(cfg.sampling.top_k),
        "presence_penalty": float(cfg.sampling.presence_penalty),
    }


def run_eval(llm, tokenizer, cfg, args, eval_dir: Path, pending, memory: GpuMemoryProbe) -> None:
    eval_path = eval_dir / f"eval.shard{args.shard}.jsonl"
    num_samples = int(cfg.eval.num_samples)
    chunk_size = max(1, int(cfg.engine.target_concurrent_sequences) // num_samples)
    seed_base = int(cfg.seed) + int(cfg.eval.eval_seed_offset) + args.round_index * int(cfg.eval.num_problems)
    sampling = _sampling_kwargs(cfg)

    started = time.monotonic()
    generate_seconds = 0.0
    generated_tokens = 0
    n_correct = 0
    n_total = 0
    n_truncated = 0
    response_tokens = 0

    for chunk_start in range(0, len(pending), chunk_size):
        chunk = pending[chunk_start : chunk_start + chunk_size]
        texts = [
            render_prompt(tokenizer, ex["prompt"], enable_thinking=bool(cfg.model.enable_thinking))
            for _, ex in chunk
        ]
        params = [
            build_sampling_params(
                n=num_samples,
                # Per (round, problem), independent of sharding and resume.
                seed=seed_base + problem_index,
                fast_presence_penalty=bool(cfg.sampling.fast_presence_penalty),
                **sampling,
            )
            for problem_index, _ in chunk
        ]
        began = time.perf_counter()
        outputs = llm.generate(texts, params)
        seconds = time.perf_counter() - began
        memory.sample()

        chunk_tokens = 0
        for (problem_index, example), request_output in zip(chunk, outputs):
            for sample_index, completion in enumerate(request_output.outputs):
                truncated = completion.finish_reason == "length"
                verdict = grade(str(completion.text), example["answer"])
                chunk_tokens += len(completion.token_ids)
                response_tokens += len(completion.token_ids)
                n_correct += int(verdict["correct"])
                n_total += 1
                n_truncated += int(truncated)
                append_jsonl(
                    eval_path,
                    {
                        "problem_index": problem_index,
                        "sample_index": sample_index,
                        "id": example["id"],
                        "response_length": len(completion.token_ids),
                        "truncated": bool(truncated),
                        "correct": bool(verdict["correct"]),
                        # Distinguishes "answered wrong" from "output became
                        # unparseable" -- a template/thinking regression would
                        # otherwise look identical to a capability drop.
                        "has_answer": bool(verdict["has_answer"]),
                        "has_boxed": bool(verdict["has_boxed"]),
                        "gold_parsed": bool(verdict["gold_parsed"]),
                    },
                )

        generate_seconds += seconds
        generated_tokens += chunk_tokens
        done = min(chunk_start + chunk_size, len(pending))
        wall = time.monotonic() - started
        eta = (wall / done) * (len(pending) - done) if done else 0.0
        print(
            f"eval  problems {done}/{len(pending)}  {seconds:.1f}s  "
            f"gen {chunk_tokens / seconds if seconds else 0.0:.0f} tok/s "
            f"(run {generated_tokens / generate_seconds if generate_seconds else 0.0:.0f})  "
            f"correct {n_correct}/{n_total}  truncated {n_truncated}  "
            f"elapsed {format_duration(wall)}  eta {format_duration(eta)}",
            flush=True,
        )

    wall = time.monotonic() - started
    accuracy = n_correct / n_total if n_total else 0.0
    mean_length = response_tokens / n_total if n_total else 0.0
    print(
        f"eval done: {len(pending)} problems  correct {n_correct}/{n_total} ({accuracy:.3f})  "
        f"truncated {n_truncated}  mean {mean_length:.0f} tok/sample  "
        f"{generated_tokens / generate_seconds if generate_seconds else 0.0:.0f} gen tok/s over "
        f"{format_duration(generate_seconds)} of generate ({format_duration(wall)} wall)",
        flush=True,
    )


def run_rollouts(
    llm, tokenizer, cfg, args, rollouts_dir: Path, pending, model_path: str, memory: GpuMemoryProbe
) -> None:
    traj_path = rollouts_dir / f"trajectories.shard{args.shard}.jsonl"
    tokens_dir = rollouts_dir / "tokens"
    num_rollouts = int(cfg.rollout.num_rollouts)
    chunk_size = max(1, int(cfg.engine.target_concurrent_sequences) // num_rollouts)
    # Round-offset base; generate_trajectories_vllm derives base + chunk_start
    # per chunk, so the whole run stays below eval_seed_offset.
    base_seed = int(cfg.seed) + args.round_index * int(cfg.rollout.num_prompts)

    eos_ids = collect_eos_ids(tokenizer, model_path)
    eos_id = int(tokenizer.eos_token_id)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id

    started = time.monotonic()
    n_correct = 0
    n_total = 0
    n_truncated = 0
    n_repaired = 0
    throughput: dict[str, Any] = {}

    def on_chunk(metrics: dict[str, Any]) -> None:
        throughput.update(metrics["cumulative"])
        memory.sample()
        cumulative = metrics["cumulative"]
        done = cumulative["prompts"]
        wall = time.monotonic() - started
        eta = (wall / done) * (len(pending) - done) if done else 0.0
        print(
            f"rollouts  chunk {metrics['chunk_index']}  prompts {done}/{len(pending)}  "
            f"{metrics['seconds']:.1f}s  "
            f"gen {metrics['generated_tokens_per_s']:.0f} tok/s "
            f"(run {cumulative['generated_tokens_per_s']:.0f})  "
            f"total {metrics['total_tokens_per_s']:.0f} tok/s  "
            f"seq {metrics['sequences_per_s']:.2f}/s  "
            f"mean {metrics['mean_generated_tokens_per_sequence']:.0f} tok/trace  "
            f"elapsed {format_duration(wall)}  eta {format_duration(eta)}",
            flush=True,
        )

    batches = generate_trajectories_vllm(
        llm,
        tokenizer,
        [row["prompt"] for row in pending],
        n=num_rollouts,
        chunk_size=chunk_size,
        seed=base_seed,
        enable_thinking=bool(cfg.model.enable_thinking),
        on_chunk=on_chunk,
        **_sampling_kwargs(cfg),
    )

    for position, batch in batches:
        row = pending[position]
        example_index = int(row["example_index"])
        # Repair BEFORE the npz and the jsonl rows so the stored lengths, the
        # stored ids, and what the trainer feeds are one and the same thing.
        n_repaired += ensure_trailing_eos(batch, eos_ids, eos_id, pad_id)
        save_npz(tokens_dir / f"example_{example_index:05d}.npz", batch)

        grades = [grade(str(response), row["reference"]) for response in batch["responses"]]
        n_correct += sum(g["correct"] for g in grades)
        n_total += len(grades)
        n_truncated += int(batch["truncated"].sum())
        chunk_seed = base_seed + (position // chunk_size) * chunk_size

        for rollout_index, verdict in enumerate(grades):
            append_jsonl(
                traj_path,
                {
                    "example_index": example_index,
                    "rollout_index": rollout_index,
                    "id": row["id"],
                    "prompt_length": int(batch["prompt_length"]),
                    "response": str(batch["responses"][rollout_index]),
                    "response_length": int(batch["response_lengths"][rollout_index]),
                    "truncated": bool(batch["truncated"][rollout_index]),
                    "finish_reason": str(batch["finish_reasons"][rollout_index]),
                    "correct": bool(verdict["correct"]),
                    "has_answer": bool(verdict["has_answer"]),
                    "has_boxed": bool(verdict["has_boxed"]),
                    "seed": chunk_seed,
                },
            )

    wall = time.monotonic() - started
    accuracy = n_correct / n_total if n_total else 0.0
    print(
        f"rollouts done: {len(pending)} prompts x {num_rollouts}  "
        f"correct {n_correct}/{n_total} ({accuracy:.3f})  truncated {n_truncated}  "
        f"eos repaired {n_repaired}",
        flush=True,
    )
    if throughput:
        print(
            f"rollout throughput: {throughput['generated_tokens_per_s']:.0f} generated tok/s  "
            f"{throughput['total_tokens_per_s']:.0f} total tok/s  "
            f"{throughput['sequences_per_s']:.2f} seq/s over "
            f"{format_duration(throughput['seconds'])} of generate ({format_duration(wall)} wall)",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    resume = bool(cfg.resume)

    round_dir = paths.round_dir(run_dir, args.arm, args.round_index)
    eval_dir = round_dir / "eval"
    rollouts_dir = round_dir / "rollouts"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_marker = eval_dir / f"done.shard{args.shard}"
    eval_marker.unlink(missing_ok=True)
    rollout_marker = rollouts_dir / f"done.shard{args.shard}"
    if not args.eval_only:
        rollouts_dir.mkdir(parents=True, exist_ok=True)
        rollout_marker.unlink(missing_ok=True)

    model_path = resolve_model_path(run_dir, args.arm, args.round_index, cfg)
    print(
        f"{args.arm} round {args.round_index} shard {args.shard}/{args.num_shards}: "
        f"model {model_path}" + ("  (eval only)" if args.eval_only else ""),
        flush=True,
    )
    if not args.eval_only:
        # Auditable from artifacts (logs rotate; the run dir survives), and
        # the entropy stage asserts it scored under this exact model.
        with open(rollouts_dir / "model_path.json", "w") as f:
            json.dump({"model_path": model_path}, f)

    # --- pending work, computed before the engine so a no-op resume skips it ---
    # The eval set is materialized once by the driver (pool/eval_problems.jsonl)
    # so problem_index -> problem is immutable for the run's lifetime; loading
    # from the live Hub dataset here would let an upstream update silently
    # remap indices between rounds.
    problems = read_jsonl(run_dir / "pool" / "eval_problems.jsonl")
    if len(problems) != int(cfg.eval.num_problems):
        raise RuntimeError(
            f"pool/eval_problems.jsonl has {len(problems)} problems, expected "
            f"eval.num_problems={cfg.eval.num_problems}; run the driver "
            "(apod.main) to materialize the eval set before invoking stages"
        )
    # Intermediate rounds evaluate a PREFIX of the materialized set, so
    # problem_index means the same problem at every round and eval size.
    if args.eval_num_problems is not None:
        problems = problems[: args.eval_num_problems]
    eval_done = load_complete_rows(
        eval_dir / f"eval.shard{args.shard}.jsonl", "problem_index", int(cfg.eval.num_samples), resume
    )
    eval_pending = [
        (i, ex)
        for i, ex in enumerate(problems)
        if i % args.num_shards == args.shard and i not in eval_done
    ]

    rollout_pending: list[dict[str, Any]] = []
    if not args.eval_only:
        pool = read_jsonl(run_dir / "pool" / "prompts.jsonl")
        rollout_rows = [row for row in pool if int(row["round"]) == args.round_index]
        if not rollout_rows:
            raise RuntimeError(
                f"pool/prompts.jsonl has no rows for round {args.round_index}; "
                "was the pool built for enough rounds?"
            )
        rollout_done = load_complete_rows(
            rollouts_dir / f"trajectories.shard{args.shard}.jsonl",
            "example_index",
            int(cfg.rollout.num_rollouts),
            resume,
        )
        rollout_pending = [
            row
            for row in rollout_rows
            if int(row["example_index"]) % args.num_shards == args.shard
            and int(row["example_index"]) not in rollout_done
        ]
        if (
            float(cfg.sampling.presence_penalty) != 0.0
            and not bool(cfg.sampling.fast_presence_penalty)
            and rollout_pending
        ):
            # generate_trajectories_vllm routes a nonzero penalty through
            # extra_args; an engine built without the processor would silently
            # generate unpenalised rollouts. With penalty 0.0 there is nothing
            # to apply and no processor is needed.
            raise RuntimeError(
                "a nonzero presence_penalty requires "
                "sampling.fast_presence_penalty=true "
                "(generate_trajectories_vllm has no native-penalty path)"
            )
        print(
            f"pending: {len(eval_pending)} eval problems ({len(eval_done)} done), "
            f"{len(rollout_pending)} rollout prompts ({len(rollout_done)} done)",
            flush=True,
        )
    else:
        print(f"pending: {len(eval_pending)} eval problems ({len(eval_done)} done)", flush=True)

    wall_started = time.monotonic()
    if eval_pending or rollout_pending:
        memory = GpuMemoryProbe()
        memory.sample()
        llm = build_llm(
            model_path,
            max_model_len=int(cfg.engine.max_model_len),
            gpu_memory_utilization=float(cfg.engine.gpu_memory_utilization),
            seed=int(cfg.seed),
            fast_presence_penalty=bool(cfg.sampling.fast_presence_penalty),
        )
        tokenizer = llm.get_tokenizer()

        if eval_pending:
            run_eval(llm, tokenizer, cfg, args, eval_dir, eval_pending, memory)
        else:
            print("eval: nothing pending", flush=True)
        eval_marker.touch()

        if not args.eval_only:
            if rollout_pending:
                run_rollouts(
                    llm, tokenizer, cfg, args, rollouts_dir, rollout_pending, model_path, memory
                )
            else:
                print("rollouts: nothing pending", flush=True)
            rollout_marker.touch()

        memory.sample()
        if memory.gib():
            print(f"peak GPU memory: {memory.gib():.2f} GiB", flush=True)
    else:
        print("nothing pending; touching markers", flush=True)
        eval_marker.touch()
        if not args.eval_only:
            rollout_marker.touch()

    print(f"stage wall clock: {format_duration(time.monotonic() - wall_started)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
