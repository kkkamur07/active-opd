"""Rollout + eval stage: one vLLM engine session per (arm, round, shard).

Runs the MATH-500 eval of the round's starting model and the round's rollouts
from that same engine as ONE generate stream: eval requests first, then
rollouts (skipped with ``--eval-only`` for the final eval-only round), packed
into ``target_concurrent_sequences``-sized chunks that may hold both, so the
engine never drains at the eval->rollout boundary. Eval and rollout rows still
land in their own files and their own done-markers (``run_session``). Written
against ``docs/pipeline.md``; loads
``<run-dir>/resolved_config.yaml`` with ``OmegaConf.load`` (stages are plain
scripts, not Hydra apps). The driver launches one process per shard with
``CUDA_VISIBLE_DEVICES`` set to that shard's GPU. ``--eval-dataset <name>...``
names the eval set(s) of the session (``conf/eval/<name>.yaml``): one named
set alone evaluates it into ``eval_<name>/`` instead of the MATH-500
``eval/``; several (``--eval-dataset math500 aime2526``, the step-based
driver's every refresh) share the one generate stream, each with its own
files and markers, so the AIME 2025+2026 monitor costs no second engine
session.

Seeds:
  eval     -- deterministic per (round, problem):
              ``seed + eval_seed_offset + round * num_problems + problem_index``
              so re-runs of a round reproduce and rounds differ, independent of
              sharding and resume state.
  rollouts -- base ``seed + round * num_prompts``, plus ``chunk_start`` of
              the prompt's chunk at the rollout-only chunk size
              (``RolloutWriter.seed``); the per-row ``seed`` field records that
              chunk seed. With 8 rounds x 128 prompts the rollout seeds stay
              far below ``eval_seed_offset``, so the two streams never collide.

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
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from apod import paths
from apod.datasets import append_jsonl, read_jsonl, save_npz, write_jsonl
from apod.models import build_llm, render_prompt
from apod.models.generate_vllm import build_sampling_params, generate_requests_vllm, pack_batch
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
    parser.add_argument(
        "--eval-dataset",
        nargs="+",
        default=None,
        help="eval set(s) of this engine session (conf/eval/<name>.yaml keys); "
        "default: cfg.eval.dataset (MATH-500) alone. A non-default set reads "
        "pool/eval_problems_<name>.jsonl and writes eval_<name>/ (see "
        "eval_protocol). Several names share one generate stream, each with "
        "its own files and done-markers: the step-based driver passes the "
        "MATH-500 set and the AIME 2025+2026 monitor together",
    )
    return parse_stage_args(parser, argv)


def eval_protocol(cfg, name: str | None) -> tuple[str, str, Any]:
    """(eval subdir, pool file name, protocol) of one eval set; ``cfg`` untouched.

    The default (``cfg.eval.dataset``) keeps the ``eval/`` +
    ``pool/eval_problems.jsonl`` layout byte-identical. A named set has its
    own protocol (num_problems, num_samples, eval_seed_offset) --
    ``cfg.eval_sets.<name>`` when the launcher stamped one into
    resolved_config.yaml (apod.driver, scripts/terminal_eval.py
    --num-samples), else ``conf/eval/<name>.yaml`` -- and goes to
    ``eval_<name>/`` + ``pool/eval_problems_<name>.jsonl``, which the
    launcher materializes like the driver does for the monitor set.
    """

    if name is None or name == str(cfg.eval.dataset):
        return "eval", "eval_problems.jsonl", cfg.eval
    stamped = cfg.get("eval_sets", {}).get(name)
    if stamped is None:
        conf = Path(__file__).resolve().parents[2] / "conf" / "eval" / f"{name}.yaml"
        if not conf.exists():
            raise FileNotFoundError(f"no eval protocol for {name!r}: {conf}")
        stamped = OmegaConf.load(conf)
    return f"eval_{name}", f"eval_problems_{name}.jsonl", stamped


def select_eval_set(cfg, name: str | None) -> tuple[str, str]:
    """``eval_protocol`` that also swaps ``cfg.eval`` for the set's protocol
    (single-set callers: scripts/terminal_eval.py)."""

    subdir, pool_file, protocol = eval_protocol(cfg, name)
    cfg.eval = protocol
    return subdir, pool_file


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


class EvalWriter:
    """The eval half of one engine session.

    Builds the requests for the pending problems, appends their rows to
    ``eval.shard{K}.jsonl`` as their grades come back, and touches the
    done-marker once the LAST problem is graded -- never earlier, even when
    rollout requests of the same session are still in flight.
    """

    def __init__(self, cfg, args, eval_dir: Path, pending, marker: Path, protocol=None) -> None:
        """``protocol``: the set's eval config (``eval_protocol``); default ``cfg.eval``."""

        self.cfg = cfg
        self.pending = pending
        self.path = eval_dir / f"eval.shard{args.shard}.jsonl"
        self.marker = marker
        protocol = cfg.eval if protocol is None else protocol
        self.name = str(protocol.dataset)
        self.num_samples = int(protocol.num_samples)
        self.seed_base = (
            int(cfg.seed) + int(protocol.eval_seed_offset) + args.round_index * int(protocol.num_problems)
        )
        # (flat completions, verdict futures) per taken problem; consumed by
        # drain() after the NEXT chunk's generate returns (see run_session).
        self.grades: list[tuple[list, Any]] = []
        self.done = 0
        self.n_correct = 0
        self.n_total = 0
        self.n_truncated = 0
        self.response_tokens = 0
        self.finished = False

    def requests(self, tokenizer) -> list[tuple[str, Any]]:
        sampling = _sampling_kwargs(self.cfg)
        return [
            (
                render_prompt(tokenizer, ex["prompt"], enable_thinking=bool(self.cfg.model.enable_thinking)),
                build_sampling_params(
                    n=self.num_samples,
                    # Per (round, problem), independent of sharding and resume.
                    seed=self.seed_base + problem_index,
                    fast_presence_penalty=bool(self.cfg.sampling.fast_presence_penalty),
                    **sampling,
                ),
            )
            for problem_index, ex in self.pending
        ]

    def take(self, index: int, request_output, graders) -> None:
        problem_index, example = self.pending[index]
        flat = [
            (problem_index, example, sample_index, completion)
            for sample_index, completion in enumerate(request_output.outputs)
        ]
        # Math-verify is CPU-bound sympy with signal-based timeouts, so
        # processes, not threads; map submits every item immediately.
        verdicts = graders.map(grade, [str(c.text) for _, _, _, c in flat], repeat(example["answer"]))
        self.grades.append((flat, verdicts))

    def drain(self) -> None:
        for flat, verdicts in self.grades:
            for (problem_index, example, sample_index, completion), verdict in zip(flat, verdicts):
                truncated = completion.finish_reason == "length"
                self.response_tokens += len(completion.token_ids)
                self.n_correct += int(verdict["correct"])
                self.n_total += 1
                self.n_truncated += int(truncated)
                append_jsonl(
                    self.path,
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
            self.done += 1
        self.grades.clear()
        if self.done == len(self.pending) and not self.finished:
            self.finished = True
            accuracy = self.n_correct / self.n_total if self.n_total else 0.0
            mean_length = self.response_tokens / self.n_total if self.n_total else 0.0
            print(
                f"eval {self.name} done: {len(self.pending)} problems  correct {self.n_correct}/{self.n_total} "
                f"({accuracy:.3f})  truncated {self.n_truncated}  mean {mean_length:.0f} tok/sample",
                flush=True,
            )
            self.marker.touch()

    def progress(self) -> str:
        return (
            f"eval {self.name} {self.done}/{len(self.pending)}  correct {self.n_correct}/{self.n_total}  "
            f"truncated {self.n_truncated}"
        )


class RolloutWriter:
    """The rollout half of one engine session.

    Builds the requests for the pending prompts, writes each prompt's npz
    (after EOS repair) the moment its outputs land, appends the graded rows
    to ``trajectories.shard{K}.jsonl``, and touches the done-marker once the
    LAST prompt is graded.
    """

    def __init__(
        self, cfg, args, rollouts_dir: Path, pending, marker: Path, model_path: str, tokenizer
    ) -> None:
        self.cfg = cfg
        self.pending = pending
        self.path = rollouts_dir / f"trajectories.shard{args.shard}.jsonl"
        self.tokens_dir = rollouts_dir / "tokens"
        self.marker = marker
        self.num_rollouts = int(cfg.rollout.num_rollouts)
        # Seeds are per chunk of the pending list at the rollout-only chunk
        # size (target_concurrent_sequences // num_rollouts), exactly what
        # generate_trajectories_vllm derived when rollouts ran as their own
        # generate stream, so sharing chunks with eval requests changes no
        # SamplingParams; the per-row ``seed`` field records that chunk seed.
        self.chunk_size = max(1, int(cfg.engine.target_concurrent_sequences) // self.num_rollouts)
        self.base_seed = int(cfg.seed) + args.round_index * int(cfg.rollout.num_prompts)
        self.eos_ids = collect_eos_ids(tokenizer, model_path)
        self.eos_id = int(tokenizer.eos_token_id)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.eos_id
        # (position, row, batch, verdict futures) per taken prompt; the npz
        # is already on disk, only the jsonl rows wait for drain().
        self.grades: list[tuple[int, dict[str, Any], dict[str, Any], Any]] = []
        self.done = 0
        self.n_correct = 0
        self.n_total = 0
        self.n_truncated = 0
        self.n_repaired = 0
        self.finished = False

    def seed(self, position: int) -> int:
        return self.base_seed + (position // self.chunk_size) * self.chunk_size

    def requests(self, tokenizer) -> list[tuple[str, Any]]:
        sampling = _sampling_kwargs(self.cfg)
        return [
            (
                render_prompt(tokenizer, row["prompt"], enable_thinking=bool(self.cfg.model.enable_thinking)),
                build_sampling_params(
                    n=self.num_rollouts,
                    seed=self.seed(position),
                    fast_presence_penalty=bool(self.cfg.sampling.fast_presence_penalty),
                    **sampling,
                ),
            )
            for position, row in enumerate(self.pending)
        ]

    def take(self, position: int, request_output, graders) -> None:
        row = self.pending[position]
        example_index = int(row["example_index"])
        batch = pack_batch(request_output, row["prompt"], self.pad_id)
        # Repair BEFORE the npz and the jsonl rows so the stored lengths, the
        # stored ids, and what the trainer feeds are one and the same thing.
        self.n_repaired += ensure_trailing_eos(batch, self.eos_ids, self.eos_id, self.pad_id)
        save_npz(self.tokens_dir / f"example_{example_index:05d}.npz", batch)
        self.n_truncated += int(batch["truncated"].sum())
        verdicts = graders.map(grade, [str(r) for r in batch["responses"]], repeat(row["reference"]))
        self.grades.append((position, row, batch, verdicts))

    def drain(self) -> None:
        for position, row, batch, verdicts in self.grades:
            grades = list(verdicts)
            self.n_correct += sum(g["correct"] for g in grades)
            self.n_total += len(grades)
            for rollout_index, verdict in enumerate(grades):
                append_jsonl(
                    self.path,
                    {
                        "example_index": int(row["example_index"]),
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
                        "seed": self.seed(position),
                    },
                )
            self.done += 1
        self.grades.clear()
        if self.done == len(self.pending) and not self.finished:
            self.finished = True
            accuracy = self.n_correct / self.n_total if self.n_total else 0.0
            print(
                f"rollouts done: {len(self.pending)} prompts x {self.num_rollouts}  "
                f"correct {self.n_correct}/{self.n_total} ({accuracy:.3f})  "
                f"truncated {self.n_truncated}  eos repaired {self.n_repaired}",
                flush=True,
            )
            self.marker.touch()

    def progress(self) -> str:
        return (
            f"rollouts {self.done}/{len(self.pending)}  correct {self.n_correct}/{self.n_total}  "
            f"truncated {self.n_truncated}"
        )


def run_session(llm, tokenizer, cfg, writers, graders, memory: GpuMemoryProbe) -> None:
    """One generate stream for every writer's requests, writers in order.

    Eval problems and rollout prompts are packed into the same
    ``target_concurrent_sequences``-sized chunks, so the chunk at the
    eval->rollout boundary holds both and the engine never drains between
    the two. Requests are identical to two separate streams (texts,
    per-request SamplingParams and seeds); only their grouping into
    ``llm.generate`` calls changes.

    Grading overlaps generation: a chunk's grades are submitted the moment
    its generate returns (take) and consumed after the NEXT generate returns
    (drain), so the engine never idles for Math-Verify. The price is one
    chunk of durability: a crash during chunk k+1's generate loses chunk k's
    unwritten rows, which row-level resume regenerates.
    """

    requests: list[tuple[str, Any, Any, int]] = []  # (text, params, writer, key)
    for writer in writers:
        for key, (text, params) in enumerate(writer.requests(tokenizer)):
            requests.append((text, params, writer, key))
    total_sequences = sum(int(params.n) for _, params, _, _ in requests)
    chunks = generate_requests_vllm(
        llm,
        [(text, params) for text, params, _, _ in requests],
        chunk_budget=int(cfg.engine.target_concurrent_sequences),
    )
    started = time.monotonic()
    throughput: dict[str, Any] = {}
    for start, outputs, metrics in chunks:
        memory.sample()
        for writer in writers:
            writer.drain()  # the previous chunk: graded while this one generated
        for offset, request_output in enumerate(outputs):
            _, _, writer, key = requests[start + offset]
            writer.take(key, request_output, graders)
        throughput = metrics["cumulative"]
        wall = time.monotonic() - started
        done = throughput["sequences"]
        eta = (wall / done) * (total_sequences - done) if done else 0.0
        print(
            f"generate  chunk {metrics['chunk_index']}  "
            + "  |  ".join(writer.progress() for writer in writers)
            + f"  |  {metrics['seconds']:.1f}s  "
            f"gen {metrics['generated_tokens_per_s']:.0f} tok/s "
            f"(run {throughput['generated_tokens_per_s']:.0f})  "
            f"total {metrics['total_tokens_per_s']:.0f} tok/s  "
            f"seq {metrics['sequences_per_s']:.2f}/s  "
            f"mean {metrics['mean_generated_tokens_per_sequence']:.0f} tok/seq  "
            f"elapsed {format_duration(wall)}  eta {format_duration(eta)}",
            flush=True,
        )
    for writer in writers:
        writer.drain()
    if throughput:
        print(
            f"generate throughput: {throughput['generated_tokens_per_s']:.0f} generated tok/s  "
            f"{throughput['total_tokens_per_s']:.0f} total tok/s  "
            f"{throughput['sequences_per_s']:.2f} seq/s over "
            f"{format_duration(throughput['seconds'])} of generate "
            f"({format_duration(time.monotonic() - started)} wall)",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    resume = bool(cfg.resume)
    # One (subdir, pool file, protocol) per eval set of this session; the
    # default is cfg.eval alone.
    eval_sets = [eval_protocol(cfg, name) for name in (args.eval_dataset or [None])]

    round_dir = paths.round_dir(run_dir, args.arm, args.round_index)
    rollouts_dir = round_dir / "rollouts"
    for eval_subdir, _, _ in eval_sets:
        (round_dir / eval_subdir).mkdir(parents=True, exist_ok=True)
        (round_dir / eval_subdir / f"done.shard{args.shard}").unlink(missing_ok=True)
    rollout_marker = rollouts_dir / f"done.shard{args.shard}"
    if not args.eval_only:
        rollouts_dir.mkdir(parents=True, exist_ok=True)
        rollout_marker.unlink(missing_ok=True)

    model_path = resolve_model_path(run_dir, args.arm, args.round_index, cfg)
    print(
        f"{args.arm} round {args.round_index} shard {args.shard}/{args.num_shards}: "
        f"model {model_path}" + ("  (eval only)" if args.eval_only else "")
        + "".join(
            f"  eval set {protocol.dataset} -> {subdir}/" for subdir, _, protocol in eval_sets if subdir != "eval"
        ),
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
    eval_work = []  # (subdir, protocol, done count, pending) per eval set
    for eval_subdir, eval_pool_file, protocol in eval_sets:
        problems = read_jsonl(run_dir / "pool" / eval_pool_file)
        if len(problems) != int(protocol.num_problems):
            raise RuntimeError(
                f"pool/{eval_pool_file} has {len(problems)} problems, expected "
                f"num_problems={protocol.num_problems} of eval set {protocol.dataset}; "
                "run the driver (apod.driver / apod.main) or scripts/terminal_eval.py "
                "to materialize the eval set before invoking stages"
            )
        # Intermediate rounds evaluate a PREFIX of the materialized set, so
        # problem_index means the same problem at every round and eval size.
        if args.eval_num_problems is not None:
            problems = problems[: args.eval_num_problems]
        eval_done = load_complete_rows(
            round_dir / eval_subdir / f"eval.shard{args.shard}.jsonl",
            "problem_index", int(protocol.num_samples), resume,
        )
        eval_pending = [
            (i, ex)
            for i, ex in enumerate(problems)
            if i % args.num_shards == args.shard and i not in eval_done
        ]
        eval_work.append((eval_subdir, protocol, len(eval_done), eval_pending))
    eval_pending_total = sum(len(pending) for _, _, _, pending in eval_work)
    eval_done_total = sum(done for _, _, done, _ in eval_work)

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
            # build_sampling_params routes a nonzero penalty through
            # extra_args; an engine built without the processor would silently
            # generate unpenalised rollouts. With penalty 0.0 there is nothing
            # to apply and no processor is needed.
            raise RuntimeError(
                "a nonzero presence_penalty requires "
                "sampling.fast_presence_penalty=true "
                "(build_sampling_params has no native-penalty path)"
            )
        print(
            f"pending: {eval_pending_total} eval problems ({eval_done_total} done), "
            f"{len(rollout_pending)} rollout prompts ({len(rollout_done)} done)",
            flush=True,
        )
    else:
        print(f"pending: {eval_pending_total} eval problems ({eval_done_total} done)", flush=True)

    wall_started = time.monotonic()
    if eval_pending_total or rollout_pending:
        memory = GpuMemoryProbe()
        memory.sample()
        # Fork the grading workers BEFORE the engine exists so the children
        # never inherit a CUDA context; grade() is pure CPU (sympy).
        graders = ProcessPoolExecutor(max_workers=8)
        list(graders.map(grade, [""] * 8, ["0"] * 8))  # force all forks now
        # vLLM admits max_num_seqs=256 by default; a larger concurrency
        # target only fills the engine if that ceiling moves with it.
        concurrency = int(cfg.engine.target_concurrent_sequences)
        engine_extra = {"max_num_seqs": concurrency} if concurrency > 256 else {}
        llm = build_llm(
            model_path,
            max_model_len=int(cfg.engine.max_model_len),
            gpu_memory_utilization=float(cfg.engine.gpu_memory_utilization),
            seed=int(cfg.seed),
            fast_presence_penalty=bool(cfg.sampling.fast_presence_penalty),
            **engine_extra,
        )
        tokenizer = llm.get_tokenizer()

        # Eval requests go first in the stream (sets in the order given); a
        # resumed round whose eval is complete submits rollouts only and
        # never regenerates eval rows.
        writers: list[Any] = []
        for eval_subdir, protocol, _, eval_pending in eval_work:
            eval_marker = round_dir / eval_subdir / f"done.shard{args.shard}"
            if eval_pending:
                writers.append(
                    EvalWriter(cfg, args, round_dir / eval_subdir, eval_pending, eval_marker, protocol=protocol)
                )
            else:
                print(f"eval {protocol.dataset}: nothing pending", flush=True)
                eval_marker.touch()
        if rollout_pending:
            writers.append(
                RolloutWriter(cfg, args, rollouts_dir, rollout_pending, rollout_marker, model_path, tokenizer)
            )
        run_session(llm, tokenizer, cfg, writers, graders, memory)
        if not args.eval_only and not rollout_pending:
            print("rollouts: nothing pending", flush=True)
            rollout_marker.touch()
        graders.shutdown()

        memory.sample()
        if memory.gib():
            print(f"peak GPU memory: {memory.gib():.2f} GiB", flush=True)
    else:
        print("nothing pending; touching markers", flush=True)
        for eval_subdir, _, _ in eval_sets:
            (round_dir / eval_subdir / f"done.shard{args.shard}").touch()
        if not args.eval_only:
            rollout_marker.touch()

    print(f"stage wall clock: {format_duration(time.monotonic() - wall_started)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
