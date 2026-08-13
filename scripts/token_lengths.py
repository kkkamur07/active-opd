"""Profile student generation length: stored dataset solutions vs. vLLM samples.

Answers the two questions that gate collection: how long do the traces run, and
how many of them run into the cap. Each generation is appended to JSONL as it
finishes, so a run that dies keeps the GPU hours it already spent.

One process per GPU. Shard rule is ``example_index % num_shards == shard``,
and each shard writes its own file, so nothing merges by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from apod.datasets import DATASETS, append_jsonl, load_examples, read_jsonl
from apod.models import STUDENT_ID, build_llm, generate_trajectories_vllm

PERCENTILES = (50, 80, 95, 99)

# Column holding the reference chain-of-thought / worked solution per dataset.
STORED_COLUMNS = {"openthoughts": "COT_Reason", "math500": "solution"}


def setup_logging(output_dir: Path, level: str, shard: int) -> None:
    """Console follows ``level``; the log file always keeps full DEBUG detail."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> <level>{message}</level>")
    logger.add(
        output_dir / f"token_lengths.shard{shard}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} {level: <7} {message}",
    )


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def summarize(lengths) -> dict:
    """Mean/median/percentiles for one array of token counts."""

    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0:
        return {"n": 0}
    stats = {
        "n": int(lengths.size),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "max": int(lengths.max()),
    }
    for percentile in PERCENTILES:
        stats[f"p{percentile}"] = float(np.percentile(lengths, percentile))
    return stats


def log_summary(label: str, stats: dict) -> None:
    if not stats["n"]:
        logger.info(f"{label:26}  (no data)")
        return
    percentiles = "  ".join(f"p{p}={stats[f'p{p}']:.1f}" for p in PERCENTILES)
    logger.info(
        f"{label:26}  n={stats['n']}  mean={stats['mean']:.1f}  "
        f"median={stats['median']:.1f}  {percentiles}  max={stats['max']}"
    )


def token_lengths(tokenizer, texts, batch_size: int = 64) -> np.ndarray:
    out = np.empty(len(texts), dtype=np.int32)
    for start in range(0, len(texts), batch_size):
        batch = [text or "" for text in texts[start : start + batch_size]]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False)
        for offset, ids in enumerate(encoded["input_ids"]):
            out[start + offset] = len(ids)
    return out


def stored_lengths(dataset: str, tokenizer, count: int, seed: int) -> dict:
    """Token lengths of the reference solutions already in the dataset."""

    from datasets import load_dataset

    column = STORED_COLUMNS.get(dataset)
    name, split = DATASETS[dataset]
    rows = load_dataset(name, split=split).shuffle(seed=seed)
    if column not in rows.column_names:
        logger.warning("{}: no stored-solution column {!r}, skipping", dataset, column)
        return {"n": 0}
    count = min(count, rows.num_rows)
    stats = summarize(token_lengths(tokenizer, rows.select(range(count))[column]))
    log_summary(f"{dataset} stored", stats)
    return stats


def generate_lengths(dataset: str, indexed_examples, llm, tokenizer, out_path: Path, args) -> list[dict]:
    """Sample ``num_rollouts`` completions per example, appending as they land."""

    out_path.unlink(missing_ok=True)
    logger.info(
        "{}: generating {} examples x {} rollouts, max_new_tokens={}, presence_penalty={}",
        dataset, len(indexed_examples), args.num_rollouts, args.max_new_tokens, args.presence_penalty,
    )

    started = time.monotonic()
    total_tokens = 0
    truncated_count = 0
    rollouts = 0

    batches = generate_trajectories_vllm(
        llm,
        tokenizer,
        [example["prompt"] for _, example in indexed_examples],
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
        example_index, example = indexed_examples[position]
        for rollout_index, length in enumerate(batch["response_lengths"]):
            truncated = bool(batch["truncated"][rollout_index])
            total_tokens += int(length)
            truncated_count += truncated
            rollouts += 1
            append_jsonl(
                out_path,
                {
                    "example_index": example_index,
                    "rollout_index": rollout_index,
                    "id": str(example["id"]),
                    "prompt_length": int(batch["prompt_length"]),
                    "response_length": int(length),
                    "truncated": truncated,
                    "finish_reason": str(batch["finish_reasons"][rollout_index]),
                },
            )

        done = position + 1
        if done % args.log_every == 0 or done == len(indexed_examples):
            elapsed = time.monotonic() - started
            remaining = (elapsed / done) * (len(indexed_examples) - done)
            logger.info(
                "{}: {}/{} prompts  {:.0f} tok/s  mean {:.0f} tok/trace  truncated {}/{}  elapsed {}  eta {}",
                dataset, done, len(indexed_examples),
                total_tokens / max(elapsed, 1e-9), total_tokens / max(rollouts, 1),
                truncated_count, rollouts,
                format_duration(elapsed), format_duration(remaining),
            )

    return read_jsonl(out_path)


def parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), default=["openthoughts"])
    parser.add_argument("--num-examples", type=int, default=64)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=8, help="prompts per llm.generate call")
    parser.add_argument("--output-dir", default="outputs/token_lengths")
    parser.add_argument("--model", default=STUDENT_ID)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--skip-stored", action="store_true", help="skip the dataset-solution length stats")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-every", type=int, default=4, help="prompts between progress lines")
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.num_shards:
        parser.error(f"--shard must be in [0, {args.num_shards}); got {args.shard}")
    if args.max_new_tokens > args.max_model_len:
        parser.error(f"--max-new-tokens {args.max_new_tokens} exceeds --max-model-len {args.max_model_len}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    setup_logging(out, args.log_level, args.shard)
    logger.info("shard {}/{}  args {}", args.shard, args.num_shards, vars(args))

    llm = build_llm(
        args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    summary: dict[str, dict] = {}
    for dataset in args.dataset:
        # Every shard builds the identical example list, then keeps its slice.
        examples = load_examples(dataset, n=args.num_examples, seed=args.seed)
        mine = [(i, ex) for i, ex in enumerate(examples) if i % args.num_shards == args.shard]
        logger.info("{}: {} examples selected, {} on this shard", dataset, len(examples), len(mine))

        generated = generate_lengths(
            dataset, mine, llm, tokenizer,
            out / f"generations-{dataset}.shard{args.shard}.jsonl", args,
        )
        summary[dataset] = {
            "prompt": summarize([row["prompt_length"] for row in generated]),
            "generated": summarize([row["response_length"] for row in generated]),
            "truncated": int(sum(row["truncated"] for row in generated)),
            "rollouts": len(generated),
        }
        if not args.skip_stored and args.shard == 0:
            summary[dataset]["stored"] = stored_lengths(dataset, tokenizer, args.num_examples, args.seed)

        log_summary(f"{dataset} prompt", summary[dataset]["prompt"])
        log_summary(f"{dataset} student CoT", summary[dataset]["generated"])
        logger.info(
            "{}: truncated {} / {} at max_new_tokens={}",
            dataset, summary[dataset]["truncated"], len(generated), args.max_new_tokens,
        )

    path = out / f"summary.shard{args.shard}.json"
    path.write_text(json.dumps({"args": vars(args), "datasets": summary}, indent=2, default=str))
    logger.info("wrote {}", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
