"""Profile token lengths: stored dataset text vs. sampled student generations.

Each generation is appended to JSONL as it finishes, so a run that dies keeps
the GPU hours it already spent.

Edit the config block below and run: ``python scripts/token_lengths.py``
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from apod.datasets import DATASETS, append_jsonl, examples_from_rows, read_jsonl
from apod.models import STUDENT_ID, generate_trajectories, load_student

# --- config ---------------------------------------------------------------
PROFILE = ["openthoughts", "math500"]  # keys of apod.datasets.DATASETS
NUM_EXAMPLES = 128  # per dataset, clamped to the dataset size
MAX_NEW_TOKENS = 40000  # generation cap; anything hitting it counts as truncated
SEED = 42
OUTPUT_DIR = Path("outputs/token_lengths")
LOG_LEVEL = "INFO"  # DEBUG adds a line per example
LOG_EVERY = 10  # examples between progress lines
PERCENTILES = (50, 80, 95, 99)

# Column holding the reference chain-of-thought / worked solution per dataset.
STORED_COLUMNS = {"openthoughts": "COT_Reason", "math500": "solution"}
# --------------------------------------------------------------------------


def setup_logging() -> None:
    """Console follows LOG_LEVEL; the log file always keeps full DEBUG detail."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL, format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> <level>{message}</level>")
    logger.add(OUTPUT_DIR / "token_lengths.log", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} {level: <7} {message}")

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
        logger.info(f"{label:24}  (no data)")
        return
    percentiles = "  ".join(f"p{p}={stats[f'p{p}']:.1f}" for p in PERCENTILES)
    logger.info(
        f"{label:24}  n={stats['n']}  mean={stats['mean']:.1f}  "
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


def stored_lengths(dataset: str, rows, count: int, tokenizer) -> dict:
    """Token lengths of the reference solutions already in the dataset."""

    column = STORED_COLUMNS[dataset]
    stats = summarize(token_lengths(tokenizer, rows.select(range(count))[column]))
    log_summary(f"{dataset} stored", stats)
    return stats


def generate_lengths(dataset: str, examples, model, tokenizer, out_path: Path) -> list[dict]:
    """Sample one completion per example, appending a row as each one finishes."""

    out_path.unlink(missing_ok=True)
    logger.info("{}: generating {} examples, max_new_tokens={}", dataset, len(examples), MAX_NEW_TOKENS)

    started = time.monotonic()
    total_tokens = 0
    truncated_count = 0

    for position, example in enumerate(examples, start=1):
        example_started = time.monotonic()
        batch = generate_trajectories(
            model, tokenizer, example["prompt"], n=1, max_new_tokens=MAX_NEW_TOKENS
        )
        elapsed = time.monotonic() - example_started

        length = int(batch["response_lengths"][0])
        truncated = bool(batch["truncated"][0])
        total_tokens += length
        truncated_count += truncated
        append_jsonl(
            out_path,
            {
                "id": str(example["id"]),
                "prompt_length": int(batch["prompt_length"]),
                "response_length": length,
                "truncated": truncated,
                "seconds": round(elapsed, 3),
            },
        )
        logger.debug(
            "{}[{}]: {} tokens in {:.1f}s{}",
            dataset, position - 1, length, elapsed, " TRUNCATED" if truncated else "",
        )

        if position % LOG_EVERY == 0 or position == len(examples):
            total_elapsed = time.monotonic() - started
            remaining = (total_elapsed / position) * (len(examples) - position)
            logger.info(
                "{}: {}/{} done  {:.0f} tok/s  mean {:.0f} tok/example  truncated {}  elapsed {}  eta {}",
                dataset, position, len(examples),
                total_tokens / total_elapsed, total_tokens / position, truncated_count,
                format_duration(total_elapsed), format_duration(remaining),
            )

    return read_jsonl(out_path)


def main() -> int:
    setup_logging()
    logger.info("loading student {}", STUDENT_ID)
    tokenizer, model = load_student()
    logger.info("student on {}", next(model.parameters()).device)

    from datasets import load_dataset

    summary: dict[str, dict] = {}
    for dataset in PROFILE:
        name, split = DATASETS[dataset]
        rows = load_dataset(name, split=split).shuffle(seed=SEED)
        count = min(NUM_EXAMPLES, rows.num_rows)
        logger.info("{}: {} rows available, profiling {}", dataset, rows.num_rows, count)

        # Both stages read the same shuffled rows, but examples_from_rows drops
        # any row missing a problem or an answer, so the sets can differ slightly.
        generated = generate_lengths(
            dataset,
            examples_from_rows(rows, n=count),
            model,
            tokenizer,
            OUTPUT_DIR / f"generations-{dataset}.jsonl",
        )
        summary[dataset] = {
            "stored": stored_lengths(dataset, rows, count, tokenizer),
            "prompt": summarize([row["prompt_length"] for row in generated]),
            "generated": summarize([row["response_length"] for row in generated]),
            "truncated": int(sum(row["truncated"] for row in generated)),
        }
        log_summary(f"{dataset} prompt", summary[dataset]["prompt"])
        log_summary(f"{dataset} student CoT", summary[dataset]["generated"])
        logger.info(
            "{}: truncated {} / {} at max_new_tokens={}",
            dataset, summary[dataset]["truncated"], len(generated), MAX_NEW_TOKENS,
        )

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote {}", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
