"""Profile full OpenThoughts prompt/reference/trace lengths.

This profiler intentionally uses ``truncation=False``.  ``max_new_tokens``
limits newly generated response tokens only; it must never be applied while
profiling or persisting an existing dataset trace.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .records import iter_records, record_texts


def _token_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    return len(encoded["input_ids"])


def _quantiles(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {name: None for name in ("min", "mean", "median", "p90", "p95", "p99", "max")}
    ordered = sorted(values)

    def percentile(percent: float) -> int:
        index = max(0, min(len(ordered) - 1, math.ceil(percent * len(ordered)) - 1))
        return ordered[index]

    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="outputs/trace-profile")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> int:
    args = _parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "records.jsonl"
    lengths: dict[str, list[int]] = {
        "prompt_tokens": [],
        "reference_tokens": [],
        "trace_tokens": [],
        "prompt_plus_trace_tokens": [],
    }
    included = 0
    scanned = 0
    skipped = 0
    trace_over_1024 = 0
    with trace_path.open("w", encoding="utf-8") as stream:
        # Use a bounded non-streaming dataset slice.  This keeps the measured
        # records/order and full text intact while avoiding the installed
        # datasets streaming worker's interpreter-shutdown crash.
        for row in iter_records(streaming=False, limit=args.limit):
            scanned += 1
            fields = record_texts(row)
            reason = None
            if not fields["prompt"]:
                reason = "missing_prompt"
            elif not fields["trace"]:
                reason = "missing_trace"
            if reason is not None:
                skipped += 1
                stream.write(
                    json.dumps(
                        {
                            "status": "skipped",
                            "record_index": scanned - 1,
                            "problem_id": fields["problem_id"],
                            "reason": reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if scanned >= args.limit:
                    break
                continue

            prompt_tokens = _token_length(tokenizer, str(fields["prompt"]))
            reference_tokens = _token_length(
                tokenizer,
                str(fields["reference"] or ""),
            )
            trace_tokens = _token_length(tokenizer, str(fields["trace"]))
            combined_tokens = _token_length(
                tokenizer,
                f"{fields['prompt']}\n\n{fields['trace']}",
            )
            lengths["prompt_tokens"].append(prompt_tokens)
            lengths["reference_tokens"].append(reference_tokens)
            lengths["trace_tokens"].append(trace_tokens)
            lengths["prompt_plus_trace_tokens"].append(combined_tokens)
            trace_over_1024 += int(trace_tokens > 1024)
            included += 1
            stream.write(
                json.dumps(
                    {
                        "status": "ok",
                        "record_index": scanned - 1,
                        "problem_id": fields["problem_id"],
                        "prompt": fields["prompt"],
                        "reference": fields["reference"],
                        "trace": fields["trace"],
                        "prompt_tokens": prompt_tokens,
                        "reference_tokens": reference_tokens,
                        "trace_tokens": trace_tokens,
                        "prompt_plus_trace_tokens": combined_tokens,
                        "trace_exceeds_1024": trace_tokens > 1024,
                        "truncation": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if scanned >= args.limit:
                break

    summary = {
        "status": "ok",
        "dataset": args.dataset_name,
        "split": "train",
        "model_id": args.model_id,
        "requested_record_limit": args.limit,
        "records_scanned": scanned,
        "records_included": included,
        "records_skipped": skipped,
        "length_statistics_tokens": {
            name: _quantiles(values) for name, values in lengths.items()
        },
        "trace_count_exceeding_1024": trace_over_1024,
        "trace_fraction_exceeding_1024": (
            trace_over_1024 / included if included else 0.0
        ),
        "truncation_policy": {
            "dataset_trace_truncation": False,
            "tokenizer_truncation": False,
            "max_new_tokens_note": (
                "max_new_tokens applies only to newly generated responses, "
                "never to stored dataset traces."
            ),
        },
        "records_jsonl": str(trace_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
