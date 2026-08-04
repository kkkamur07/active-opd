"""Generate bounded Qwen responses while preserving complete raw traces.

The token budget and context-limit policy are explicit CLI settings.  A
sample is never clipped to fit: if prompt tokens plus the requested
``max_new_tokens`` exceed ``--context-limit``, it is skipped (or the run
fails with ``--on-overflow fail``) and the reason is persisted in JSONL.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .records import iter_records, model_prompt, record_texts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Explicit response-only generation budget; never a trace truncation.",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=4096,
        help="Hard pre-generation prompt+response budget; inputs are never clipped.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--on-overflow",
        choices=("skip", "fail"),
        default="skip",
    )
    parser.add_argument("--output-dir", default="outputs/generated-traces")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0 or args.max_new_tokens <= 0 or args.context_limit <= 0:
        parser.error("limit, max-new-tokens, and context-limit must be positive")
    return args


def _nvidia_smi() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or f"unavailable: {result.stderr.strip()}"


def _length_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": sum(values) / len(values),
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def main() -> int:
    args = _parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to generate on CPU.")
    torch.cuda.set_device(0)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from aopd.data import MathExample
    from aopd.models import GenerationOptions, StudentModel
    from aopd.train import RolloutCollector
    from aopd.utils.reproducibility import seed_everything

    repo_root = Path(__file__).resolve().parent.parent
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs")):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "responses.jsonl"
    student = StudentModel.from_config(config["model"]["student"])
    lengths: list[int] = []
    response_lengths: list[int] = []
    skipped = Counter()
    started = time.perf_counter()
    generated = 0
    try:
        student.load()
        devices = sorted({str(p.device) for p in student.model.parameters()})
        if not devices or any(not device.startswith("cuda") for device in devices):
            raise RuntimeError(f"Student placement is not CUDA-only: {devices}")
        bf16_enabled = bool(torch.cuda.is_bf16_supported())
        generation = GenerationOptions(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=float(config["generation"]["temperature"]),
            top_p=float(config["generation"]["top_p"]),
            top_k=int(config["generation"]["top_k"]),
            repetition_penalty=float(config["generation"]["repetition_penalty"]),
            num_return_sequences=1,
            enable_thinking=True,
        )
        collector = RolloutCollector(
            student,
            config={
                "num_rollouts_per_prompt": 1,
                "max_new_tokens": args.max_new_tokens,
            },
        )
        seed_everything(args.seed)
        with raw_path.open("w", encoding="utf-8") as stream:
            for record_index, row in enumerate(iter_records()):
                if record_index >= args.limit:
                    break
                fields = record_texts(row)
                base = {
                    "record_index": record_index,
                    "problem_id": fields["problem_id"],
                    "prompt": fields["prompt"],
                    "reference": fields["reference"],
                    "trace": fields["trace"],
                    "max_new_tokens": args.max_new_tokens,
                    "context_limit": args.context_limit,
                    "truncation": False,
                }
                if not fields["prompt"]:
                    reason = "missing_prompt"
                    skipped[reason] += 1
                    stream.write(json.dumps({**base, "status": "skipped", "reason": reason}) + "\n")
                    continue
                prompt = model_prompt(str(fields["prompt"]))
                prepared = student.prepare_inputs(prompt, generation=generation)
                prompt_length = int(prepared["attention_mask"].sum().item())
                if prompt_length + args.max_new_tokens > args.context_limit:
                    reason = "prompt_plus_budget_exceeds_context_limit"
                    skipped[reason] += 1
                    stream.write(
                        json.dumps(
                            {
                                **base,
                                "status": "skipped",
                                "reason": reason,
                                "prompt_tokens": prompt_length,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if args.on_overflow == "fail":
                        raise RuntimeError(
                            f"record {record_index} exceeds context policy: "
                            f"{prompt_length}+{args.max_new_tokens}>{args.context_limit}"
                        )
                    continue
                try:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16 if bf16_enabled else torch.float16,
                        enabled=True,
                    ):
                        rollouts = collector.collect(
                            [
                                MathExample(
                                    prompt=prompt,
                                    reference_answer=fields["reference"],
                                    problem_id=fields["problem_id"],
                                )
                            ],
                            generation=generation,
                        )
                except Exception as exc:  # noqa: BLE001 - persist per-record failures
                    reason = f"generation_error:{type(exc).__name__}"
                    skipped[reason] += 1
                    stream.write(
                        json.dumps(
                            {
                                **base,
                                "status": "skipped",
                                "reason": reason,
                                "error": str(exc),
                                "prompt_tokens": prompt_length,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                rollout = rollouts[0]
                generated += 1
                response_tokens = max(
                    0,
                    len(rollout.input_ids) - int(rollout.prompt_length or 0),
                )
                lengths.append(prompt_length)
                response_lengths.append(response_tokens)
                stream.write(
                    json.dumps(
                        {
                            **base,
                            "status": "ok",
                            "prompt_tokens": prompt_length,
                            "response_tokens": response_tokens,
                            "response": rollout.response,
                            "response_trace_complete": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary = {
            "status": "ok",
            "dataset": args.dataset_name,
            "requested_samples": args.limit,
            "generated_samples": generated,
            "skipped_samples": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "prompt_length_tokens": _length_stats(lengths),
            "response_length_tokens": _length_stats(response_lengths),
            "generation_policy": {
                "max_new_tokens": args.max_new_tokens,
                "context_limit": args.context_limit,
                "on_overflow": args.on_overflow,
                "inputs_truncated": False,
                "stored_traces_truncated": False,
            },
            "cuda": {
                "device": torch.cuda.get_device_name(0),
                "devices": devices,
                "bf16_autocast_enabled": bf16_enabled,
                "amp_dtype": "bfloat16" if bf16_enabled else "float16",
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "nvidia_smi": _nvidia_smi(),
            },
            "wall_seconds": time.perf_counter() - started,
            "raw_jsonl": str(raw_path),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        student.unload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
