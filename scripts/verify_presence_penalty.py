"""Prove token-identity of the incremental presence-penalty processor.

The optimized processor (apod/models/presence_penalty.py) claims *exact*
equivalence with vLLM's native ``presence_penalty`` path: same subtraction on
the same logits, just O(batch) bookkeeping instead of an O(batch x history)
mask rebuild per step. This script tests that claim empirically before the
experiment relies on it:

  A. penalty=1.5, fast (our processor via extra_args, native field 0.0)
  B. penalty=1.5, native (vLLM's presence_penalty field, no processor)
  C. penalty=0.0, fast   (processor registered on the engine but idle)
  D. penalty=0.0, native (no processor registered at all)

A==B is the equivalence check; C==D is the no-op check (registering the
processor with no penalty must not perturb anything).

Each run happens in a fresh subprocess so every configuration gets its own
CUDA context and engine: same engine seed, same per-request seeds, same
submission order -> the only variable is the penalty path. Token ids per
(prompt, sample) are compared exactly.

Usage (GPU 0):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/verify_presence_penalty.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-2B"
NUM_PROMPTS = 8
NUM_SAMPLES = 2
MAX_TOKENS = 512
POOL_SEED = 123
ENGINE_SEED = 7
REQUEST_SEED = 7

RUNS = {
    "A": {"penalty": 1.5, "fast": True},
    "B": {"penalty": 1.5, "fast": False},
    "C": {"penalty": 0.0, "fast": True},
    "D": {"penalty": 0.0, "fast": False},
}


def worker(penalty: float, fast: bool, out_path: Path) -> None:
    """Build one engine, generate, dump token ids + throughput as JSON."""

    import time

    from transformers import AutoTokenizer

    from apod.datasets.load import load_examples
    from apod.models.generate_vllm import build_llm, build_sampling_params, render_prompt

    examples = load_examples("openthoughts", n=NUM_PROMPTS, seed=POOL_SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    texts = [render_prompt(tokenizer, ex["prompt"]) for ex in examples]

    llm = build_llm(
        MODEL_ID,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        seed=ENGINE_SEED,
        fast_presence_penalty=fast,
    )
    params = build_sampling_params(
        n=NUM_SAMPLES,
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=penalty,
        seed=REQUEST_SEED,
        fast_presence_penalty=fast,
    )

    began = time.perf_counter()
    outputs = llm.generate(texts, params)
    seconds = time.perf_counter() - began

    tokens = {}
    generated = 0
    for p, request_output in enumerate(outputs):
        for s, completion in enumerate(request_output.outputs):
            ids = list(completion.token_ids)
            tokens[f"{p}:{s}"] = ids
            generated += len(ids)

    out_path.write_text(
        json.dumps(
            {
                "penalty": penalty,
                "fast": fast,
                "tokens": tokens,
                "generated_tokens": generated,
                "seconds": seconds,
                "tok_per_s": generated / max(seconds, 1e-9),
            }
        )
    )


def compare(label: str, left: dict, right: dict, names: tuple[str, str]) -> bool:
    """Exact per-(prompt, sample) token-id comparison; diagnose any divergence."""

    ok = True
    for key in sorted(left["tokens"], key=lambda k: tuple(map(int, k.split(":")))):
        a, b = left["tokens"][key], right["tokens"][key]
        if a == b:
            continue
        ok = False
        step = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(
            f"  MISMATCH (prompt:sample {key}): first divergent step {step} "
            f"({names[0]} len {len(a)} tok {a[step] if step < len(a) else '<end>'} vs "
            f"{names[1]} len {len(b)} tok {b[step] if step < len(b) else '<end>'}); "
            f"identical prefix of {step} tokens"
        )
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {label} — {len(left['tokens'])} sequences compared token-for-token")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--penalty", type=float)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.worker:
        worker(args.penalty, args.fast, args.out)
        return 0

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "verify_presence_penalty"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, spec in RUNS.items():
        out_path = out_dir / f"run_{name}.json"
        print(f"\n=== run {name}: penalty={spec['penalty']} fast={spec['fast']} ===", flush=True)
        cmd = [
            sys.executable, __file__, "--worker",
            "--penalty", str(spec["penalty"]), "--out", str(out_path),
        ]
        if spec["fast"]:
            cmd.append("--fast")
        subprocess.run(cmd, check=True)
        results[name] = json.loads(out_path.read_text())
        print(
            f"run {name}: {results[name]['generated_tokens']} tokens in "
            f"{results[name]['seconds']:.1f}s = {results[name]['tok_per_s']:.0f} tok/s"
        )

    print("\n=== comparisons ===")
    ok_ab = compare("A==B (penalty=1.5, fast vs native)", results["A"], results["B"], ("fast", "native"))
    ok_cd = compare("D==C (penalty=0.0, idle processor vs none)", results["C"], results["D"], ("idle-proc", "no-proc"))
    print(
        f"\nthroughput: fast {results['A']['tok_per_s']:.0f} tok/s vs "
        f"native {results['B']['tok_per_s']:.0f} tok/s "
        f"(short {MAX_TOKENS}-token traces; the native path's cost grows with history length)"
    )
    return 0 if (ok_ab and ok_cd) else 1


if __name__ == "__main__":
    sys.exit(main())
