"""A/B: MTP speculative decoding vs plain decoding for the 4B student (2026-11-13).

Same requests (first N MATH-500 problems x 4 samples, 16k cap, the run's
sampling: T=1.0, top_p 0.95, top_k 20, per-request seeds), same engine
settings; GPU 0 plain, GPU 1 MTP (method "mtp", k draft tokens). Speculative
decoding with rejection sampling leaves the output distribution unchanged; the
check here is speed (generated tok/s), and that accuracy / length agree
within noise.

  python scripts/spec_ab.py --n 80 --k 2        # launches both sides
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
OUT = REPO_ROOT / "outputs" / "runs" / "spec-ab-4b"


def side(n: int, k: int, spec: bool, conc: int) -> None:
    from apod.datasets.io import read_jsonl
    from apod.models import build_llm, render_prompt
    from apod.models.generate_vllm import build_sampling_params
    from apod.verification import grade

    problems = read_jsonl(REPO_ROOT / "outputs/runs/kl50/pool/eval_problems.jsonl")[:n]
    extra = {"max_num_seqs": conc}
    if spec:
        extra["speculative_config"] = {"method": "mtp", "num_speculative_tokens": k}
    llm = build_llm("Qwen/Qwen3.5-4B", max_model_len=18432, gpu_memory_utilization=0.92, seed=42,
                    fast_presence_penalty=False, **extra)
    tok = llm.get_tokenizer()
    texts = [render_prompt(tok, p["prompt"], enable_thinking=True) for p in problems]
    params = [build_sampling_params(n=4, seed=42 + 100000 + i, fast_presence_penalty=False, max_tokens=16384,
                                    temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0.0)
              for i in range(n)]
    t0 = time.time()
    outs = llm.generate(texts, params)
    wall = time.time() - t0
    gen = sum(len(c.token_ids) for o in outs for c in o.outputs)
    correct = [grade(c.text, p["answer"]) for o, p in zip(outs, problems) for c in o.outputs]
    res = {"spec": spec, "k": k if spec else 0, "n_problems": n, "samples": 4 * n, "wall_s": round(wall, 1),
           "generated_tokens": gen, "gen_tok_per_s": round(gen / wall, 1),
           "mean_len": round(gen / (4 * n), 1),
           "truncated": sum(c.finish_reason == "length" for o in outs for c in o.outputs),
           "accuracy_strict": round(sum(bool(x["correct"]) and bool(x["has_boxed"]) for x in correct) / len(correct), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{'mtp' if spec else 'plain'}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--conc", type=int, default=160)
    p.add_argument("--side", choices=["plain", "mtp"])
    a = p.parse_args()
    if a.side:
        return side(a.n, a.k, a.side == "mtp", a.conc)
    OUT.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu, s in ((0, "plain"), (1, "mtp")):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1"}
        log = open(OUT / f"{s}.log", "w")
        procs.append(subprocess.Popen([sys.executable, __file__, "--side", s, "--n", str(a.n), "--k", str(a.k),
                                       "--conc", str(a.conc)], env=env, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT))
    print("exit codes", [pr.wait() for pr in procs])
    for s in ("plain", "mtp"):
        f = OUT / f"{s}.json"
        print(s, f.read_text() if f.exists() else "FAILED (see log)")


if __name__ == "__main__":
    main()
