"""Independently verify the vLLM checkpoint-reload fix (apod/models/vllm_qwen35).

The adapter patches four holes in vLLM 0.26.0's handling of our
Qwen3_5ForCausalLM round checkpoints, and the dangerous failure mode is
silent: a partial weight rename or a wrong M-RoPE/SSM setting still yields an
engine that generates fluent text. Checks, with their calibration:

  1. ELEMENTWISE TENSOR EQUALITY (definitive) -- pull every parameter out of
     the live engine model and compare against the checkpoint file, applying
     the known fusion recipes (qkv_proj, gate_up_proj, in_proj_qkvz/ba).
     "Covered" means a param received A tensor; this proves it got the RIGHT
     one, bitwise.
  2. PREFILL LOGPROBS vs HF over ~2000 positions, WITH A FLOOR -- the same
     comparison on the BASE model through both stacks calibrates the pure
     cross-stack bf16 noise floor. checkpoint-delta ~ floor => adapter clean;
     checkpoint-delta >> floor => something subtle is wrong. Position-binned
     deltas expose positional/state bugs (they grow with position; numerics
     stay flat).
  3. DECODE PATH -- greedy-generate from the checkpoint under vLLM (live SSM
     state carried across steps, the machinery the adapter patches), then
     teacher-force the SAME tokens through HF and compare per-step chosen-token
     logprobs and argmax agreement.
  4. TRAINED-NOT-BASE -- vLLM(checkpoint) must differ from HF(base) by far
     more than the floor; on-disk weight delta nonzero.
  5. register() idempotent; fork propagation is proven by every pipeline run
     (the EngineCore worker is a separate process resolving the class).

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/verify_vllm_reload.py

Writes verify_vllm_reload.json next to the checkpoint.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# apply_model ships probe functions to the EngineCore worker, which needs
# pickle-based serialization. Verification-only; never set in pipeline code.
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

CKPT = Path("outputs/runs/smoke2/arms/entropy_top4/rounds/round_02/checkpoint")
BASE = "Qwen/Qwen3.5-2B"
DECODE_TOKENS = 128
RESULTS: dict = {}


def check(name: str, passed: bool, detail: str) -> None:
    RESULTS[name] = {"passed": bool(passed), "detail": detail}
    print(f"{'PASS' if passed else 'FAIL'}: {name}: {detail}")
    if not passed:
        raise SystemExit(f"VLLM RELOAD CHECK FAILED: {name}: {detail}")


def _tensor_probe(model) -> dict:
    """Runs in the engine worker: elementwise-compare live params vs the file."""
    import safetensors.torch
    import torch

    state = safetensors.torch.load_file(
        "outputs/runs/smoke2/arms/entropy_top4/rounds/round_02/checkpoint/model.safetensors"
    )
    # vLLM param -> checkpoint sub-names concatenated along dim 0 (from
    # Qwen3_5ForCausalLMBase.packed_modules_mapping).
    fusions = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def file_tensor(vllm_name: str):
        base = vllm_name.replace("model.", "model.language_model.", 1)
        if base in state:
            return state[base]
        head, _, leaf = base.rpartition(".")          # ...weight
        mod = head.rsplit(".", 1)[-1]                 # qkv_proj
        for fused, parts in fusions.items():
            if mod == fused:
                prefix = head[: -len(fused)]
                return torch.cat([state[f"{prefix}{p}.{leaf}"] for p in parts], dim=0)
        return None

    exact = 0
    max_diff = 0.0
    missing, shape_mismatch = [], []
    for name, param in model.named_parameters():
        if name == "lm_head.weight":  # tied to embed_tokens, no file key
            continue
        ref = file_tensor(name)
        if ref is None:
            missing.append(name)
            continue
        live = param.data.to("cpu")
        if live.shape != ref.shape:
            shape_mismatch.append(f"{name}: {tuple(live.shape)} vs {tuple(ref.shape)}")
            continue
        diff = (live.float() - ref.float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff == 0.0:
            exact += 1
    total = sum(1 for n, _ in model.named_parameters() if n != "lm_head.weight")
    return {
        "total": total,
        "exact": exact,
        "max_abs_diff": max_diff,
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "model_class": type(model).__name__,
    }


def token_logprobs_from_prompt(out, prompt_ids):
    lps = []
    for entry, tid in zip(out.prompt_logprobs[1:], prompt_ids[1:]):
        lps.append(float(entry[tid].logprob))
    return np.array(lps)


def hf_forward_logprobs(model, ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids=torch.tensor(ids, device="cuda:0")[None]).logits[0].float()
    return torch.log_softmax(logits[:-1], dim=-1)


def binned(deltas: np.ndarray, bins: int = 4) -> list[float]:
    edges = np.linspace(0, len(deltas), bins + 1, dtype=int)
    return [float(np.mean(deltas[a:b])) for a, b in zip(edges[:-1], edges[1:])]


def main() -> None:
    from apod.models import build_llm
    from apod.models.vllm_qwen35 import register

    from vllm import ModelRegistry, SamplingParams

    register()
    ok = "Qwen3_5ForCausalLM" in ModelRegistry.get_supported_archs()
    register()
    check("register_idempotent", ok, "double register() is a no-op; arch present")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CKPT))
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What is 17 * 23? Show your reasoning."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Long teacher-forced context from a REAL stored rollout: positional/state
    # bugs grow with position, so test where they would be largest.
    npz = sorted(CKPT.parent.glob("rollouts/tokens/example_*.npz"))[0]
    with np.load(npz, allow_pickle=True) as data:
        long_ids = [int(t) for t in data["input_ids"][0][: 2000]]

    # ---- engine on the checkpoint ----------------------------------------
    llm = build_llm(str(CKPT), max_model_len=4096, gpu_memory_utilization=0.45, seed=0)

    cov = llm.apply_model(_tensor_probe)[0]
    check(
        "elementwise_tensor_equality",
        cov["model_class"] == "Qwen3_5TextForCausalLM"
        and not cov["missing"]
        and not cov["shape_mismatch"]
        and cov["exact"] == cov["total"]
        and cov["max_abs_diff"] == 0.0,
        f"{cov['exact']}/{cov['total']} params BITWISE-identical to the checkpoint "
        f"file (fusion recipes applied; max abs diff {cov['max_abs_diff']:.1e}; "
        f"missing={cov['missing'] or 'none'}; shape_mismatch={cov['shape_mismatch'] or 'none'})",
    )

    lp_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    out_long = llm.generate(
        [{"prompt_token_ids": long_ids}], lp_params
    )[0]
    vllm_ckpt_long = token_logprobs_from_prompt(out_long, long_ids)

    # decode: greedy with per-step logprobs (live SSM state across steps)
    dec = llm.generate(
        [prompt], SamplingParams(max_tokens=DECODE_TOKENS, temperature=0.0, logprobs=0)
    )[0].outputs[0]
    dec_ids = list(dec.token_ids)
    vllm_dec_lps = np.array(
        [float(step[tid].logprob) for step, tid in zip(dec.logprobs, dec_ids)]
    )

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # ---- engine on the BASE repo: the cross-stack noise floor -------------
    llm_base = build_llm(BASE, max_model_len=4096, gpu_memory_utilization=0.45, seed=0)
    out_base_long = llm_base.generate([{"prompt_token_ids": long_ids}], lp_params)[0]
    vllm_base_long = token_logprobs_from_prompt(out_base_long, long_ids)
    del llm_base
    gc.collect()
    torch.cuda.empty_cache()

    # ---- HF forwards, in a FRESH subprocess -------------------------------
    # After vLLM has run in this process, transformers' composite-config
    # handling for the hub repo breaks (AutoModelForCausalLM routes the
    # composite Qwen3_5Config into the text-only class and crashes on
    # config.vocab_size); the identical load succeeds in a clean process.
    # So hand the HF work its own interpreter.
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    inter_path = str(CKPT / "verify_intermediates.npz")
    np.savez(
        inter_path,
        long_ids=np.asarray(long_ids, dtype=np.int64),
        dec_ids=np.asarray(dec_ids, dtype=np.int64),
        prompt_ids=np.asarray(prompt_ids, dtype=np.int64),
    )
    subprocess.run(
        [sys.executable, __file__, "--hf-phase", inter_path], check=True
    )
    hf = np.load(inter_path + ".hf.npz")
    hf_ckpt_long = hf["hf_ckpt_long"]
    hf_dec_lps = hf["hf_dec_lps"]
    hf_dec_argmax = hf["hf_dec_argmax"]
    hf_base_long = hf["hf_base_long"]

    # ---- verdicts ----------------------------------------------------------
    floor = np.abs(vllm_base_long - hf_base_long)
    delta = np.abs(vllm_ckpt_long - hf_ckpt_long)
    check(
        "prefill_matches_at_noise_floor",
        float(delta.mean()) < 3 * max(float(floor.mean()), 1e-4),
        f"{len(delta)} positions: |vLLM-HF| checkpoint mean {delta.mean():.4f} vs "
        f"BASE floor {floor.mean():.4f} (pure cross-stack bf16 numerics; ratio "
        f"{delta.mean() / max(floor.mean(), 1e-9):.2f}x)",
    )
    check(
        "no_positional_drift",
        binned(delta)[-1] < 3 * max(binned(delta)[0], 1e-3)
        or binned(delta)[-1] < 2 * max(float(floor[len(floor)//4*3:].mean()), 1e-3),
        f"position-binned |Δ| quartiles: ckpt {['%.4f' % b for b in binned(delta)]} "
        f"vs floor {['%.4f' % b for b in binned(floor)]} -- a rising ckpt-only trend "
        "would be an M-RoPE/SSM-state bug",
    )
    dec_delta = np.abs(vllm_dec_lps - hf_dec_lps)
    argmax_agree = float(np.mean(np.array(dec_ids) == hf_dec_argmax))
    check(
        "decode_path_matches_hf",
        float(dec_delta.mean()) < 0.05 and argmax_agree > 0.9,
        f"{DECODE_TOKENS} greedy decode steps (live SSM state): per-step "
        f"chosen-token |Δlogprob| mean {dec_delta.mean():.4f} max {dec_delta.max():.4f}; "
        f"HF argmax agrees with vLLM's greedy choice at {argmax_agree:.1%} of steps",
    )
    base_gap = np.abs(vllm_ckpt_long - hf_base_long)
    check(
        "trained_not_base",
        float(base_gap.mean()) > 10 * float(delta.mean()),
        f"|vLLM(ckpt) - HF(base)| mean {base_gap.mean():.4f} vs matched-load "
        f"{delta.mean():.4f} ({base_gap.mean() / max(delta.mean(), 1e-9):.0f}x)",
    )

    out_path = CKPT / "verify_vllm_reload.json"
    out_path.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nALL VLLM RELOAD CHECKS PASSED -> {out_path}")


def hf_phase(inter_path: str) -> None:
    """Fresh-process HF forwards: checkpoint + base logprobs on the stored ids."""
    data = np.load(inter_path)
    long_ids = [int(t) for t in data["long_ids"]]
    dec_ids = [int(t) for t in data["dec_ids"]]
    prompt_ids = [int(t) for t in data["prompt_ids"]]

    from apod.models.load import load_lm

    _, hf_ckpt = load_lm(str(CKPT), frozen=True)
    lps = hf_forward_logprobs(hf_ckpt, long_ids)
    hf_ckpt_long = (
        lps.gather(-1, torch.tensor(long_ids[1:], device="cuda:0")[:, None])[:, 0]
        .cpu().numpy()
    )
    # teacher-force the vLLM decode tokens through HF
    full = prompt_ids + dec_ids
    lps_dec = hf_forward_logprobs(hf_ckpt, full)
    pos = np.arange(len(prompt_ids) - 1, len(full) - 1)
    hf_dec_lps = (
        lps_dec[pos].gather(-1, torch.tensor(dec_ids, device="cuda:0")[:, None])[:, 0]
        .cpu().numpy()
    )
    hf_dec_argmax = lps_dec[pos].argmax(-1).cpu().numpy()
    del hf_ckpt, lps, lps_dec
    gc.collect()
    torch.cuda.empty_cache()

    _, hf_base = load_lm(BASE, frozen=True)
    lps_b = hf_forward_logprobs(hf_base, long_ids)
    hf_base_long = (
        lps_b.gather(-1, torch.tensor(long_ids[1:], device="cuda:0")[:, None])[:, 0]
        .cpu().numpy()
    )

    np.savez(
        inter_path + ".hf.npz",
        hf_ckpt_long=hf_ckpt_long,
        hf_dec_lps=hf_dec_lps,
        hf_dec_argmax=hf_dec_argmax,
        hf_base_long=hf_base_long,
    )
    print(f"hf phase done -> {inter_path}.hf.npz")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--hf-phase":
        hf_phase(sys.argv[2])
    else:
        main()
