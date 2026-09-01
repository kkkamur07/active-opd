# Fused Gated DeltaNet kernels for the HF-side stages

Qwen3.5-2B and -9B run 18 of their 24 layers as Gated DeltaNet linear
attention (`layer_types`: 3 linear : 1 full). vLLM ships its own Triton GDN
kernels, so rollouts and eval are unaffected. Every stage that runs the model
through transformers instead (`apod/stages/train.py`, `scripts/oracle_kl.py`,
`apod/stages/entropy.py`, `scripts/kl_drift.py`, the LR probes) uses whatever
transformers resolves for those layers, and today that is the pure-torch
reference loop (`docs/perf_review.md`, finding 1).

## How transformers picks the implementation (transformers 5.15.0, verified)

`transformers/models/qwen3_5/modeling_qwen3_5.py` decorates the GDN math with
`use_kernel_func_from_hub_with_fallback(<func>, <package>)`
(`transformers/integrations/hub_kernels.py:764`). The decorator runs once, at
module import, and binds the implementation in this order:

1. Hub kernels (`kernels` package, `use_kernels=True` at model load, needs
   Hub access) -- not installed and every launch sets `HF_HUB_OFFLINE=1`, so
   this layer is inert.
2. `importlib.import_module(<package>)` and the function at the mapped
   internal path (`fla.ops.gated_delta_rule.chunk_gated_delta_rule`,
   `causal_conv1d.causal_conv1d_fn`).
3. The decorated torch function itself.

| decorated function | package | resolves with the `kernels` extra |
|---|---|---|
| `torch_chunk_gated_delta_rule` (every prefill/training forward) | `fla` | **yes** -> `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule` |
| `torch_recurrent_gated_delta_rule` (single-token cached decode only) | `fla` | no (fla 0.5.2 names it `fused_recurrent_gated_delta_rule`); irrelevant, no stage decodes through HF |
| `causal_conv1d_fn` / `causal_conv1d_update` | `causal_conv1d` | no; deliberately not installed (see below) |

The pip name for `fla` is `flash-linear-attention` (meta-package) plus
`fla-core` (the kernels). Both are `py3-none-any` wheels: pure Python +
Triton, no nvcc, no ABI. Requirements (`torch>=2.7`, `triton>=3.3`,
`einops`) are already satisfied by the cu130 venv; a dry-run resolve against
it adds exactly `fla-core==0.5.2` and `flash-linear-attention==0.5.2`.

`causal-conv1d` only ships an sdist (1.7.0) and needs nvcc against the
mixed cu130 header set (`docs/guide.md`); the depthwise conv fallback is
cheap, so it is not part of the extra.

## Install (after the sweep ends -- never while stages are running)

```bash
uv sync --extra vllm --extra train --extra kernels
```

`pyproject.toml` declares the extra and `uv.lock` already resolves it, so
this is offline-safe once the two wheels are in the uv cache (they are
fetched on first sync). Every stage subprocess imports transformers fresh,
so a launch that starts after the sync picks the kernels up automatically;
nothing in `conf/` or the scripts changes.

## Verify the fused path is active

```python
import inspect
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
for fn in (m.torch_chunk_gated_delta_rule, m.torch_recurrent_gated_delta_rule, m.causal_conv1d_fn):
    impl = inspect.getclosurevars(fn).nonlocals["implementation"]
    print(f"{fn.__name__:35s} -> {impl.__module__}.{impl.__qualname__}")
```

Run it with the venv's interpreter (`HF_HUB_OFFLINE=1 .venv/bin/python
check.py`). Expected after the install:

```
torch_chunk_gated_delta_rule        -> fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule
torch_recurrent_gated_delta_rule    -> transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule
causal_conv1d_fn                    -> transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn
```

Before the install every line ends in `modeling_qwen3_5.<same name>`. The
check works on CPU (it inspects the binding, it does not run a kernel); on a
CPU-only process fla prints a "Triton is not supported ... roll back to CPU"
warning, which is harmless.

## Numerics: fresh runs only

The torch fallback casts q/k/v/beta/g to fp32 and runs the whole chunked
recurrence in fp32, casting the output back to bf16
(`modeling_qwen3_5.py:265-329`). The fla kernel consumes bf16 q/k/v with
fp32 accumulation and a different chunking/reduction order. Every hidden
state after the first GDN layer therefore differs at bf16 rounding level, so
losses, reverse-KL scores, entropies and hence selections are **not bitwise
reproducible** against runs made with the fallback (kl50, kl50w, the LR
sweep). They are statistically equivalent; there is no known accuracy
change.

Rule, same as ADR 0002's sampling-setting rule: install between runs, never
mid-run. Any run that is meant to be compared with an earlier one must
either keep the fallback or restart every arm from round 0 on the kernels.
Record which was in use (the check above) in the run notes.

## Expected effect

Estimate from `docs/perf_review.md` (derived from measured utilisation,
not yet measured on the GPUs): the HF-side stages sit at 18-29% of A100 peak
because the linear layers are a Python loop over 64-token chunks. Per
round-arm: train 25.4 -> ~13-16 min, oracle scoring 27.5 -> ~15-18 min; the
LR sweep shrinks proportionally. Settle it with one A/B: one train step and
one scoring shard on a round-0 selection, fallback vs kernels.

## Measured 2026-09-01 (A100-80GB, bf16, seq 8192, GPU 1, `.venv-kernels`)

| workload | torch fallback | fla 0.5.2 | speedup |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B forward, batch 4 | 20.3k tok/s (1.61 s) | 41.7k tok/s (0.79 s) | 2.05x |
| Qwen3.5-9B forward, batch 4 | 7.0k tok/s (4.66 s) | 11.8k tok/s (2.79 s) | 1.67x |
| Qwen3.5-2B fwd+bwd, batch 2, grad ckpt | 1.5k tok/s (11.08 s) | 8.5k tok/s (1.92 s) | 5.8x |

Peak memory unchanged (58.4 GiB for the train case). The backward of the torch
chunk fallback is the dominant training cost; the fused kernel removes it.
Installed into the main `.venv` on 2026-09-01 after the LR sweep was stopped;
no run was live. Any run whose earlier steps were trained without the kernels
(kl50w banks step 0 from kl50) mixes the two paths at bf16 rounding level.
