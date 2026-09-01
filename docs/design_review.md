# Design review: the refresh loop (engine restarts, training passes, eval)

Date: 2026-09-01. Scope: the planned r1/r2 loop from `docs/decisions.md` and
ADR 0005/0006 -- 100 training steps per arm, a refresh every 10 steps (eval
MATH-500 500x4 + AIME 60x16, re-rollout 80 questions x 4 (r1) or x 12 (r2),
optional scoring, 10 train steps, checkpoint), arms sequential on 2x A100-80GB.
Question: how to restructure training / vLLM engine lifetimes so less
wall-clock goes to bring-up and idle GPUs. Report only; no code changed.

Every claim about the installed stack was checked against `.venv`
(vLLM 0.26.0, TRL 1.10.0, torch 2.11.0+cu130, transformers 5.15.0). Costs
are from `outputs/runs/kl50/driver.log` unless marked *est.*

## 0. Where the minutes actually go (r1, per arm)

Measured unit costs (kl50, cap 8192, concurrency 128, 2 GPUs, one shard per GPU):

| unit | measured | source |
|---|---|---|
| MATH-500 500x4 generate | 15:07 per shard (8.3k gen tok/s per GPU, 7.5k tok/sample, 78% cap-hit) | driver.log:1828-1833 |
| rollouts 136x12 | 13:25 (8.1k tok/s, 7.9k tok/trace) | driver.log:2040-2048 |
| engine bring-up, warm compile cache | **41 s** from the previous stage's exit to `init engine ... took 14.8 s` (weights 1.2 s, compile-cache load 3.8 s, graph capture 5 s) | driver.log:1480 -> 1590 |
| engine bring-up, cold compile cache | 137-156 s | docs/perf_review.md finding 7 |
| engine session tail (last grades, teardown) | 10-20 s (generate 15:07 vs wall 15:16; stage wall 29:31 vs generate 28:32) | driver.log:1828, 2040-2050 |
| train pass, 17 steps, DDP micro 8 | 1514 s train (89 s/step) + **27 s** process/loads/save | driver.log:1375-1410 |
| optimizer_state.pt save/load (kl50w) | +~15 s *est.* (8 GB) | train.py:326-337 |

Derived r1 per-refresh cost in the current design (fresh process per stage,
AIME as a second `--eval-only` launch, fused kernels for training):

| phase | minutes | arithmetic |
|---|---|---|
| session 1: MATH-500 + rollouts 80x4 | 19.0 | 1.2 bring-up+tail + 15.1 + 320 x 7.9k / (2 x 8.1k tok/s) = 2.6 (+ half-empty second chunk) |
| session 2: AIME 60x16 | 8.9 | 1.2 + 960 x 8.0k / (2 x 8.3k) = 7.7 |
| train 10 steps | 6.3 | 0.75 overhead + 10 x ~33 s *est.* (89 s split ~60 student / 18 teacher / 11 heads+optim; fused 60/5.8 + 18/1.67 + 11 = 32 s; docs/kernels.md) |
| **per refresh** | **34** | |
| per arm (10 refreshes + terminal eval 25 min) | **~6.1 h** | 10 x 34 + 25 = 365 min |
| 4 arms | ~24.4 h | |

Shares per arm: eval generation 251 min (69%), bring-ups + tails 22 x 1.2 +
10 x 0.75 = 34 min (9%), rollouts 26 min (7%), training 55 min (15%). With the
fused kernels the loop is **eval-bound 4:1 over training**; engine restarts are
a 9% item. That ordering drives the ranking below. With
`target_concurrent_sequences: 256` (already in `conf/engine/default.yaml`,
never run yet; perf_review estimates 1.5-1.7x on decode) the eval terms shrink
to ~9.5 + 5 min and the per-arm total to ~4.6 h.

For r2 add per refresh: rollouts 80x12 = 7.6 min instead of 2.6, and exact-KL
scoring of 960 trajectories ~13.5 min (kl50: 1632 traj in 23 min on the fused
path) for the three KL arms.

## 1. Options, ranked by minutes saved per arm

Summary table (r1, per arm, against the 128-concurrency baseline above; the
arithmetic is in each section):

| rank | option | min saved / arm | risk | measurement change |
|---|---|---|---|---|
| 1 | D2 vLLM concurrency 512 (`max_num_seqs`) | ~135 *est.*, needs a 10-min probe (256 alone ~80) | low-med | none (seeds per request unchanged; not bitwise) |
| 2 | A persistent sleeping engine + in-place weight reload | 24 (12 once D1 is in); removes the 2.5-min cold-cache case | medium | none |
| 3 | B two single-GPU arm pipelines (arms concurrent) | ~20 equivalent (1.3 h over 4 arms); makes A simple | medium | grad accumulation order (not bitwise) |
| 4 | D1 one engine session for MATH-500 + AIME + rollouts | 16 | low | chunk composition (not bitwise) |
| 5 | E trainer process reuse | 5-8 | med-high | none |
| 6 | C teacher on vLLM for scoring | 0 in r1; ~50 in r2 KL arms | high | **changes the selection statistic** |
| -- | F rollout cadence 5 / eval cadence 10 | costs +27..39 | low | on-policy staleness halves |
| -- | G reliability items | 0 | -- | -- |

### A. Keep one vLLM engine per GPU alive; sleep during training; hot-swap weights

What the installed vLLM offers (all verified):

- `LLM.sleep(level, mode)` / `LLM.wake_up(tags)` at
  `vllm/entrypoints/llm.py:796-833`. Level 1 offloads the `weights` pool to
  CPU and discards the `kv_cache` pool; level 2 discards both. `wake_up`
  accepts `tags=["weights"]` / `["kv_cache"]` for a two-phase wake.
- Requires `enable_sleep_mode=True` at engine construction
  (`vllm/engine/arg_utils.py:682`; `vllm/config/model.py:545-550` turns on the
  cumem allocator automatically). `build_llm`'s `**extra` already passes it
  through `_filter_engine_kwargs` (`apod/models/generate_vllm.py:23-35, 220`).
- Only two allocations live in the cumem pools: weights
  (`vllm/v1/worker/gpu_worker.py:426`) and the KV cache
  (`gpu_worker.py:731`). For Qwen3.5 the GDN/mamba state is part of the KV
  cache tensors created inside that `kv_cache` context, so it is discarded and
  rebuilt on wake like any KV block; nothing hybrid-specific gates sleep
  (grep for `enable_sleep_mode` finds only the config lines). Sleep also runs
  `gc.collect(); torch.cuda.empty_cache()` (`vllm/device_allocator/cumem.py:280-281`).
- What stays resident while asleep: the EngineCore CUDA context (~0.5 GiB),
  the captured cudagraphs (0.43 GiB for the 2B, driver.log:838, captured in
  `compile_or_warm_up_model` outside the pools), and small runner buffers.
  **Budget ~1.5 GiB, measure it**: the worker logs `Sleep mode freed X GiB, Y
  GiB memory is still in use` on every sleep (`gpu_worker.py:213-218`).
- In-place reload of a new checkpoint: `LLM.collective_rpc("reload_weights",
  kwargs={"weights_path": ckpt_dir})` (`llm.py:560`, worker
  `gpu_worker.py:444`, runner `vllm/v1/worker/gpu_model_runner.py:5480-5556`).
  It sets `model_config.model = weights_path`, iterates the safetensors with
  the default loader and calls `model.load_weights(...)`, which is our
  adapter's override with the `model.language_model.` -> `model.` mapper
  (`apod/models/vllm_qwen35.py`), then re-runs post-load processing
  (`initialize_layerwise_reload`/`finalize_layerwise_reload`) so the fused
  qkv/gate_up/in_proj tensors are rebuilt. TRL's comment that `reload_weights`
  "reloads the initial checkpoint" (`trl/generation/vllm_generation.py:543`)
  refers to the call without `weights_path`; this build takes the path.
  Weights load in ~1.2 s from page cache (driver.log:1537).
- Prefix cache must not survive a weight change. `sleep(level>=1)` clears it
  (`vllm/v1/engine/core.py:877` `clear_prefix_cache = level >= 1`); if the
  engine is ever reloaded without sleeping, call `llm.reset_prefix_cache()`
  (`llm.py:789`). Serving old-weight KV blocks would be a silent wrong-model
  bug, so put the reset in the reload helper unconditionally.
- Weight-transfer push from a trainer (`start_weight_update` / `update_weights`
  at `gpu_worker.py:1338`) also exists but needs a configured transfer engine;
  disk reload is enough here because the checkpoint is written anyway.

Per-refresh sequence per GPU: `wake_up(["weights"])` -> `collective_rpc(
"reload_weights", weights_path=...)` -> `reset_prefix_cache()` ->
`wake_up(["kv_cache"])` -> generate eval + rollouts -> `sleep(level=1)` ->
(trainer runs) -> repeat. That is the order TRL's online-DPO colocate path uses
(`trl/experimental/online_dpo/online_dpo_trainer.py:706-739`).

Memory feasibility with the trainer as a separate process on the same GPU:

| trainer config | measured peak | + sleeping engine ~1.5 GiB | fits 80 GiB? |
|---|---|---|---|
| DDP micro 8 (current) | 71.5 GiB (conf/train/gkd.yaml comment) | ~73 | yes, ~7 GiB margin -- tight |
| micro 4 | 59.2 GiB (apod run; 58.4 in docs/kernels.md) | ~61 | yes, comfortable |

Level 2 saves nothing here (334 GB host RAM holds the 3.6 GB CPU copy trivially)
and forces a full weight load on every wake; use level 1.

Savings: each session currently costs 1.2 min warm (2.5 cold); the hot swap
costs ~5 s (weights wake ~1 s, reload 1.2 s, KV re-alloc, no graph capture,
no import). Per arm: 22 sessions x 1.1 = **24 min**; after D1 merges the AIME
session: 11 x 1.1 = **12 min**. Same for r2.

Files: `apod/models/generate_vllm.py` (pass `enable_sleep_mode`, a
`reload_checkpoint(llm, path)` helper that reloads + resets prefix cache; the
`_stable_model_alias` symlink and compile-cache pinning become irrelevant for
a live engine), a loop that owns the engine (see B; or a `--serve` mode of
`apod/stages/rollout_eval.py` driven by the bucket driver -- that needs an
IPC surface and is the more complex shape). Numerics: requests and seeds are
unchanged, so the sampled distribution is identical; bitwise reproducibility
is not guaranteed, same as today's resume path. Blockers: none in the stack;
validate the reload path once with the elementwise tensor probe from
`scripts/verify_vllm_reload.py` (`apply_model(_tensor_probe)` against the
new checkpoint after a reload) and check the "still in use" line does not grow
over 11 sleep/wake cycles.

### B. Two-GPU arm pipelining

Arms are independent, so the simplest pipelining is not "train A while B
evals, then swap" but **one arm per GPU, two arms at a time, each arm a
single-GPU pipeline**: engine (persistent, per A) and a single-GPU trainer
alternate on the same card; no coordination between GPUs at all.

Single-GPU training fits: micro 4 x accum 8 = effective 32 (the assert at
`apod/stages/train.py:184-196` passes with `WORLD_SIZE=1`), 59.2 GiB
measured at cap 8192; 9B teacher + 2B student + grads + bf16 Adam moments are
already in that number (the teacher is replicated per rank today). Step time
doubles (all 32 rows on one card, ~64 s *est.* fused), no all-reduce.

Wall-clock arithmetic (r1, concurrency 128):

| | per refresh | per arm | 4 arms |
|---|---|---|---|
| sequential, 2 GPUs per stage (section 0) | 34 min | 6.1 h | 24.4 h |
| pipelined, 1 GPU per arm, 2 arms concurrent | 1.2+30.2+5.2 (session 1) + 1.2+15.4 (AIME) + 0.75+10.7 (train) = 64.6 min | 11.6 h | 2 pairs x 11.6 = 23.1 h |

Both GPUs are busy in both designs (eval shards and DDP both scale ~linearly),
so the only recoverable time is the serial overhead that today idles *both*
cards: bring-ups, tails, trainer init, checkpoint save = ~34 min per arm. In
the pipelined shape one card computes while the other pays its overhead, so
half of that comes back: **~1.3 h over 4 arms, ~20 min per arm**. With A on
top (bring-ups gone) the difference between the two shapes shrinks further.

Why still do it: it is the natural host for A (one Python process per GPU
builds the engine once, runs `generate`, sleeps, launches
`python -m apod.stages.train` as a subprocess with `CUDA_VISIBLE_DEVICES`
pinned, wakes and reloads -- no IPC), an engine or trainer crash stalls one
arm, not the run, and per-arm latency is the only cost (first two arms land
together at 11.6 h instead of one at 6.1 h).

Blocker in the current code: `_stable_model_alias` is one symlink per config
hash shared by every checkpoint load (`generate_vllm.py:133-157`); two arms
loading different checkpoints concurrently would re-point it under each other
and `build_llm` hard-fails on purpose (`generate_vllm.py:257-266`). Key the
alias by arm (or drop it: with A the engine is built once from the base model
and every later load goes through `reload_weights(weights_path=<real path>)`,
which never touches the alias). Also `bucket_stats.jsonl` is appended by every
arm; per-arm files or a lock.

Numerics: the 32 rows of a step are identical; micro 4 x 8 on one rank sums
them in a different bf16 order than 8 x 2 x 2 ranks -- not bitwise, same
gradient in expectation. The `order_rows` rank interleave of ADR 0007 is moot
with one rank.

### C. Teacher on vLLM for scoring (r2 only)

r1 trains on all 4 rollouts, so there is no scoring and nothing to save. For
r2's three KL arms: 960 trajectories x ~8k = 7.7M positions per refresh.

- HF exact path (current, fused kernels): 23 min for 1632 trajectories ->
  **13.5 min** per refresh.
- vLLM teacher prefill: 9B at ~50% of A100 peak = 156 TFLOPS / 18 GFLOP per
  token = 8.7k tok/s per GPU *est.* -> 7.7M / (2 x 8.7k) = 7.4 min, plus a 9B
  engine session (bring-up 45 s, driver.log:36-133) unless it too is kept
  asleep (16.8 GiB weights, driver.log:82) -> **~8.5 min**. Student-side
  log-probs are free at rollout time with `logprobs=0`: `logprobs_mode`
  defaults to `raw_logprobs` (`vllm/config/model.py:229`), and at temperature
  1.0 raw is the model distribution. Saving ~5 min x 10 = **~50 min per KL arm**.
- What is and is not computable: the sampled-token estimator `rkl_mc` (already
  implemented in HF form, `scripts/oracle_kl.py:173-192`) and the **exact**
  top-16 overlap ratio and intersection-renormalised advantage of Eq. 6-7
  (`oracle_kl.py:151-162` only needs both models' logprobs on ids that are in
  *both* top-16 sets, which `prompt_logprobs=16` on the teacher and
  `logprobs=16` on the student provide). The exact full-vocab mean reverse KL
  is not computable from top-k; the selection statistic would change.
- Blockers: `prompt_logprobs` on 8k-token prompts materialises the full
  [tokens_in_chunk, 248k] fp32 logits per prefill chunk
  (`gpu_model_runner.py:5562` `_get_prompt_logprobs_dict`); with the
  LLM-class default `max_num_batched_tokens` = 8192 on A100
  (`vllm/engine/arg_utils.py:2466-2478`) that is an 8 GiB transient -- lower
  it to 2048-4096 for the scoring engine. The validation artefacts exist
  (`outputs/runs/kl50/arms/kl_high/rounds/round_00/oracle/oracle_kl_mcn.shard*.jsonl`,
  `oracle_kl.py --validate-mc`), and the user ruled MC out for the current run
  (todo.md "TODO ONLY, explicitly NOT this run"). Do not touch before r1; for
  r2 only if `--validate-mc` shows tertile agreement well above the 0.33
  chance level and the user re-decides.

### D. Eval cost

D1. **One engine session for MATH-500 + AIME + rollouts.** Today the AIME
monitor is a second `--eval-only` launch per refresh (ADR 0006; `pipeline.md`
stage CLI), i.e. a second bring-up and a second drain tail every refresh.
`run_session` already packs several writers into one stream
(`apod/stages/rollout_eval.py:497-559`); `EvalWriter` takes its own dir,
marker and seed base per writer (`rollout_eval.py:293-310`), so a second
`EvalWriter` for `aime2526` (own `cfg.eval` block via `select_eval_set`,
`rollout_eval.py:138-161`) drops in. Saving: 11 x (1.2 bring-up + ~0.3 boundary
tail) = **~16 min per arm**. Files: `rollout_eval.py` (`--eval-dataset`
becomes a list; per-writer eval config), `scripts/bucket_experiment.py`
(`run_rollout_eval`), ADR 0006 / pipeline.md wording. Numerics: identical
request texts, SamplingParams and per-(refresh, question, sample) seeds; the
chunk composition changes, so outputs are not bitwise identical to two
separate launches (same caveat as any batching change; the distribution is
unchanged). Fresh-run change, fine for r1.

D2. **Concurrency 512.** The 2B decode step at 128 sequences is overhead-bound
(15 ms/step for a model whose weights read in ~2 ms; perf_review). The engine
has KV room for 4,996,677 tokens at 0.95 (driver.log:822); 512 x (8192 + ~300)
= 4.35M fits, cudagraphs are already captured up to 512 (driver.log:36
`cudagraph_capture_sizes ... 512`), and the LLM-class default `max_num_seqs`
on A100 is 256 (`arg_utils.py:2466-2478`), so 512 needs
`max_num_seqs=512` through `build_llm(**extra)` plus
`target_concurrent_sequences: 512`. Estimate: step time grows to ~25-30 ms
(2B x 2 FLOP x 512 = 2.3 TFLOP of GEMM per step plus the 6 full-attention
layers' KV reads) -> 17-20k tok/s per GPU, 2-2.4x. Eval+AIME per refresh
22.8 -> ~10.5 min: **~135 min per arm** *est.*; 256 alone (already configured)
~80 min. **Measure before believing it**: one chunk of 128 problems x 4 on the base
model with `max_num_seqs=512` vs 32 x 4 at 128, ten minutes of GPU. Risks: KV headroom is
~13% at 512 (preemption by recompute if exceeded -- lower to 448 if the log
shows preemptions); hybrid prefix caching in `align` mode is flagged
experimental (driver.log:20-23, `vllm/model_executor/models/config.py:558-602`).
Numerics: seeds per request unchanged; sampling is not bitwise identical across
batch compositions (perf_review finding 2). Measurement protocol unchanged.

D3. **Prefix caching across the 4/16 samples, `n=4` in one request vs 4
requests.** Nothing to gain. Prompts average 102 tokens (max 323; kl50
trajectories) against 7.5k generated per sample, so prefill is ~1.3% of the
work even without sharing. vLLM V1 fans `n>1` out into `n` child requests
with `seed + index` (`vllm/v1/engine/parallel_sampling.py:68-80`); four
separate requests with seeds `seed+i` are scheduled identically and seeded
identically, so ADR 0003's per-(refresh, question, sample) seeds hold either
way and neither form is faster.

D4. **Options that change what is measured** (listed, not recommended; ADR
0003/0006: "reduce rounds, never samples"): intermediate MATH-500 100x4
(perf_review finding 3; -12 min x 9 refreshes), AIME every 20 steps
(-7.7 x 5 = 38 min), AIME avg@8 (-3.9 x 11 = 42 min). Only with an explicit
user decision and an ADR amendment.

D5. Decoupling eval from the loop (evaluate saved checkpoints on a trailing
schedule) saves nothing while both GPUs are already saturated; it only helps
if the training loop should finish early (e.g. to start r2 sooner) and the
evals are allowed to trail.

### E. Trainer process reuse

Per pass the fresh process costs 27 s (kl50) plus ~15 s for the 8 GB
optimizer round trip (kl50w): **~5-8 min per arm** at 10 passes. Numerics of
the current design are already clean: bf16 weights via safetensors and bf16
Adam moments via `torch.save` are lossless, the scheduler is deliberately
round-local (`train.py:287-310`), and `_restore_optimizer_state` name-maps
the moments. In-process reuse would be identical up to kernel
non-determinism.

What reuse needs: call `trainer.train()` again with a new `train_dataset`,
set `trainer.lr_scheduler = None` so the per-refresh warmup+cosine cycle is
rebuilt (transformers only creates a scheduler when it is None), keep
`trainer.optimizer`, reset `trainer.state`/`global_step` bookkeeping for the
TensorBoard export, and keep the teacher resident. The catch is memory: a
resident trainer holds 9B teacher (18 GiB) + student, grads and Adam
(~14.5 GiB) = ~33 GiB, leaving the engine ~45 GiB -> `gpu_memory_utilization`
~0.55 -> ~2.9M KV tokens, enough for 256 concurrent 8k sequences, not 512.
Offloading teacher + optimizer to host RAM around generation (22 GB over
PCIe, ~3-5 s each way) restores the 0.95 engine. That is exactly TRL's GRPO
colocate shape (`trl/generation/vllm_generation.py:351-372`: `LLM(...,
enable_sleep_mode=True)` in the trainer process, `sleep(level=2)`,
`wake_up(["weights"])`, push params with `load_weights` at line 394,
`wake_up(["kv_cache"])`), which relies on
`distributed_executor_backend="external_launcher"` (worker in-process) --
different from our separate-EngineCore engine, where the equivalent is the
disk `reload_weights` of option A. GKD's own on-policy path (`lmbda`,
`trl/experimental/gkd/gkd_trainer.py:449-530`) generates with HF `generate`
inside the training step; at 8k tokens that is far slower than vLLM and shares
the card with the 9B teacher, so it stays at 0.0. Verdict: small win, large
surface; do after r1 if at all.

### F. What the OPD literature does with the loop, and what transfers

- Thinking Machines / Tinker cookbook (`tinker_cookbook/distillation/train_on_policy.py`,
  fetched 2026-09-01): weights go to the sampler **every step**
  (`compute_full_batch_metrics_and_get_sampling_client`), teacher log-probs
  come from the teacher *sampling client* (`compute_logprobs_async` on the
  sampled trajectory), the per-token advantage is the sampled-token log-ratio
  `reverse_kl = sampled_logprobs - teacher_logprobs`, fed to an
  importance-sampling loss; no other staleness handling. So the reference
  recipe is 0-1 step stale, uses the MC estimator as the *loss* (ours is the
  exact full-vocab reverse KL via Liger, beta=1), and computes the teacher on
  the inference stack (option C's shape).
- verl hybrid engine / OpenRLHF: trainer and vLLM colocated, vLLM asleep
  during the update, weights synced in place each step (NCCL broadcast or
  `update_weights`); SEAD, FiRe-OPD and the 2026 OPD papers are verl-style
  per-step-sync loops. TRL's own answer is GRPO colocate (section E) and the
  `trl vllm-serve` server with `update_named_param` over NCCL
  (`trl/scripts/vllm_serve.py:117-149`).
- TRL GKD `lmbda` (in-trainer HF generation) does not apply at this cap.

Two things transfer to this scale:

1. **Staleness is a design variable we do not log.** Steps 10k+1..10k+9 train
   on rollouts from step 10k; the references are 0-1 steps stale. Cheap
   monitor: store the sampled-token log-prob from vLLM at rollout time
   (`logprobs=0`, one float per token in the npz) and log per training step
   the mean `log pi_theta(y_t) - log pi_rollout(y_t)` over the batch (the
   chunked lm_head pass planned for the ADR 0005 per-step diagnostics can
   gather it). Blockwise Policy-Drift Gating (arXiv 2606.24084, lit review) is
   the same quantity used as a gate. Zero GPU cost, tells us whether 10-step
   refreshes are off-policy in practice.
2. **Rollout cadence and eval cadence need not be the same number.** With
   training at ~6 min per 10 steps and eval at ~23 min, re-rolling every 5
   steps (40 questions x 4) and evaluating every 10 halves staleness for
   +10 rollout sessions per arm: 10 x (1.2 + 1.5) = 27 min at 128 concurrency,
   ~15 min with A. Worth it only if the staleness monitor says so.

### G. Reliability, resume, I/O, disk

- Resume granularity today: eval and rollouts row-level with torn-tail repair
  (`rollout_eval.py:185-204`), scoring per trajectory, train per pass. A train
  crash now costs <=6 min (fused), so `save_steps` inside a pass stays
  unnecessary.
- **Checkpoint atomicity gap**: `save_pretrained` writes `model.safetensors`
  in place (`train.py:321-325`); the *next* refresh's `resolve_model_path`
  accepts any `*.safetensors` present (`rollout_eval.py:174-182`) while the
  driver's skip check requires the `train/done.shard0` marker
  (`bucket_experiment.py:252`). A kill mid-write leaves a torn file the
  rollout stage would load. Fix: save into `checkpoint.tmp` and `os.replace`,
  or make `resolve_model_path` require the marker.
- Checkpoint I/O: 3.6 GB weights (`ls`: 3,763,692,048 B) + 8 GB
  `optimizer_state.pt` per pass, ~15-30 s on the boot disk; 11 passes x 4 arms
  x 11.6 GB = 510 GB if everything is kept -- prune to last 2 + step-100
  (`prune_checkpoints`, `bucket_experiment.py:645`; kl50w already deletes the
  consumed optimizer file, `:795-802`): ~61 GB. Disk: 159 GB free of 246 GB
  (`df`), kl50 holds 22 GB, HF cache 24 GB. The two 375 GB NVMe drives are
  ephemeral and deferred by user decision (todo.md); a persistent engine
  removes the compile-cache-miss risk that was the original I/O pain.
- Persistent-engine designs raise the blast radius of an engine crash to the
  arm's loop; the existing stage markers make any (arm, refresh) boundary a
  restart point, so the loop process must be restartable and idempotent, as
  `drive_kl50` is. Log the sleep "still in use" GiB per cycle (section A) and
  fail if it drifts by more than ~1 GiB.
- With B, both GPUs write concurrently: per-arm logs and stats files, and the
  alias fix above.

## 2. Recommended sequence

Before r1 launches (in this order; each is independent):

1. **D2 probe (10 min of GPU)**: base model, chunk of 128 problems x 4 with
   `max_num_seqs=512` vs the current 32 x 4 at 128; read gen tok/s and the
   preemption counter. Adopt 512 (or 448) if it delivers >=1.8x; 256 is
   already configured and is the floor.
2. **D1 merge AIME into the MATH-500 + rollout session** (small, no numerics
   risk beyond chunk composition, fixes a doc statement in ADR 0006).
3. **A on B: one process per GPU per arm** with a persistent
   `enable_sleep_mode` engine (level 1) and the single-GPU trainer (micro 4 x
   accum 8) as a subprocess; `reload_weights(weights_path)` +
   `reset_prefix_cache` between refreshes; alias keyed by arm or bypassed.
   Verify once: reload tensor equality (verify_vllm_reload probe), "still in
   use" GiB after sleep, trainer peak with the sleeping engine present.
   Skip this if the two-day budget is not there: r1 loses ~24 min per arm
   plus the cold-cache exposure, nothing else.
4. **G checkpoint atomicity** (tmp dir + `os.replace`) and prune policy for
   11 checkpoints per arm.
5. **F staleness monitor**: store rollout-time sampled-token log-probs and log
   the per-step log-ratio with the other ADR 0005 diagnostics.

Later (after r1, before r2): C only after `--validate-mc` and a user
decision; E only if the loop becomes per-step (then it is the TRL-colocate
shape, and the trainer owning the engine is the right design); D4 only by ADR
amendment; the 5-step rollout cadence only if the staleness monitor motivates
it.

## 3. Summary

The r1 loop is eval-bound: at the measured 8.3k tok/s per GPU, MATH-500 500x4
plus AIME 60x16 is 23 of the ~34 minutes of each refresh, training is ~6 with
the fused kernels, and all engine and trainer restarts together are ~34
minutes per arm (9%). The stack supports everything needed to remove the
restarts -- vLLM 0.26 `sleep`/`wake_up` with tag-wise wake, in-place
`reload_weights(weights_path)` through our Qwen3.5 adapter, `reset_prefix_cache`,
and a 59-72 GiB training peak that coexists with a level-1 sleeping engine --
so a persistent engine per GPU is feasible and is best hosted by running each
arm as an independent single-GPU pipeline (two arms at a time), which also
overlaps the serial overheads and isolates failures. But the largest lever by
far is decode concurrency (512 sequences per engine, est. 2x on the eval
terms, ~135 min per arm, to be measured in ten minutes), followed by merging
the AIME launch into the main session (16 min) and the persistent engine
(12-24 min). Moving the teacher to vLLM changes the selection statistic and is
worth nothing in r1; trainer reuse is worth 5-8 minutes. Two design points
the literature suggests and we do not track: on-policy staleness across a
10-step refresh (log the behaviour-vs-current log-ratio; it is free) and the
option to re-roll every 5 steps while evaluating every 10.
