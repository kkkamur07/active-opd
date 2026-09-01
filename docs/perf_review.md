# Performance review (2026-09-01, report only — no code changed)

Scope: throughput and wall-clock of the kl50/kl50w loop as driven by
`scripts/bucket_experiment.py --kl50w` (cap 8192, 136 prompts x 12 rollouts,
500x4 eval every round, 4 arms x 3 rounds). Every number below is read from
`outputs/runs/kl50/driver.log` and the kl50w sweep logs, not estimated, unless
marked *est.* Nothing here was profiled on the GPUs (the lr sweep is live).

## Measured cost of one round-arm (kl50, KL arm, intermediate round)

| stage | file | measured | notes |
|---|---|---|---|
| eval 500x4 | `apod/stages/rollout_eval.py:run_eval` | 15.2 min (8.3k gen tok/s per GPU, 250 problems/shard) | mean 7.5k tok/sample, 79% cap-hit |
| rollouts 136x12 | `apod/stages/rollout_eval.py:run_rollouts` | 13.6 min (8.1k gen tok/s per GPU) | mean 7.9k tok/trace, 91-93% cap-hit |
| engine bring-up + teardown | `apod/models/generate_vllm.py:build_llm` | ~0.7 min | init 14-17 s warm, process start ~30 s |
| oracle KL scoring 1632 traj | `scripts/oracle_kl.py:compute` | 27.5 min (13.0M positions, ~4.05k pos/s per GPU) | student+teacher forwards + fp32 full-vocab math |
| GKD train 17 steps | `apod/stages/train.py` | 25.4 min train + ~1 min bring-up/save | 89 s/step at micro 8 x accum 2 x 2 ranks (253k tokens/step) |
| **total** | | **~83 min** | random arm: ~55 min (no scoring) |

Run-level: 4 arms x 3 rounds gives ~13.7 h plus probes; the per-arm lr
sweep now running is ~40 trials x ~12.5 min = ~8.3 h on its own.

Per-stage utilisation, derived from the measurements:

- Train step: ~39 GFLOP/token (student fwd+recompute+bwd 16, teacher fwd 18,
  fused heads ~5) x 126.6k tokens per GPU per step = 4.9 PFLOP in 89 s =
  **55 TFLOPS, ~18% of A100 peak**.
- Scoring: (9B + 2B) x 2 FLOP x 4.05k pos/s = **~90 TFLOPS, ~29% of peak**.
  The fp32 vocab math (log_softmax, KLs, entropies, two top-16s) moves
  roughly 40 MB per position pair = ~160 GB/s, i.e. **<10% of HBM
  bandwidth** — it is NOT the bottleneck, the decoder forwards are.
- vLLM decode: 128 concurrent sequences at 8.3k tok/s = **15 ms per decode
  step** for a 2B model whose weights read in ~2 ms; the step is
  overhead-bound, not bandwidth-bound, so concurrency is the lever.

## Findings, ranked by minutes saved per round-arm

### 1. Qwen3.5 linear-attention layers run on the pure-torch fallback (train + scoring) — est. 20-25 min

- Where: not our code. `transformers/models/qwen3_5/modeling_qwen3_5.py:248`
  (`torch_chunk_gated_delta_rule`) and `:385` are decorated
  `use_kernel_func_from_hub_with_fallback(..., "fla")`; the resolution order
  is Hub kernels -> the `fla` package -> the torch reference loop. Neither
  `flash-linear-attention` (`fla`), `causal_conv1d`, nor `kernels` is in
  `.venv`, and every launch sets `HF_HUB_OFFLINE=1`, so **18 of the 24
  layers** (`layer_types` in the 2B and 9B configs: 3 linear : 1 full) use a
  Python chunk loop (128 sequential 64-token chunks per layer at 8k tokens)
  in the student forward/backward and the teacher forward. vLLM is
  unaffected (it ships its own GDN Triton kernels), which is exactly why
  rollouts run at 8k tok/s while HF-side stages sit at 18-29% MFU.
- Consumers: `apod/stages/train.py:206,246` (student + teacher),
  `scripts/oracle_kl.py:111-148`, `apod/stages/entropy.py:75`,
  `scripts/kl_drift.py`, both LR probes.
- Change: `uv add flash-linear-attention` (pure Triton; Triton 3.6 is already
  installed, no nvcc — the cu130 nvcc/header mismatch documented in
  `docs/guide.md` makes `causal-conv1d` risky and it is not needed: the
  depthwise conv fallback is cheap). transformers picks `fla` up
  automatically through the decorator; verify with a one-line check that
  `torch_chunk_gated_delta_rule.__wrapped__`/the resolved implementation is
  the fla one, then re-measure one train step and one scoring shard. Do this
  AFTER the live run: it touches `uv.lock` and the module every subprocess
  imports.
- Estimate: HF stages are ~2x below the utilisation the dense parts alone
  should reach; train 25.4 -> ~13-16 min, scoring 27.5 -> ~15-18 min.
  Unmeasured; a 15-minute A/B on one round-0 selection settles it.
- Risk: medium. Package/Triton version compatibility; a kernel that changes
  the recurrent-state dtype path (`mamba_ssm_dtype: float32` in the config)
  must be checked for the bf16 cast points.
- **CHANGES NUMERICS (loudly): yes, at bf16 rounding level.** The fla
  kernels compute the same recurrence with different accumulation order and
  fused precision. Losses, KL scores and selections will not be bitwise
  reproducible against kl50/kl50w; they are statistically equivalent. Any
  run that is meant to be compared against kl50w must either keep the
  fallback or restart every arm from round 0 (same rule as ADR 0002's
  "changing a sampling setting invalidates all curves").

### 2. vLLM concurrency: 128 of the scheduler's 256 slots used (eval + rollouts) — est. 9-10 min

- Where: `conf/engine/default.yaml` `target_concurrent_sequences: 128` ->
  `apod/stages/rollout_eval.py:250` (eval chunk = 32 problems x 4) and
  `:357` (rollout chunk = 10 prompts x 12 = 120). vLLM 0.26's `LLM`-class
  default on A100 is `max_num_seqs=256` (`vllm/engine/arg_utils.py:2476`),
  and the engine reports KV room for 305 concurrent 16k-token requests
  (`driver.log`: "Maximum concurrency ... 304.97x" at 0.95 utilisation).
- Change: `target_concurrent_sequences: 256` (one config value; no code).
  Going past 256 also needs `max_num_seqs` passed through `build_llm`'s
  `**extra`. A 2B model at 15 ms/step is overhead-bound, so doubling the
  batch should cost well under 2x per step; a single 32-problem vs
  64-problem eval chunk timing on the base model gives the real number.
- Estimate: eval 15.2 -> ~9-10 min, rollouts 13.6 -> ~9 min at 1.5-1.7x.
- Risk: low (memory headroom is measured: 76.8 GiB peak at 0.95 with 5M KV
  tokens free). USER declined this bump for kl50 ("vLLM concurrency bump"
  in the run notes); re-propose for the next run, not mid-run.
- Changes numerics: **sampling outputs are not bitwise-identical** under a
  different batch composition (kernel reduction order), same as today's
  resume path already is; the sampled distribution is unchanged (per-request
  seeds). Flag it in the run manifest if adopted mid-experiment.

### 3. Full 500x4 eval every intermediate round (bucket driver) — 12 min on 2 of 3 rounds per arm

- Where: `scripts/bucket_experiment.py:161-178` `run_rollout_eval` never
  passes `--eval-num-problems`, so every round evaluates all 500 problems;
  the Hydra driver already has the 100x4 intermediate protocol
  (`apod/main.py:151-155`, `conf/eval/math500.yaml
  intermediate_num_problems`) and `scripts/eval_table.py` already
  subset-matches the round-0 baseline.
- Change: pass `["--eval-num-problems", "100"]` for non-terminal rounds in
  `run_rollout_eval` (kl50w's rounds 1 and 2; round 0 is banked, round 3 is
  terminal).
- Estimate: 15.2 -> ~3 min on intermediate rounds: **~12 min x 8 round-arms
  = 1.6 h per run.**
- Risk: none to training. USER declined for kl50 (wanted full curves);
  listed for completeness.
- Changes numerics: **changes what is measured mid-run** (SE ~1.8 pts at
  100x4 vs ~1.1 at 500x4; the first-100 prefix is ~3.5 strict points harder
  than the full set per `eval_table.py`). Terminal endpoint unaffected.

### 4. Scoring cost per se: MC reverse KL instead of exact full-vocab KL — est. 15-17 min, BLOCKED

- Where: `scripts/oracle_kl.py:160-190`. Recorded in `todo.md` as "TODO
  ONLY, explicitly NOT this run" (USER 2026-08-31).
- Given finding 1, the vocab math is <10% of scoring; the win here comes
  from dropping the student forward (log pi_S(y_t) is free from vLLM
  `logprobs=0` at generation) and running the teacher as one vLLM prefill
  pass (`prompt_logprobs`) instead of an HF forward.
- Changes numerics: **yes — changes the selection statistic** (sampled-token
  estimator under top-k/top-p-truncated sampling is not the mean KL). Must
  be validated offline against the stored exact scores first
  (`oracle_kl.py --analyze`, per-prompt Spearman + tertile agreement).
  Not before that.

### 5. Chunk drain tails and serial grading in the vLLM stages — est. 2-3 min

- Where: `apod/stages/rollout_eval.py:262-336` and `:406-439`. Each
  `llm.generate` chunk waits for its longest sequence; slot utilisation
  simulated from the kl50 eval/rollout length distributions is **0.89 for
  eval (chunk 32) and 0.93-0.95 for rollouts (chunk 10)**, and grading of
  chunk k (`graders.map`, `:292-298`, `:414-416`) runs before chunk k+1 is
  submitted, so the GPU idles for the grading time (small now: 8 workers,
  ~0.5 s per chunk).
- Change (two small edits, same output rows): submit grading as futures and
  drain them after the next `generate` returns; and merge the eval and
  rollout request lists of one engine session into a single `generate` so
  there is one drain tail instead of two. Row-level resume, seeds and
  EOS handling are untouched — the requests and their seeds are identical,
  only their submission grouping changes. Continuous admission (submit all
  250 problems, let the scheduler admit as slots free) would lift eval
  utilisation to ~0.92 but costs the per-chunk crash durability the
  chunking exists for; not recommended.
- Estimate: ~1.5 min eval, ~0.8 min rollouts, ~0.2 min grading.
- Risk: low. Changes numerics: same non-bitwise caveat as finding 2
  (batch composition); distribution identical.

### 6. Oracle scoring batch shape — est. 1-2 min

- Where: `scripts/oracle_kl.py:85` `MAX_BATCH = 4`, `:131-140` same-length
  grouping. With 93% cap-hit, groups average 3.24 of 4 (504 forwards for
  1632 trajectories). At 8k tokens a batch-4 forward is already
  compute-saturated on the dense layers, so raising `MAX_BATCH` to 8 (with
  `POSITION_CHUNK` halved to keep the fp32 transients at ~4 GiB) buys
  little on its own; it matters more after finding 1, when the linear
  layers stop being launch-bound. Changes numerics: no (per-row math
  unchanged; bf16 GEMM batching can differ in the last bit).

### 7. Per-launch bring-up: ~2.5 min per round-arm, spread over three processes

- rollout_eval: ~0.7 min (engine 14-17 s warm via the pinned compile cache
  and config-hash alias, `generate_vllm.py:133-177`, plus Python/vLLM import
  and NVML probe). Scoring: ~0.7 min (two model loads per shard,
  `oracle_kl.py:111-116`). Train: ~1 min (torchrun, two 9B teacher loads,
  `GKDTrainer` init) plus the 4 GB checkpoint + 8 GB `optimizer_state.pt`
  writes (`train.py:321-337`, ~30-40 s on the boot disk).
- Change: none recommended. Persistent engine/trainer processes would save
  ~2 min per round-arm at the price of the process-per-stage isolation the
  driver is built on (`apod/main.py` docstring). The NVMe move noted in
  `todo.md` would take ~30 s off the checkpoint/optimizer writes; deferred
  by USER decision.

### 8. Things checked that are NOT worth touching

- **Round-0 shared work**: already shared. `_copy_shared_round0`
  (`bucket_experiment.py:332-347`) symlinks tokens and copies ~50 MB of
  jsonl per arm per launch; the round-0 oracle scoring runs once; the
  teacher-ceiling eval (`KL50_TEACHER`) is marker-skipped on relaunch; kl50w
  banks kl50's round 0 (`_bank_round0`). No redundant GPU work found.
- **Teacher forwards repeated on identical trajectories**: the train stage
  recomputes teacher logits for the 544 selected trajectories that the
  scoring stage already forwarded (4.3M positions, ~4 min of the 25). Caching
  would need full-vocab teacher log-probs (2 TB at 8k x 544 x 248k x 2 B);
  top-k truncation would change the objective. Rejected. The oracle16k
  `teacher_scoring` block (`drive()`, `:571-575`, `:601-606`) re-forwards
  the frozen teacher over its own rollouts every round for the same reason;
  it is disabled in kl50/kl50w and only matters for `--drive`.
- **JSON/npz IO in hot loops**: `append_jsonl` per row (open/close per
  write), `load_complete_rows` rewrites, `np.savez_compressed` per example
  (400 KB), `Dataset.from_list` over 4.4M Python ints: all sub-second to
  seconds per stage. `_score_count` re-reads every oracle shard on each
  resume check: milliseconds.
- **Padding waste**: train uses length-grouped batching (shipped); scoring
  batches only exact-length groups (no pads at all); vLLM has no padding.
- **Shard imbalance**: eval shards finish within 20 s of each other;
  scoring splits npz files by index parity (both shards ~6.5M positions).
- **DDP traffic**: 4 GB bf16 grad all-reduce per 89 s step; negligible.
- **Resume gaps**: eval and rollouts resume at row level; scoring at
  trajectory level; selection/train at marker level. The only gap is that a
  crash mid-train redoes the whole 25-minute round (`save_strategy: "no"`).
  An HF `save_steps` every ~5 steps would cap the loss at ~8 min for +30 s
  per save, but it adds a second checkpoint format and a resume path to a
  stage that has never crashed mid-round; not recommended under the
  keep-it-simple rule.
- **`apod/stages/entropy.py`** scores one trajectory per forward
  (`:173-196`); it is not on the kl50 path (oracle KL replaced it). If it
  comes back, borrow `oracle_kl.py`'s same-length batching.
- **lr sweep (`scripts/lr_sweep.py`)**: per-trial overhead is ~45 s against
  ~12 min of steps; the ~8 h total is the design (40 trials x 8 steps),
  and finding 1 would shrink it proportionally. Nothing else to gain.

## Suggested order

1. After the live run: install `flash-linear-attention`, A/B one train step
   and one scoring shard (finding 1). Decide with the numerics caveat in
   view — it is a restart-from-round-0 change for any compared run.
2. Next run's config: `target_concurrent_sequences: 256` (finding 2),
   measured on the base model first.
3. Only with USER sign-off on the measurement change: intermediate 100x4
   eval in the bucket driver (finding 3).
4. Opportunistic, no numerics impact: grading futures + merged generate
   (finding 5).

## Probes 2026-09-01 (fused GDN kernels on, .venv fla 0.5.2)

Batch probe (kl50 pass-0 kl_high selection, 544 rows, cap 8192, 17 steps,
2 DDP ranks, diagnostics on; scratchpad bench/batch_probe.py):

| micro x accum | peak MiB | s/step |
|---|---|---|
| 16 x 1 | 73,949 | 42.6 |
| 8 x 2 | 56,885 | 44.5 |

Adopted 16 x 1 (largest micro under 75,000 MiB), conf/train/gkd.yaml and
conf/experiment/refresh_8k.yaml; 8 x 2 is the fallback if a real run OOMs.
Pre-kernels reference: micro 4 ~99 s/step, micro 8 71.5 GiB peak.

Concurrency probe (eval-only MATH-500, one engine per GPU, 256 vs 512
target_concurrent_sequences = max_num_seqs; scratchpad bench/conc_probe.py):

| model | conc | requests | wall s | req/min | preempt |
|---|---|---|---|---|---|
| 2B | 256 | 1024 | 856 | 71.8 | 0 |
| 2B | 512 | 1024 | 856 | 71.8 | 0 |
| 9B | 256 | 512 | 884 | 34.7 | 0 |
| 9B | 512 | 512 | 1205 | 25.5 | 0 |

512 gives nothing for the 2B and slows the 9B; conf/engine/default.yaml stays
at 256.
