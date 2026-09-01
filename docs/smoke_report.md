> **Status (2026-09-01): historical record.** This is the 2026-08-14 smoke cycle
> for the entropy-selection `apod` run (entropy_top4 / random_top4 / all,
> `python -m apod.main`). The verification results (checkpoint reload, compile
> caches, loss identities, EOS handling) still describe the code in use; the
> projections, gates, and go/no-go numbers were superseded by the KL-bucket
> experiments driven by `scripts/bucket_experiment.py` (oracle16k, oracle8k/4k,
> kl50, kl50w). Current costs: `docs/perf_review.md`; current loop: `docs/pipeline.md`.

# Smoke test report

Hardware: 2x A100 80GB (driver R595, Debian 13). Stack: TRL 1.10.0 GKDTrainer,
vLLM 0.26.0, transformers 5.x, torch 2.11.0+cu130.

Runs referenced:
- `outputs/runs/smoke` — first end-to-end smoke (rounds:1 semantics, 1024 cap,
  presence_penalty 1.5, 8 rollouts, single-GPU train). COMPLETE, all checks
  passed. Source of the verification numbers below.
- `outputs/runs/smoke2` — final smoke config (rounds 3, 12 rollouts, k 4,
  presence_penalty 0.0, 2048 cap, DDP train). IN PROGRESS — section 9.

## 1. Executive summary (read this; detail below)

**Where things stand (2026-08-14):** every structural, numerical, and
semantic check passes with real numbers; the 8192 measurement pass
(smoke8k) is done; the one remaining launch gate is teacher accuracy.

**Verified** (details in the cited sections):
- Checkpoint reload into vLLM is bitwise-correct: 248/248 params identical,
  logit deltas below the base-vs-base noise floor, 100% decode argmax
  agreement (§3). Compile caches pinned: warm init 15 s vs 137 s (§4).
- The training signal is the intended one: reverse KL(student‖teacher),
  beta=1, loss identities verified (§5); entropy is provably scored by the
  generating model (§9a).
- Train at 8192 works via Liger fused chunked JSD after finding and fixing
  an upstream TRL↔liger bug (their "chunked" loss doesn't chunk at batch
  size 1 → 75.7 GiB spike → OOM). Fix verified: fp32 gradients bitwise
  identical to the eager objective; peak now ~34 GiB/GPU, step 94.4 s at
  effective batch 16 (§10).

**Key numbers:** base-model eval at 8192: strict avg@4 0.250 / loose 0.469
(strict = \boxed required, the pre-registered PRIMARY endpoint, §11);
cap_hit 0.97 on rollouts — and 62.5% of traces still don't finish at a
24,576 budget, so truncation is intrinsic and the cap-hit gate is RETIRED
(user decision): cap_hit and the strict-loose gap are now per-round
metrics that should improve if distillation works. Cap stays 8192.

**Projected real run:** ~59 h of 2-GPU time (measured inputs, §10).
User-approved cost levers implemented: intermediate rounds eval 100×4
(full 500×4 at round 0 + terminal), round-0 eval computed once and shared
across arms. Rounds stay 8; the `all` arm stays (~60% of training budget,
a knowing choice).

**Recommendation:** pass the teacher-accuracy gate (§11), then launch
smoke3; if smoke3 is clean, go for the real run.

## 2. Configuration decisions locked in this smoke cycle

| knob | was | now | why |
|---|---|---|---|
| rollout.num_rollouts | 8 | 12 (k stays 4) | 4-of-8 keeps the top 50% (~0.80σ above mean) — barely different from random at the boundary; 4-of-12 keeps the top 33% (~1.10σ). Measured cutoff gaps at 4-of-8: median 0.0086 nats, min 0.0003 — noise-dominated. |
| sampling.presence_penalty | 1.5 | 0.0 | ADR 0004: generation, entropy scoring, and the KL objective now share one distribution — genuinely on-policy, no length→entropy coupling. Repetition risk is measured, not assumed (§6, §9). |
| train.gradient_accumulation_steps | 16 | 8 | 2 DDP ranks × 1 × 8 = effective batch 16, unchanged. |
| training layout | 1 GPU + idle GPU 1 | torchrun DDP × 2 | §7. |
| vLLM compile caches (both) | path-keyed (per-round miss) | config-keyed pin + guarded model alias | §4. |
| smoke profile | 1 round, 1024 cap | 3 rounds, 2048 cap | exercises checkpoint→checkpoint training twice, the natural-EOS path, and length variance. |

## 3. vLLM checkpoint-reload bug: root cause, fix, independent verification

Round ≥1 crashed: both EngineCore shards raised
`'Qwen3_5TextConfig' object has no attribute 'vision_config'`. Four
independent vLLM 0.26.0 gaps, all hit because transformers saves our round
checkpoints as `Qwen3_5ForCausalLM` (`qwen3_5_text`) while vLLM only wired
the multimodal arch:

1. Arch not in vLLM's registry → `_normalize_arch` suffix-swaps to the
   multimodal class → vision_config crash.
2. transformers 5.x saves weights nested (`model.language_model.*` — its
   registered conversion format, conversion_mapping.py:852); vLLM's text
   class loads flat `model.*` with no rename.
3. vLLM's text class lacks `IsHybrid` → GDN/mamba cache sizing never runs →
   `assert mamba_block_size` fails.
4. Text config carries interleaved M-RoPE (`mrope_section [11,11,10]`) →
   runner demands `SupportsMRoPE`, which the text class lacks.

Fix: `apod/models/vllm_qwen35.py` — a registered subclass adding the weight
mapper, `IsHybrid` (state-shape classmethods borrowed from the multimodal
class; they read only `hf_text_config`), and text-only M-RoPE positions
(plain index over T/H/W, delta 0). Checkpoint files untouched. HF reload was
verified healthy independently (0 missing/0 unexpected keys, tensors
bit-identical to the file), so stage_entropy/stage_train were never affected.

**Independent verification, hardened per user requirement** (floor control,
elementwise tensors, decode path, 2000 positions —
`scripts/verify_vllm_reload.py` on smoke2's round_02 DDP checkpoint, ALL 6
PASS, JSON beside the checkpoint):

- **Elementwise tensor equality (definitive)**: every parameter pulled from
  the LIVE engine model and compared against the checkpoint file with the
  packed-fusion recipes (qkv, gate_up, in_proj_qkvz/ba): **248/248 BITWISE
  identical, max abs diff 0.0** — not statistical, exact.
- **Prefill vs HF, calibrated by a floor**: 1999 real-rollout positions.
  Checkpoint through both stacks: mean |Δ| **0.0061**. The SAME comparison
  on the BASE model (pure cross-stack bf16 numerics, no adapter involved):
  **0.0073**. Ratio **0.83×** — the adapter's delta is BELOW the noise
  floor. (The earlier uncalibrated 0.0189 over 64 positions is thereby
  explained: it was numerics, not a defect.)
- **No positional drift**: |Δ| by position quartile 0.0082 / 0.0069 /
  0.0042 / 0.0050 (flat-to-decreasing; floor shows the same profile) — an
  M-RoPE/SSM-state bug would rise with position.
- **Decode path (live SSM state across steps)**: 128 greedy tokens from
  vLLM, teacher-forced through HF: per-step chosen-token |Δlogprob| mean
  **0.0036**, max 0.0567; HF argmax matches vLLM's greedy choice at
  **100.0%** of steps.
- **Trained, not base**: |vLLM(ckpt) − HF(base)| mean **0.2077** = **34×**
  the matched-load delta.
- `register()` idempotent; fork propagation proven by the EngineCore worker
  (separate process) resolving the class in every pipeline run.
- First-pass verification (earlier, smoke round_00 checkpoint): 248/248
  coverage, all 320 keys routed, twice in fresh processes — retained in git
  history; superseded by the hardened numbers above.

## 4. Engine init: recurring compile-cache miss, fixed

vLLM's torch.compile cache key hashes the model PATH
(`ModelConfig.compute_hash`), so every round's checkpoint dir was a full
cache miss:

There are TWO path-keyed caches, and they needed two separate fixes:

1. **torch.compile cache** — bypassed by pinning
   `compilation_config.cache_dir` to a directory keyed by the config.json
   content hash (identical across round checkpoints, invalidated by any real
   config change).
2. **AOT function cache**
   (`torch_compile_cache/torch_aot_compile/{sha256(env, vllm_config.compute_hash(), fn)}`,
   `vllm/compilation/decorators.py:~525`) — NOT covered by the cache_dir pin;
   the hash includes the model path and vLLM exposes no direct override
   (`local_cache_dir` is set FROM the computed path, never read). Fixed by
   serving local checkpoints through a config-hash-named symlink alias
   (`$VLLM_CACHE_ROOT/apod_model_alias/<hash>`), atomically re-pointed at the
   current checkpoint, so the path string vLLM hashes is constant across
   rounds. Because the alias is one mutable pointer shared by every
   checkpoint load, `build_llm` hard-fails after engine construction if the
   resolved alias no longer matches the checkpoint it was asked to serve
   (the pipeline is strictly sequential, so this should never fire — but the
   failure it guards is silent wrong-weights).

Measured (smoke2, all from `init engine ... took` log lines):

| load | init engine time |
|---|---|
| base repo (warm) | 13.3–14.5 s |
| checkpoint, default cache (every round!) | **153.1–156.0 s** (compilation ~26 s + ~125 s warmup) |
| checkpoint, pinned compile cache warm but AOT miss | **137.6–137.7 s** (compilation only 10.4–10.7 s; the rest is AOT recompile + warmup) |
| checkpoint, both caches warm via alias (terminal eval, first true reuse) | **14.97 / 15.01 s** (compilation 3.9 s) |

All round checkpoints share one config.json, so one alias entry serves every
checkpoint load: the remaining smoke2 arms and all ~21 checkpoint engine
builds in the real run hit the warm path. Recovered overhead for the real
run: ~21 × ~123 s ≈ **43 min**.

## 5. Semantic verification of the learning signal — ALL PASS

`scripts/verify_semantics.py` on real round-0 artifacts
(`.../round_00/train/verify_semantics.json`); one batch, inline computations:

- **Objective is exact reverse KL**: trainer beta=1 loss **0.15249** vs
  hand-recomputed KL(student‖teacher) from the two logit tensors **0.15248**
  (rel diff 2e-07). Forward KL on the same batch is 0.12635 — the direction
  is not flipped. KL(student‖student) = 0 exactly; KL ≥ 0.
- **No prompt/padding leakage**: 1024 scored positions == 1024 completion
  tokens; the 84-token prompt fully masked (labels −100).
- **No in-trainer generation**: lmbda=0.0 gates generation on
  `random() <= 0` (never fires); loss temperature is fixed 1.0 in TRL
  (args.temperature is generation-only — verified in gkd_trainer.py source).
- **Teacher sanity**: teacher ppl **1.43** < student ppl **1.53** on a
  held-out MATH-500 reference solution; printed top-5s are coherent (e.g. at
  a solution-final position: `<|im_end|>` 0.65, `\n\n` 0.31). Tokenizers
  identical (vocab 248044, eos 248046, same ids on same text). One finding:
  the 2B and 9B chat-template DEFAULTS differ (2B closes the think block,
  9B leaves it open) — harmless here because the pipeline always passes
  `enable_thinking` explicitly, under which both render token-identically.
- **Teacher frozen**: all 427 params requires_grad=False (now set
  explicitly in the train stage; TRL itself only relies on torch.no_grad),
  eval mode, none of the optimizer's 320 tensors are the teacher's, param
  sha256 **bit-identical across 20 optimizer steps**.
- **Training path works**: student digest changes; overfit micro-run on 2
  trajectories drops loss **0.1526 → 0.0746** in 20 steps; step-0 loss
  matches the hand-computed KL scale; grad norms 1.3–41.2 (the 41 is the
  overfit run's constant-LR tail, not the pipeline setting).
- **Eval/scoring path**: math-verify grades boxed answers correctly
  (incl. fraction equivalence), 20 stored rollout rows regrade identically,
  3/64 correct overall so the parser is not rejecting everything.
  stage_entropy's stored 0.4460 == recomputation on the same student to
  0.0 diff — entropy comes from the same full-vocab T=1 distributions the
  objective uses.

## 6. Entropy-selection diagnostics (penalty-on baseline, 1024 cap)

From `outputs/runs/smoke` round 0 (64 trajectories, presence_penalty 1.5):

- entropy ↔ correctness: r = **−0.018** (n=64, 3 correct) — no signal at
  this scale; needs the real-run volume to say anything.
- entropy ↔ length: **not computable** — all 64 responses were exactly 1024
  tokens (cap_hit 1.0). This is the structural reason the smoke moved to a
  2048 cap.
- Selection boundary: per-problem gap between 4th and 5th ranked entropy:
  {0.0003, 0.0011, 0.008, 0.0082, 0.009, 0.0245, 0.0291, 0.0349} — median
  0.0086 nats. At 4-of-8 the boundary was noise for at least half the
  problems; hence num_rollouts 12.
- Degeneracy eyeball: the highest-entropy trajectory (H=0.976) is genuine
  step-by-step reasoning, repeated 8-gram rate 0.97% — high entropy was NOT
  selecting repetition loops under the penalty.
- Presence penalty decision and tradeoff: ADR 0004. Penalty-off repetition
  numbers land in §9.

## 7. Both-GPUs proposal (layout + why)

**Rollouts/eval: two independent single-GPU engines, problems sharded by
`example_index % 2` (already the case — `num_gpus: 2`).** For a 2B model
(3.5 GiB of weights), TP=2 buys no memory headroom and costs an all-reduce
on every layer of every decode step; two replicas double throughput with
zero cross-GPU traffic and are resume-friendly per shard (measured: each
engine independently sustains ~3,290 gen tok/s at 64 concurrent seqs;
2 engines = ~6,580 tok/s aggregate with nothing crossing the bus).

**Training: DDP via `torchrun --standalone --nproc_per_node=2 -m
apod.stages.train`.** Student (2B) and frozen teacher (9B) are replicated
per rank; each rank trains its sampler shard. Per-step cross-GPU traffic is
ONLY the student gradient all-reduce (~4 GB bf16, bucketed and overlapped
with backward); the teacher never syncs. Memory per rank ≈ 4 (weights) + 4
(grads) + 16 (Adam fp32) + 18 (teacher bf16) ≈ 42 GB + activations — fits
80 GB with gradient checkpointing at 8192 (measured at 2048 in §9;
extrapolation to 8192 in §10). Effective batch preserved (accum 16→8).
Checkpoint saved by rank 0 only, from the unwrapped module (no `module.`
prefixes).

**Rejected alternatives:**
- *vLLM-precomputed teacher logits*: full-vocab logits are 248,320 × 8,192 ×
  2 B ≈ **4.1 GB per trajectory**, ~1.5–2 TB per real round — not storable.
  Top-k truncation changes the objective exactly where reverse KL needs the
  teacher: wherever the STUDENT puts mass, which is the tail top-k discards.
  Would be a different experiment. (User confirmed teacher-in-the-loss is
  fine; this stays rejected on data-movement and correctness grounds.)
- *Frozen teacher moved to GPU 1, single-rank training*: TRL runs student
  and teacher forwards sequentially inside compute_loss and
  `accelerator.prepare_model` places the teacher on the training device;
  cross-device placement buys zero overlap without custom async code, ships
  every batch's hidden-state-sized activations across the bus, and leaves
  GPU 1 idle between teacher calls anyway. DDP does strictly better: both
  GPUs run the full loss pipeline on disjoint data.
- *TP=2 for rollouts*: see above — all-reduce per layer for a model that
  fits 20x over in one card.

**Remaining idle-time map per round (after DDP):** entropy stage runs on
both GPUs (sharded) — busy; rollout_eval both GPUs — busy; train both GPUs
— busy under DDP; the residual idle windows are the per-stage model
load/setup serial sections (~14 s engine init, ~60–100 s HF/TRL trainer
setup per round). Closing those would need cross-stage process reuse
(keeping a trainer or engine alive across rounds), which trades the
resume-per-stage protocol for marginal minutes — not worth it at 8 rounds.

## 8. Round-0 eval determinism across arms

Round 0 of every arm evaluates the SAME base model, so the results must
match. In `outputs/runs/smoke`: random_top4 and all returned IDENTICAL row
sets (same single correct: problem 4, sample 0) — vLLM eval is deterministic
across arms within one code state (per-request seeds, identical batch
composition). entropy_top4's round-0 eval differed (correct: problem 10,
samples 0+1) because it ran in the earlier session under pre-fix code and
was resumed, not re-run. smoke2 (all three arms fresh, identical code) is
the clean test: if its three round-0 evals are not identical, eval carries
irreducible noise that must be priced into arm comparisons. → §9.

## 9. smoke2 results (rounds 3, 12 rollouts, penalty 0.0, 2048 cap, DDP) — IN PROGRESS

Round-0 findings already in (entropy_top4, 96 rollouts + 32 eval rows):

- **Repetition with the penalty OFF: a non-event.** Repeated-8-gram rate
  mean 0.008, median 0.005, worst single trajectory 0.048; **0/96
  trajectories above 30%**. At temperature 1.0 / top_p 0.95 / top_k 20 the
  penalty was not load-bearing for this model. ADR 0004's risk did not
  materialize at 2048.
- **cap_hit_rate is STILL 1.000 at 2048** — every rollout and every eval
  sample ran to the cap (`finish_reason` = length ×96; lengths all exactly
  2048). Plainly: the model never finishes naturally even at double the old
  cap, the natural-EOS path remains effectively untested, and the
  entropy↔length correlation is STILL structurally uncomputable (zero
  length variance). Expect heavy truncation at 8192 too in early rounds;
  the length→entropy coupling question can only be answered from real-run
  data once traces start finishing.
- **Selection boundary improved as designed**: 4th-vs-5th-of-12 entropy
  cutoff gaps {0.0026 … 0.0624}, median **0.0215 nats** — 2.5x the 0.0086
  median at 4-of-8. Two problems still sit at ~0.003 (noise), six are now
  clearly resolved.
- entropy ↔ correctness: r = **−0.168** (8/96 correct) — weak
  higher-entropy→less-correct trend, consistent with uncertainty sampling.
- Rollout accuracy 8/96 (8.3%) vs 3/64 (4.7%) in the penalty-on 1024 smoke;
  base-model eval at 2048: avg@2 **0.156** / pass@2 **0.25** (vs 0.062 at
  1024) — the extra length helps even though everything still truncates
  (the boxed answer often lands before the cap).
- **First DDP train succeeded**: 32 trajectories × 2048 completion tokens in
  train_runtime 71.95 s (0.445 samples/s ≈ 911 tok/s) vs the single-GPU
  93.1 s at HALF the tokens (352 tok/s) — **~2.6x token throughput**. Both
  GPUs ~44.4 GB and busy during the step (sampler CSV). Checkpoint saved by
  rank 0 with clean keys (320, no `module.` prefix) and was immediately
  consumed by round 1's engines.

- **entropy_top4 full-arm curve is flat — an eval-RESOLUTION artifact, not a
  stale checkpoint** (peer session cross-checked independently). avg@n was
  0.15625 at rounds 0, 1, 2 (5/32 each) and 0.09375 (3/32) at the terminal
  eval, while the manifests show a different `model_path` each round and
  rollout_accuracy DID move (0.0833 / 0.1354 / 0.0625). A 32-sample eval has
  a binomial SE of ~6.4 points at p≈0.15 — it cannot resolve per-round
  changes of a few points; identical small counts are the expected outcome.
  The real-run config is already sized for this: 500 problems × 4 samples =
  2000 trials, SE ≈ 0.8 points (`conf/eval/math500.yaml`). Stated plainly:
  smoke eval numbers gate PLUMBING, not arm ordering; do not shrink the
  real-run eval, or the 8-round arm curves will be noise.
- **Disk is a hard constraint the smoke surfaced**: checkpoints are 3.6 GB;
  the real run would produce 24 ≈ 86 GB on a 74 GB disk. random_top4's
  round-0 checkpoint save crashed with ENOSPC at disk 100%. Fix: the driver
  now prunes round r−1's `model.safetensors` once round r's checkpoint
  exists (never read again; resume needs only the latest) — peak live
  weights ~7.2 GB per arm plus one retained final per finished arm.
  `check_run.py` expects weights only on each arm's latest trained round.

### 9a. Entropy is scored by the generating model — verified three ways

The selection signal is only meaningful if the entropy stage scores
trajectories under the SAME weights that generated them. Verified:

1. **Log audit, every entropy stage of the run** (both shards each):
   r00 → `Qwen/Qwen3.5-2B`, r01 → `round_00/checkpoint`,
   r02 → `round_01/checkpoint`. Exactly the expected r−1 pattern, no
   exceptions.
2. **Now auditable from artifacts, and asserted at runtime**: the rollout
   stage records its resolved model in `rollouts/model_path.json`; the
   entropy stage hard-fails if its own resolution differs, and writes
   `entropy/meta.json`. Logs rotate; the run dir survives.
3. **Statistical identity check**: for tokens sampled from a model,
   E[−log p] tracks that model's entropy. Under our truncated sampling
   (top_p 0.95 / top_k 20), −E[log p_full] = H(q) − log Z for the
   renormalized kept-mass distribution q, which sits slightly BELOW the
   full-distribution entropy H(p) that the entropy stage computes — the gap
   should be small and positive. Observed per trajectory (entropy_top4):

   | round | n | mean H | mean −logp | gap (H−NLL) | gap>0 | corr(H, NLL) |
   |---|---|---|---|---|---|---|
   | 0 | 96 | 0.5336 | 0.4418 | +0.0918 | 96/96 | 0.996 |
   | 1 | 96 | 0.2589 | 0.2157 | +0.0432 | 96/96 | 0.993 |
   | 2 | 96 | 0.2756 | 0.2276 | +0.0481 | 96/96 | 0.998 |

   Correlation ≈0.99+ with a uniformly positive, small gap of the predicted
   sign is very hard to fake with a mismatched scorer: scoring under any
   other model breaks this identity.

**Cross-round entropy caveat**: rounds score DIFFERENT prompt slices (the
pool advances 32 prompts per round), so round-over-round entropy drops
(0.53 → 0.26) conflate model sharpening with prompt difficulty. All checks
above are within-round; read `mean_entropy_selected` across rounds in the
arm curves with the same caution.

- **DDP rank balance: no persistent skew** (checked against a peer-observed
  100%/29% snapshot). Sampler time series, train windows only: entropy_top4
  rounds 0/1/2 per-GPU util means 34/36, 49/43, 26/43 (%); random_top4+all
  trains (87 paired samples) GPU0 38.2% vs GPU1 39.1%, sd ~22 each. Both
  ranks swing 0–100% in phase with the step (allreduce/checkpointing
  boundaries); only 6/87 pairs show >40-point skew — transient, alternating
  sides. No length-bucketing needed. The REAL observation: both ranks
  average only ~40% util at ~120 W during train at 2048 tokens — training
  is overhead-bound (gradient checkpointing, batch 1/rank, short steps),
  not GPU-bound; expect utilization to rise at 8192 (4× activations).
- **check_run.py: ALL CHECKS PASSED** on the completed run (all arms, all
  rounds, metrics.jsonl; retention-pruned rounds correctly tolerated).
  First natural EOS finishes ever observed: random_top4 round-1 rollouts
  94/96 cap-hit (2 finished), rollout accuracy 20/96 that round.

### 9b. Completed-run results (all three arms)

| arm | rnd | avg@2 | pass@2 | roll_acc | traj_cum | loss |
|---|---|---|---|---|---|---|
| entropy_top4 | 0 | 0.156 | 0.250 | 0.083 | 32 | 0.157→0.160 |
| entropy_top4 | 1 | 0.156 | 0.250 | 0.135 | 64 | 0.135→0.128 |
| entropy_top4 | 2 | 0.156 | 0.250 | 0.062 | 96 | 0.133→0.084 |
| entropy_top4 | T | 0.094 | 0.125 | — | 96 | — |
| random_top4 | 0 | 0.156 | 0.250 | 0.042 | 32 | 0.135→0.137 |
| random_top4 | 1 | 0.250 | 0.438 | 0.208 | 64 | 0.129→0.136 |
| random_top4 | 2 | 0.125 | 0.188 | 0.010 | 96 | 0.161→0.138 |
| random_top4 | T | 0.062 | 0.125 | — | 96 | — |
| all | 0 | 0.188 | 0.312 | 0.094 | 96 | 0.114→0.090 |
| all | 1 | 0.125 | 0.250 | 0.104 | 192 | 0.107→0.077 |
| all | 2 | 0.094 | 0.125 | 0.010 | 288 | 0.115→0.102 |
| all | T | 0.125 | 0.188 | — | 288 | — |

Read per §11: at 32 eval samples (SE ~6.3 points) NONE of the arm or round
differences are signal — the table gates plumbing (three arms, two
checkpoint→checkpoint transitions each, terminal evals, retention pruning,
resume across a mid-run crash + disk-full ENOSPC), all of which passed.
`roll_acc` additionally compares different prompt slices per round.

- **Round-0 eval identity**: entropy_top4 ≡ random_top4 EXACTLY (same driver
  process). `all` differs by one correct sample (0.188 vs 0.156 avg@2, 6/32
  vs 5/32) — it ran in the RESUMED driver process after the ENOSPC crash.
  Manifests confirm all three round-0 evals targeted `Qwen/Qwen3.5-2B` with
  identical seeds; the one-sample difference is vLLM's known non-bitwise
  batch-composition numerics across a process restart, not a model mix-up
  (the elementwise/decode verification in §3 rules that class out directly).

## 10. 8192 measurements (smoke8k, 2026-08-14) — TWO GATE FAILURES

`+experiment=smoke8k` (real 8192 cap, real num_samples 4, 8 prompts x 12,
eval 16x4). Round-0 rollout_eval and entropy completed; train OOM-crashed.

**Measured, per shard (one 2B engine per A100):**
- Generation throughput at 8192: 4422 / 4508 tok/s (vs ~4.4k at 2048 — the
  engine is still batch-bound, not length-bound, at this scale).
- rollout_eval round wall: 189.5 s (96 rollouts + 64 eval samples, both GPUs).
- Entropy stage at real lengths: 71.2 s.
- Eval avg@4 at 8192 (base model, 16 problems): LOOSE 0.469 (pass@4 0.625),
  STRICT (\boxed required) 0.250 — vs ~0.17 loose / ~0 strict at the 2048
  cap. Split by termination: finished rows 12/12 correct under BOTH gradings
  (extraction is perfect when traces conclude); truncated rows 0.346 loose
  vs 0.077 strict — the gap is the model stating the right answer
  mid-thinking and failing to stop (18/52 last-expression matches is far
  above chance). 8192 recovers real capability, but the loose number
  overstates it ~2x. Both are now in eval summary.json
  (`strict_avg_at_n`/`strict_pass_at_n`); policy (settled with peer, see
  §11): STRICT is the primary endpoint — loose is nearly blind to
  termination improvements, the dominant hypothesized effect — loose is
  secondary, and the strict-vs-loose gap is the per-round "knows it but
  won't stop" diagnostic.

**Gate 1 — cap-hit at 8192: FAILED (still ≈1.0).**
- Rollouts: 93/96 truncated (0.969); finish_reason `stop` only 3/96.
- Response-length quantiles [min, p10, p25, p50, p75, p90, max] =
  [2758, 8192, 8192, 8192, 8192, 8192, 8192] — pinned at the cap.
- Eval: cap_hit 0.8125, mean response 7858 tok.
- `no_answer_rate` 0.0 is NOT evidence of completion: `has_answer` uses the
  deliberately loose expression extractor (52/52 truncated eval rows "have an
  answer" but only 4/52 have `\boxed{}`). Truncated-and-correct (14/93
  rollouts, 18/52 eval) means the last extractable expression matched gold
  mid-thinking, not that the trace concluded.

**Gate 2 — DDP train at 8192: OOM on both ranks (NEW failure).**
- `per_device_train_batch_size=1` (already minimum), grad checkpointing on.
- Failed allocation: 7.64 GiB with ~71 GiB already allocated = one fp32
  full-vocab tensor (≈7.7k tok x 247,808 vocab x 4 B). GKD's generalized-JSD
  loss materializes several such tensors (student/teacher log-softmaxes +
  grads) on top of frozen teacher 9B (18 GB) + student + AdamW states.
  A single 8192-token sequence does not fit the current single-GPU layout.

**Resolution for gate 2 — Liger fused chunked JSD (option D1), adopted:**
TRL 1.10.0's experimental GKDTrainer natively supports
`LigerFusedLinearJSDLoss` via `use_liger_kernel`: the lm_head matmul is fused
into the divergence and computed in 1024-token chunks, so the full-vocab
logit tensors never exist. Verified before adopting:
- Beta convention identical: liger 0.8.1's `beta==1` branch is
  `F.kl_div(teacher_log_probs, student_log_probs, log_target=True)` =
  KL(student‖teacher) — the same reverse-KL line as TRL's eager special case.
  TRL zeroes the hard-CE weight (`weight_hard_loss=0.0`), keeping the loss
  soft-only, and neither path applies temperature.
- Empirical loss equivalence on the same real batch (2 selected smoke8k rows
  tail-truncated to 1200 tok, 2232 valid tokens): eager 0.15349653 vs liger
  0.15332031 — 0.11% relative in bf16.
- GRADIENT equivalence (the fused path computes the lm_head grad via a
  completely different code path, so forward equality alone is insufficient):
  bf16 — tied-embedding grad cos 0.99962 (rel 3.0e-2), mid-layer
  (layers.10.linear_attn.out_proj) cos 0.99976, global-norm ratio 0.9969.
  Falsification rerun with the student in fp32: gradients BITWISE IDENTICAL
  (max|Δ| = 0.0 on both tensors, global-norm ratio exactly 1.0, loss Δ
  1.5e-8). The bf16 deltas are accumulation-order numerics; the fused loss
  is exactly the same objective and the same backward.
- transformers' model-instance liger patching does not apply to qwen3_5
  (unsupported type → warning only); only the chunked loss is in effect.
- Added `liger-kernel==0.8.1` (resolves against torch 2.11.0+cu130);
  `use_liger_kernel: true` in conf/train/gkd.yaml.

**Options considered before adopting D1 (recorded for the decision trail;
sent to peer session):**
- A. Cap 4096: fits, cheap, but accepts ≈1.0 truncation as the design —
  reframe cap_hit as a tracked metric, not a gate. Dense per-token reverse KL
  on a truncated prefix is still a valid objective; all arms share the cap.
- B. Keep 8192, restructure memory: frozen teacher alone on GPU 1, student
  trains single-process on GPU 0 (grad_accum 16 keeps effective batch 16).
  Frees ~18 GB; loses DDP; moderate train-stage change; still tight.
- C. Disable thinking mode: everything finishes and fits, but changes what is
  being distilled.
**Natural finish-length distribution (base 2B, 24,576 budget, 16 prompts x
2, run's sampling params):** 62.5% (20/32) still truncate at 24,576. Only
9.4% finish under 8192, 12.5% under 12,288, 18.75% under 16,384. Among the
12 natural finishes: min 2,486, median 16,282, p90 18,465, max 20,857. The
length distribution is effectively unbounded at temperature 1.0 — NO
practical cap eliminates truncation, so truncated-prefix distillation is the
design, not a compromise. Consequence: the cap-hit gate is retired to a
tracked per-round metric (if distillation works, cap_hit should FALL across
rounds — the 9B presumably terminates better, and that behavior is part of
what reverse KL should transfer). Pending user confirmation via peer.

**Upstream bug found while fixing gate 2: TRL's liger integration does not
actually chunk for small batches.** liger 0.8.1's fused_linear_distillation
computes `num_chunks = max(1, student_input.shape[0] // CHUNK_SIZE)` and
chunks over dim 0, expecting `[rows, H]` — TRL passes 3D `[B, T, H]`. With
per-device batch 1, shape[0] == 1 → ONE chunk holding the entire sequence →
the full-vocab fp32 logits materialize inside `torch.func.grad_and_value`
anyway (the "chunked" loss silently degenerates exactly when needed most).
Memory bisect at 8276 tokens, single GPU: models 20.2G → student forward
(gc) 21.7G peak → teacher forward 22.8G peak → liger loss forward **75.7G
peak**. Fix: flatten to `[B*T, H]`/`[B*T]` before the loss (mathematically
identical: independent positions, ignore_index masking, same normalization
set) via a thin wrapper around `trainer.liger_loss` in train.py — no TRL
fork. Verified: loss 0.100769 vs 0.100586 (0.18%, chunk-boundary fp32
accumulation order), loss-forward peak 75.7G → 28.9G; backward peak 26.9G.
Candidate for upstreaming to TRL.

**Measured train at 8192 with the fixed liger loss (smoke8k rerun):**
- Step time 94.4 s at effective batch 16 (2 ranks x 1 x 8, ~8.2k-tok
  sequences); stage wall 194.8 s for 2 steps — trainer init + teacher load +
  checkpoint save add only ~6 s. Loss mean 0.102, grad_norm ~1.7, steady
  ~34 GiB/GPU (vs 80 GiB OOM before the flatten fix). tail_truncated_rows 0.

**Projected real-run wall clock (all inputs measured, not assumed):**
Inputs: gen throughput 8,930 tok/s combined (4,422 + 4,508 per shard at
8192); mean rollout length ~7,900 tok; mean eval length ~7,860 tok; train
94.4 s/step at effective 16; entropy ~10.4k tok/s (71.2 s for 0.74M tok).
Real config: 128 prompts x 12 rollouts, eval 500x4, 8 rounds + terminal
eval, arms sequential.

Per round, per arm:
- rollouts: 1536 seqs x 7.9k = 12.1M tok → ~23 min
- eval 500x4: 2000 seqs x 7.86k = 15.7M tok → ~29 min (every round + terminal)
- entropy (entropy_top4 only): 12.1M tok → ~19 min
- train top-4 arms: 512/16 = 32 steps → ~51 min
- train `all` arm: 1536/16 = 96 steps → ~2.5 h

Per-arm totals (8 rounds + terminal eval):
- entropy_top4: ~16.8 h; random_top4: ~14.2 h; all: ~27.7 h
- **TOTAL ≈ 59 h (~2.5 days) of continuous 2-GPU time**, before any
  failure/restart slack.

Levers, with the user's decisions (relayed 2026-08-14):
1. **Eval cadence — APPROVED and IMPLEMENTED** (~8 h saved): full 500x4 at
   round 0 and terminal; `eval.intermediate_num_problems: 100` (prefix of
   the materialized set, clamped to num_problems, fingerprinted) for rounds
   1-7 → intermediate evals 29→6 min. The pre-registered primary endpoint
   is TERMINAL strict avg@4, which keeps full precision.
2. **Round-0 eval dedup — APPROVED and IMPLEMENTED** (~1 h free): round 0
   evaluates the identical base model in every arm, so the driver copies
   the first completed arm's round-0 eval rows+markers into later arms
   (`eval/reused_from.json` records provenance); the stage's resume path
   still validates the copied rows. Rollouts are NOT shared (they are each
   arm's training data).
3. **The `all` arm stays** (~20 h of the ~34 h training subtotal, 3x data
   at the same step time) — the deliberate upper bound, now a knowing
   choice.
4. **Rounds stay 8 for now**; decide 8-vs-5 (~21 h) after smoke3 shows
   whether the arm curves separate early or late. Plain config value,
   deferring costs nothing.

The flatten wrapper itself was gradient-re-verified (the wrapper changes the
chunk structure, so the earlier unflattened proof did not cover it): fp32,
2916 valid tokens = 3 chunks of accumulation — loss Δ 7.5e-9, mid-layer grad
BITWISE identical, tied-embedding grad max|Δ| 5.4e-8 (single-ulp
accumulation ordering), global grad-norm ratio exactly 1.0.

**Teacher-accuracy gate: PASS (2026-08-14).** Qwen3.5-9B, eval-only
pipeline run (run_name=teacher_eval, 100 MATH-500 problems x 1 sample,
the run's exact sampling config and 8192 cap): pass@1 STRICT 0.60 / loose
0.68, cap_hit 0.47, mean response 6,363 tok. By difficulty level (strict /
loose / cap_hit): L1 1.00/1.00/0.00 (n=8), L2 0.88/0.94/0.25 (16),
L3 0.68/0.79/0.37 (19), L4 0.55/0.68/0.50 (22), L5 0.37/0.43/0.71 (35) —
clean monotone profile, no dead band. Versus the student's strict 0.25:
~35 points of strict headroom overall, and the teacher both terminates far
better (cap_hit 0.47 vs 0.81) and answers more — the exact behavior profile
reverse-KL distillation is supposed to transfer. Teacher accuracy is a
CEILING for what distillation can reach: at 0.60 strict the ceiling is
comfortably above everything the arms could plausibly do in 8 rounds.

All launch gates are now passed or retired. GO for smoke3, then the real
run on a clean smoke3. (A presence_penalty=1.5 length control was planned to test
whether ADR 0004's penalty removal caused the non-termination; USER
DECISION 2026-08-14: penalty stays 0.0 for the real run and its
implementation is retained-but-disabled, so the control was not run — the
decision was made on other grounds and the GPU time buys nothing.)

## 11. Pre-registered decision rule, gates, and known confounds

Written BEFORE the real run so the analysis cannot be post-hoc.

**Primary metric**: MATH-500 STRICT avg@4 (`\boxed` required) at the final
round (terminal eval), per arm; 500×4 = 2000 samples; at the round-0 strict
level of ~0.25, SE ≈ ±0.97 points, so the 3-point MDE is ~3 SE. Strict is
primary because the dominant round-0 failure mode is "computes the answer
then fails to stop" (§10): a model that learns to terminate converts
loose-credits into strict-credits, so LOOSE barely moves under the main
hypothesized effect while strict moves sharply — loose as primary would be
nearly blind to it. Loose avg@4 is secondary; the strict-vs-loose gap is
the "knows it but won't stop" diagnostic; pass@4 (both gradings) is the
diversity/mode-collapse monitor. Interpretation rule fixed in advance: a
strict null with a loose difference reads as "answer-knowledge moved
without termination improving" and is reported as such, not upgraded to a
positive result; the primary claim is keyed to strict only.

**Minimum detectable effect**: with one seed per arm, differences below
~3 points between arms are not separable from noise. Decision rule:
- `entropy_top4 − random_top4 ≥ 3 points` → entropy selection carries signal.
- `|entropy_top4 − random_top4| < 3 points` → null; consult the oracle-KL
  diagnostic to attribute it (entropy a bad proxy vs selection premise weak).
  Do NOT reinterpret secondary metrics to rescue the result.
- `all` above both is EXPECTED and uninformative about selection (see
  confounds); `all` at or below the top-4 arms would be notable (data
  quality > quantity at fixed budget).
- Single seed means run-to-run variance is unmeasured; if the observed
  effect lands near the threshold, the follow-up is a repeat seed of
  entropy_top4 and random_top4, not a stronger claim.

**Gates before launch** (each cheap now, expensive to discover mid-run):
1. **Teacher accuracy on MATH-500** (pass@1/pass@4, by difficulty level)
   under the run's sampling config. The teacher bounds every arm; ppl gap
   (1.43 vs 1.53) and reverse KL (0.152) are small, so confirm there is
   headroom to transfer before spending 24 training rounds. Weak teacher →
   change teacher or dataset first.
2. ~~cap_hit_rate at 8192~~ **RETIRED as a gate (USER DECISION 2026-08-14)**
   after measurement: cap_hit is 0.969 at 8192 AND 62.5% of traces still
   run past a 24,576 budget (§10) — truncation is intrinsic to the base
   model's sampling behavior, so no practical cap eliminates it and
   truncated-prefix distillation is the design. `cap_hit_rate` and the
   strict-vs-loose gap are TRACKED PER-ROUND METRICS instead: if
   distillation works, cap_hit should fall and the gap should narrow —
   two independent signals of the same termination improvement, which is a
   stronger claim than either alone. The run stays at max_new_tokens=8192;
   scaling beyond 8192 is a deliberate follow-up conditional on supportive
   results (liger makes the memory tractable — the binding constraint is
   wall clock, roughly linear in tokens — not something ruled out).
(The oracle-KL diagnostic was initially a third gate; USER DECISION
2026-08-14: run it later as a separate experiment over stored rollouts, not
as a launch prerequisite. It remains the attribution tool for a null result.)

**Known confounds, stated so results are not over-read**:
- **vLLM cross-process nondeterminism floor** (measured at round 0, the one
  point where arms are provably identical): re-evaluating the SAME model in
  a fresh engine process flips ~1 sample in 64 (smoke8k round-0: `all`
  0.45313 vs 0.46875 in the other two arms, mean_response_length differing
  in the decimals proves a genuine recompute) and flipped 1 in 32 once in
  smoke2 — a per-sample flip rate of roughly 1.5-3% from batch-composition
  numerics. Round-0 evals are now deduplicated (identical by construction),
  but every later-round arm comparison crosses processes, so this floor
  applies: at 2000 samples, ~30-60 flips with roughly balanced direction
  contribute ~0.3-0.4 points of run-to-run SD — below the ±0.97 sampling SE
  and inside the 3-point MDE, but a real noise source independent of
  sampling error. (Verify the dedup from smoke3 artifacts:
  eval/reused_from.json in arms 2-3 and byte-identical round-0 rows.)
- `rollout_accuracy` is NOT comparable across rounds: each round rolls out a
  fresh 128-prompt slice, so round-over-round movement mixes model change
  with slice difficulty. Eval avg@4 (fixed MATH-500) is the only
  cross-round-comparable accuracy. Same caveat for `mean_entropy_selected`.
- The `all` arm is NOT compute-matched: 1536 trajectories/round vs 512 (3×
  data and ~3× training compute). It is a data/compute upper bound, kept
  deliberately (user decision); random_top4 is the like-for-like selection
  control (it IS "all" subsampled to 512 uniformly).
- Smoke arm differences are NOT signal: smoke2's 32-sample eval has SE ~6.3
  points (observed 5/32, 5/32, 5/32, 3/32 across genuinely different
  checkpoints). smoke3's config raises this to 400 samples (SE ~1.8) and
  matches the real avg@4 protocol.

## 12. smoke3 results and launch decision

**smoke3 validated the plumbing, NOT the method.** Clean across all of it:
checkpoint-to-checkpoint train transitions at 8192 with the fused loss
(losses 0.06-0.22, no drift), round-0 eval dedup fired twice
(`reused from arm 'entropy_top4'`, 131 s vs 334 s round wall), intermediate
eval subset, retention, resume, ~7.7k gen tok/s per shard (above the 4.45k
used in the ~59 h projection — real run likely faster).

**The method result at smoke scale is adverse.** All three arms,
strict / loose / pass@4 / cap_hit / mean len (100 problems x 4 samples per
round; r0 is the shared base-model eval, dedup-copied across arms):

| round | entropy_top4 | random_top4 | all |
|---|---|---|---|
| r0 base | 0.240 / 0.385 / 0.62 / 0.80 / 7598 | (same, reused) | (same, reused) |
| r1 | 0.043 / 0.258 / 0.61 / **1.00** / 8192 | 0.335 / 0.390 / 0.62 / 0.71 / 7121 | 0.138 / 0.273 / 0.50 / **0.99** / 8168 |
| r2 | 0.248 / 0.378 / 0.65 / 0.77 / 7162 | 0.328 / 0.380 / 0.59 / 0.73 / 6748 | 0.263 / 0.338 / 0.54 / 0.87 / 7553 |
| r3 terminal | **0.113** / 0.205 / 0.45 / 0.91 / 7830 | **0.153** / 0.205 / 0.44 / 0.97 / 8136 | **0.058** / 0.153 / 0.32 / 0.99 / 8176 |

(check_run.py on the finished run: ALL CHECKS PASSED, metrics.jsonl 12
rows complete.)

Reading of the table:

- **Every arm suffers a termination collapse (cap_hit → ~1.0, strict
  crashes), but on a DIFFERENT schedule**: entropy at r1 (recovers r2,
  re-collapses r3), `all` at r1 (partial recovery r2, re-collapses r3),
  random not until r3. ALL THREE arms end BELOW baseline on the
  pre-registered primary endpoint, with pass@4 (the mode-collapse
  monitor) down 0.62 → 0.45/0.44/0.32. The `all` arm — 3x the data per
  round — has the WORST terminal, so more smoke-scale data alone did not
  stabilize it.
- **The arms separate immediately — at round 1 — and with opposite
  signs**: after one identical-size update, entropy's selected-for-entropy
  data produced a full collapse (strict −0.20) while random's produced the
  best number any arm reached (+0.095 strict over baseline, sustained two
  rounds). `all` (3x the data per round) landed in between. Caveat: smoke3
  is 8 prompts/round vs the real run's 128, so smoke separation timing is
  weak evidence for real-run timing — but it says round-1 metrics of the
  real run are already informative, not burn-in to be ignored.
- random_top4's two good rounds show the method WORKS in-regime: distilling
  on-policy at 8192 lifted strict avg@4 by ~9.5 points in one round. The
  failure mode is the instability, not the objective.

Mechanism is OPEN — candidate stories and their status:
- data-composition EOS erasure: falsified by entropy's r1→r2 recovery on
  equally EOS-free (32/32 truncated) training data, and by random's r3
  collapse from its LEAST-truncated data;
- first-update optimizer shock: falsified by gradient logs (the
  collapse-producing round is the calmest, max grad-norm 3.27; the biggest
  spike, 9.81, precedes the recovery);
- surviving candidates: termination SATURATION/oscillation (EOS-free data
  pushes stop probability down when present; teacher stop-mass — teacher
  cap_hit 0.47 — pulls it up from the floor; small rounds overshoot) and
  iterated reverse-KL mode-seeking (beta=1.0). Against the latter: answer
  diversity is FLAT across rounds (9.38/9.12/9.25 distinct answers per 12
  rollouts). Diagnostic queued: stop-mass probe (student p(stop) at
  teacher-favored stop positions, base vs each round checkpoint).

**Launch decision (user, 2026-08-15): LAUNCH the real run, with effective
batch raised 16 → 32** (`gradient_accumulation_steps: 16`, asserted
composition 2 ranks x 1 x 16). Rationale: smoke3's per-round update is 32
trajectories / 8 steps at effective batch 4 — the real run trains on 512
trajectories (top-4 arms) at effective 32, i.e. 16x the data averaged over
8x larger batches per step, which directly targets the oscillation. This
is a hypothesis the real run tests, stated here before its results are
known. No abort rule: per-round metrics are watched and discussed, the
run is not auto-stopped.

## Appendix: first-smoke reference numbers (outputs/runs/smoke)

- check_run.py: ALL CHECKS PASSED (all arms, both rounds, metrics.jsonl).
- Presence-penalty processor verification (while it was on): PASS —
  32,768 tokens bit-identical between the incremental processor and vLLM's
  native penalty path. The processor is now dormant (ADR 0004).
- Train stage (single-GPU, 1024 cap, penalty-on data): ~11.6 s/step,
  ~352 tok/s, loss 0.328→0.302 (entropy_top4), 0.337→0.314 (random_top4),
  0.274→0.227 (all, 64 trajectories); GPU 0 SM ~30% avg, 40.8/81.9 GB;
  GPU 1 idle — the motivation for DDP.
- Rollout throughput per engine at 64 concurrent seqs: ~3,290 gen tok/s
  (chunk logs); eval-phase 16-seq batches: ~1,800 tok/s warm.
- entropy_top4 accuracy moved 0.062→0.094 avg@2, 0.062→0.188 pass@2 after
  one round (smoke-scale noise, but the loop moves and the reload serves
  trained weights).
