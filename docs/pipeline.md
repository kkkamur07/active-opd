# Pipeline contract

Interface spec for the APOD experiment pipeline. Every stage script, the
driver, and the plot script are written against this document. Design
rationale lives in `CONTEXT.md` and `docs/adr/`; settings live in `conf/`
(nothing is hard-coded).

## Round semantics

- `round_XX/` holds the work of round X: an **eval of the model the round
  starts with** (base student for X=0 — the anchor), rollouts from that same
  model, scoring, selection, one GKD training pass, and the **checkpoint
  produced at the end of round X**.
- After the last training round, the driver runs an eval-only round
  `round_{rounds:02d}` (just `eval/`) to measure the final checkpoint.
- Plot point r = (cumulative trajectories trained through round r−1,
  avg@4 from `round_r/eval`). Point 0 is the untrained anchor.
- Arms run sequentially; within a stage, work is sharded across `num_gpus`
  processes by `example_index % num_shards == shard` (eval:
  `problem_index % num_shards == shard`).
- Prompt pool: sampled once at run start (`load_examples("openthoughts",
  n=rounds*num_prompts, seed=pool_seed)`); round r consumes slice
  `[r*num_prompts, (r+1)*num_prompts)`. All arms see identical prompts in a
  given round.

## Directory layout

```
outputs/runs/<run_name>/
  resolved_config.yaml            # OmegaConf dump; stages read THIS, not conf/
  pool/prompts.jsonl
  pool/eval_problems.jsonl        # MATH-500 monitor set (cfg.eval), materialized once
  pool/eval_problems_<dataset>.jsonl   # named eval sets (aime2526), materialized once
  metrics.jsonl                   # one row per (arm, round); driver appends
  plots/accuracy_vs_steps.png
  arms/<arm>/rounds/round_XX/
    manifest.json                 # driver: config stamp, timings, throughput
    eval/eval.shard{K}.jsonl
    eval/summary.json             # driver merges shards -> avg@4, pass@4, ...
    eval_<dataset>/eval.shard{K}.jsonl   # named set (--eval-dataset aime2526), same schema
    rollouts/trajectories.shard{K}.jsonl
    rollouts/tokens/example_{example_index:05d}.npz
    entropy/entropy.shard{K}.jsonl
    selected/selected.jsonl
    train/log_history.jsonl
    train/summary.json
    checkpoint/                   # HF save_pretrained(model) + tokenizer
  terminal_eval/cap<N>[_k<K>]/    # scripts/terminal_eval.py: a derived run dir
    resolved_config.yaml          #   (cap N, optional k override stamped in) whose
    pool/eval_problems*.jsonl     #   round_{R+1} evaluates a symlinked round_R/checkpoint
    arms/<arm>/rounds/round_{R+1}/eval/, eval_aime2526/, terminal_summary.json
```

Model path for round X stages: `round_{X-1}/checkpoint` if it exists, else
`cfg.model.student_id` (X=0). Resume: each stage writes an empty
`done.shard{K}` marker in its stage dir on success; with `resume: true` the
driver skips any stage whose markers are all present.

## File schemas

`pool/prompts.jsonl` — one row per prompt:
`{example_index, id, prompt, reference, round}`
(`example_index` is global across the run: `round*num_prompts + i`.)

`rollouts/trajectories.shard{K}.jsonl` — one row per trajectory:
`{example_index, rollout_index, id, prompt_length, response, response_length,
truncated, finish_reason, correct, has_answer, has_boxed, seed}`
Grading via `apod.verification.grade` at generation time; `truncated` rows
are graded (incorrect unless boxed answer appeared) but stay eligible for
selection/training (ADR 0002).

`rollouts/tokens/example_XXXXX.npz` — `apod.datasets.io.save_npz` batch:
`input_ids [num_rollouts, width] int32` (prompt+response, right-padded),
`prompt_length`, `response_lengths`, `truncated`, `responses`.

`eval/eval.shard{K}.jsonl` — one row per (problem, sample):
`{problem_index, sample_index, id, response_length, truncated, correct}`
(responses themselves are not persisted for eval; grade at generation time).

`eval/summary.json` (driver merges shards):
`{avg_at_n, pass_at_n, num_problems, num_samples, cap_hit_rate,
mean_response_length}`

`entropy/entropy.shard{K}.jsonl` — one row per scored trajectory:
`{example_index, rollout_index, entropy, mean_logprob, scored_tokens}`
where `entropy` = mean over response positions of full-vocab
`Entropy(pi_S(.|x, y_<t))` under the round's starting model.

`selected/selected.jsonl` — one row per kept trajectory:
`{example_index, rollout_index, entropy|null, correct, truncated,
response_length}` sorted by (example_index, rollout_index).

`train/log_history.jsonl` — one row per optimizer step (logging_steps=1),
the trainer's own row plus the batch diagnostics of `DiagGKDTrainer`
(`cfg.train.diag_every`, `diag_chunk`):
`{loss, grad_norm, learning_rate, epoch, step, response_tokens,
cap_hit_frac, overlap_ratio_top16, overlap_adv_top16, abs_entropy_gap,
bf16_rounded_frac, bf16_rounded_frac_<block>}`
where `step` is the trainer-local step (1-based within this launch),
`bf16_rounded_frac` the fraction of trainable elements whose Adam update
this step (lr · m̂ / (√v̂ + ε), bias-corrected, from the post-step state)
is smaller than half a bf16 ulp of the weight and is therefore lost in
pure-bf16 training, broken down by `<block>` ∈ {embeddings, attention,
mlp, lm_head, other} (attention covers both full attention and Gated
DeltaNet layers; other is norms and the GDN scalar vectors),
`response_tokens` the labelled response tokens of the effective batch (all
ranks, all micro-batches), `cap_hit_frac` the fraction of the step's
trajectories with the npz `truncated` flag, and the three distribution
diagnostics are token-weighted means over those response tokens, computed
by a no_grad chunked lm_head pass over the same hidden states the Liger
fused loss consumes: `overlap_ratio_top16` = mean |top16_S ∩ top16_T| / 16,
`overlap_adv_top16` = mean over tokens with a non-empty intersection of
−KL(p̄_S ‖ p̄_T) / |∩| with both distributions renormalized over the
intersection (Rethinking OPD Eq. 6-7, exactly `scripts/oracle_kl.py`'s
per-token math), `abs_entropy_gap` = mean |H_S − H_T| in nats. The three
are absent on steps `diag_every` skips and on the eager (non-Liger) loss
path. A trailing row without `step` carries the trainer's runtime summary.

`train/summary.json`:
`{num_trajectories, tokens_trained, tail_truncated_rows, train_loss_mean,
train_loss_final, wall_clock_s}`

`metrics.jsonl` — one row per (arm, round), appended by driver:
`{arm, round, trajectories_round, trajectories_cumulative, tokens_trained,
avg_at_n, pass_at_n, eval_cap_hit_rate, rollout_cap_hit_rate,
rollout_accuracy, mean_entropy_selected|null, train_loss_mean|null,
train_loss_final|null, wall_clock: {rollout_eval_s, entropy_s, train_s},
rollout_throughput_tok_s}`
(the eval-only final round writes eval fields with train fields null).

## Stage CLIs

Stages are plain scripts (NOT Hydra apps — only the driver is); they load
`resolved_config.yaml` with `OmegaConf.load`. The driver launches them via
`subprocess` with `CUDA_VISIBLE_DEVICES` set to the shard's GPU.

```
python -m apod.stages.rollout_eval --run-dir D --arm A --round R --shard K --num-shards N [--eval-only] [--eval-num-problems M] [--eval-dataset NAME]
    one vLLM engine session and ONE generate stream: MATH-500 eval requests
    first, then rollouts (skipped with --eval-only for the final round),
    packed into target_concurrent_sequences-sized chunks that may hold
    both; separate files and done-markers per kind; prints throughput.
    --eval-dataset NAME (a conf/eval/NAME.yaml key other than cfg.eval.dataset,
    e.g. aime2526) evaluates pool/eval_problems_NAME.jsonl under NAME's own
    protocol (num_problems, num_samples, seed offset; cfg.eval_sets.NAME in
    resolved_config.yaml overrides the conf file) into eval_NAME/; the default
    keeps eval/ byte-identical. The AIME 2025+2026 monitor (ADR 0006) is a
    second --eval-only launch of this stage per refresh, at the run cap
python -m apod.stages.entropy      --run-dir D --arm A --round R --shard K --num-shards N
    HF forward entropy scoring of that round's trajectories (run only for
    entropy_top4 unless selection.score_all_arms)
python -m apod.stages.train        --run-dir D --arm A --round R [--global-step-offset S]
    single process on cfg.train.train_gpu (torchrun DDP with num_gpus > 1);
    GKDTrainer over selected trajectories; writes checkpoint/ +
    train/log_history.jsonl + summary. Every optimizer step is logged with
    batch diagnostics (see the log_history schema) and forwarded to the
    arm's W&B run at global step S + trainer step, where S is the number of
    training steps the arm took before this launch (the caller's running
    total across refreshes; default 0)
python -m apod.main            (Hydra app, conf/config.yaml)
python -m apod.plotting        --run-dir D
    reads metrics.jsonl -> plots/accuracy_vs_steps.png; one curve per arm,
    x = training step (trajectories / effective_batch, 32), y = avg_at_n
python scripts/terminal_eval.py --run-dir D --arm A [--round R] [--max-new-tokens N] [--num-samples K] [--gpus 0,1]
    evaluates round_R/checkpoint (default: newest with weights) on MATH-500
    avg@4 + AIME 2025/2026 avg@16 under the monitor protocol (run cap) or
    with the overrides for a headline table (ADR 0006): launches
    rollout_eval --eval-only per dataset in D/terminal_eval/cap<N>[_k<K>]/,
    then prints and writes a strict table (avg@k, naive SE, question-level
    cluster bootstrap 95% CI, pass@k, cap-hit, mean length; per-year split)
```

## Tracking interface (`apod/tracking.py`, W&B)

One W&B run per arm, shared by every process that logs for it: the driver
(refresh evals) and each train-stage launch (per-step rows). Settings in
`conf/tracking.yaml` → `cfg.tracking` (`mode` online | offline | disabled,
default offline; `project`, `entity`, `experiment`, `dir`). Every function
is a no-op when `mode: disabled`, when wandb is not installed (a warning is
logged once), or in a non-zero torchrun rank, so nothing depends on the
package (it ships in the `train` extra).

```python
from apod import tracking

tracking.init(cfg, run_dir, arm) -> bool
    # Start or RESUME the arm's run in this process. project from cfg.tracking,
    # group = run_dir.name (e.g. r1-correctness-8k), name = f"{run_dir.name}/{arm}",
    # job_type = arm, tags = [f"cap{cfg.sampling.max_new_tokens}", experiment],
    # config = the resolved cfg, id = tracking.run_id(run_dir, arm): deterministic
    # from (run dir name, arm), resume="allow", so restarts and later processes
    # continue the same run. Returns True when a run is live.
tracking.log_step(step, {"loss": ..., "grad_norm": ..., ...})
    # Per-training-step scalars. step = the GLOBAL training step (across
    # refreshes); keys without a "/" are namespaced "train/<key>". The train
    # stage calls this itself for every optimizer step at
    # --global-step-offset + trainer step, with every numeric log_history field.
tracking.log_refresh(step, {"eval/math500/strict_avg4": ..., ...})
    # Refresh eval scalars at global step `step` (the number of training steps
    # the evaluated weights have taken), keys as given. Convention:
    #   eval/math500/strict_avg4, eval/math500/strict_pass4,
    #   eval/math500/cap_hit_rate, eval/math500/mean_response_length,
    #   eval/aime/strict_avg16, eval/aime/strict_pass16, eval/aime/cap_hit_rate
tracking.finish()
    # Flush and close this process's handle; the run stays resumable.
```

Driver contract: `init` once per arm before its first `log_refresh`, `finish`
before the arm ends. The train stage does its own `init`/`finish` inside its
subprocess. Do not log from the driver while a train stage of the same arm
is running (the pipeline is sequential, so this is free). All scalars are
logged against a `global_step` metric declared with `define_metric`, never
W&B's internal step, so processes cannot trip its monotonic step check.

Offline runs (the default) write `<run_dir>/wandb/offline-run-*-<id>`, one
directory per process; upload them with
`wandb sync <run_dir>/wandb/offline-run-*` — every directory of an arm
carries the same id, so they fill in one run.

## Selection interface (`apod/selection.py`)

```python
def select_trajectories(arm, trajectories, *, k, num_rollouts, seed):
    """trajectories: merged rows for one round (entropy merged in when the
    arm needs it). Per example_index keep k of num_rollouts:
      entropy_top4: highest entropy, ties -> lower rollout_index
      random_top4:  default_rng(seed + example_index).choice, no replacement
      all:          keep everything
    Truncated rows are eligible. Returns selected.jsonl-shaped rows."""
```

## File ownership (parallel agents — do not cross-edit)

- rollout/eval agent: `apod/stages/rollout_eval.py`
- scoring/selection agent: `apod/stages/entropy.py`, `apod/selection.py`
- driver agent: `apod/main.py`, `apod/plotting.py`
- train stage (main session): `apod/stages/train.py`, `conf/`
- `pyproject.toml`, `CONTEXT.md`, ADRs, existing `apod/` modules: main
  session only. Reuse existing helpers (`apod.models.generate_vllm`,
  `apod.datasets.io`, `apod.verification`) rather than duplicating them.
```
