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

`train/summary.json`:
`{num_trajectories, tokens_trained, train_loss_mean, train_loss_final,
wall_clock_s}`

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
python -m apod.stages.train        --run-dir D --arm A --round R
    single process on cfg.train.train_gpu; GKDTrainer over selected
    trajectories; writes checkpoint/ + train/log_history.jsonl + summary
python -m apod.main            (Hydra app, conf/config.yaml; round-based, superseded by apod.driver)
python -m apod.driver +experiment=<r1_correctness_8k|r2_trajsel_8k|r3_qentropy_8k>
    step-based driver (see "Step-based driver" below)
python -m apod.plotting        --run-dir D [--band-points P]
    reads metrics.jsonl -> plots/accuracy_vs_steps.png; one curve per arm,
    x = training step. Driver runs: strict avg@n per eval set with a
    P-point noise band (default driver.noise_band_points) + cap-hit panel;
    apod.main runs: avg@n / pass@n with x = trajectories / effective_batch
python scripts/terminal_eval.py --run-dir D --arm A [--round R] [--max-new-tokens N] [--num-samples K] [--gpus 0,1]
    evaluates round_R/checkpoint (default: newest with weights) on MATH-500
    avg@4 + AIME 2025/2026 avg@16 under the monitor protocol (run cap) or
    with the overrides for a headline table (ADR 0006): launches
    rollout_eval --eval-only per dataset in D/terminal_eval/cap<N>[_k<K>]/,
    then prints and writes a strict table (avg@k, naive SE, question-level
    cluster bootstrap 95% CI, pass@k, cap-hit, mean length; per-year split)
```

## Question bank

`python -m apod.bank` (module `apod/bank.py`, settings `conf/bank.yaml` under
`cfg.bank`) builds the labelled question pool of the correctness experiment
(ADR 0005) at `outputs/runs/<bank.name>/` (`bank-8k`). It is not a run: no
arms, no training steps. It is built once and read by the drivers of runs 1
and 3.

Build loop (resumable; each step is one `apod.bank` worker subprocess per GPU
with `CUDA_VISIBLE_DEVICES` pinned, sharded by `example_index % num_shards`;
one vLLM engine per process, torn down before the next step):

1. `pool/questions.jsonl` — every usable OpenThoughts question in one seeded
   order (`data.pool_seed`, a permutation, so raising `bank.student_questions`
   later extends the same order without regenerating anything).
2. Student sweep, chunk by chunk (`chunk_questions`, 1000) up to
   `student_questions` (14,500): `num_rollouts` (4) student rollouts at
   `max_new_tokens` (8192), reusing the rollout stage's `RolloutWriter` +
   `run_session` (request packing, chunk seeds with the chunk index as the
   seed round and `chunk_questions` as the stride, grading pool overlapped
   with generation, npz + EOS repair, done-markers). After each chunk, its
   entropy: `student/entropy/` via the entropy stage's `trace_scores` under
   the HF student, H(q) = mean H(tau) over the question's rollouts.
3. Teacher sweep, narrow: each teacher chunk is the next `chunk_questions`
   *eligible* questions in pool order, where eligible = not teacher-swept,
   student label C or W, and some bucket the teacher label could put it in
   (C -> both_right or mixed; W -> teacher_right_student_wrong, both_wrong or
   mixed) is still below `target_per_bucket` (800). Student-M questions are
   mixed without a teacher sweep. The plan of chunk C is persisted in
   `teacher/chunks/chunk_C.json` before it runs, so a resume reruns the same
   chunk. Stops when every bucket is full or nothing is eligible.

Labels (strict, CONTEXT.md): a rollout is correct iff `\boxed` present,
Math-Verify accepts it, and not cap-hit. Per model, `>= correct_min` (3) of
`num_rollouts` correct = C, `<= wrong_max` (1) = W, else M. Buckets:
TC/SW `teacher_right_student_wrong`, TC/SC `both_right`, TW/SW `both_wrong`,
`mixed` = any labelled question in none of those (student-M, either model M,
TW/SC), `unlabelled` = student swept, teacher not (yet).

```
outputs/runs/bank-8k/
  resolved_config.yaml            conf/ composed; rollout.num_rollouts, rollout.num_prompts
                                  (= chunk_questions, the seed stride) and
                                  sampling.max_new_tokens stamped from bank.*
  pool/questions.jsonl            {example_index, id, prompt, reference}  (whole dataset, seeded order)
  questions.jsonl                 the bank: one row per student-swept question, pool order
  student/trajectories.shard{K}.jsonl, student/tokens/example_XXXXX.npz   rollout layout (schemas above)
  student/done.chunk{CCC}.shard{K}
  student/entropy/entropy.shard{K}.jsonl, done.chunk{CCC}.shard{K}, meta.json
  teacher/trajectories.shard{K}.jsonl, teacher/tokens/, done.chunk{CCC}.shard{K}
  teacher/chunks/chunk_{CCC}.json  {chunk, example_indices, bucket_counts}
```

`questions.jsonl` — one row per student-swept question:
`{example_index, id, question, reference, chunk, student_grades[4],
student_truncated[4], student_lengths[4], teacher_grades[4]|null,
teacher_truncated[4]|null, teacher_lengths[4]|null, student_label C|W|M,
teacher_label C|W|M|null, bucket, question_entropy|null}` — grades are
strict booleans per rollout_index; `question` is the prompt text the
rollouts were rendered from (the pool row's `prompt`, boxed instruction
included); `chunk` = `example_index // chunk_questions`. Rewritten from the
raw shards by every build step and by `--report`.

```
python -m apod.bank [--gpus 0,1] [bank.student_questions=N] [bank.target_per_bucket=N]
    build or resume (a rerun with nothing pending issues no requests); only
    those two bank.* settings may change on an existing bank -- a different
    cap/rollout count is a different bank (bank.name)
python -m apod.bank --report [--bank-dir D]
    relabels, then prints bucket counts vs target, per-bucket cap-hit
    composition (student, teacher, teacher 4/4 cap-hit), and the remaining
    generation: student questions left, teacher questions the unfilled
    buckets still need at the bank's own P(teacher label | student label),
    hours at bank.student_trajectories_per_min (118 on 2 GPUs at 8192; the
    teacher bank.teacher_slowdown = 2.5x slower), and how far to raise
    student_questions when a bucket cannot fill from the current sweep
python -m apod.bank --bank-dir D --sweep student|teacher --chunk C --shard K --num-shards N
python -m apod.bank --bank-dir D --entropy --chunk C --shard K --num-shards N
    the workers the build launches (one GPU each)
```

Reading the bank from a driver (`apod.bank`):

```python
rows = load_bank(bank_dir)                    # questions.jsonl rows, pool (seeded) order
arm_rows = bucket_questions(rows, "both_wrong")   # rows of one bucket, same order; take the first 800
bucket_counts(rows)                           # Counter over BUCKETS
label_questions(bank_dir, cfg)                # rebuild questions.jsonl from the raw shards
# run 3: sort rows by row["question_entropy"] (None = not scored yet)
# a driver's pool row from a bank row: {"example_index", "id", "prompt": row["question"],
#   "reference": row["reference"], ...}
```

Tests: `tests/test_bank.py` (CPU; the FakeLLM of
`tests/test_rollout_eval_merged.py`, workers in-process).

## Selection interface (`apod/selection.py`)

```python
def select_trajectories(arm, trajectories, *, k, num_rollouts, seed):
    """trajectories: merged rows for one refresh (entropy / mean_reverse_kl
    merged in when the rule needs them). Per example_index keep k of
    num_rollouts by the rule (arm = a rule name or a legacy alias):
      all_k          keep everything (alias: all)
      random_k       default_rng(seed + example_index).choice, no replacement
                     (alias: random_top4) -- the no-selection baseline
      entropy_top_k  highest entropy, ties -> lower rollout_index (alias: entropy_top4)
      kl_high/mid/low rank by mean_reverse_kl descending; slice [0,k) / [k,2k) / [2k,3k)
    Truncated rows are eligible. Returns selected.jsonl-shaped rows
    {example_index, rollout_index, entropy|null, mean_reverse_kl|null,
    correct, truncated, response_length}."""
```

`needs_entropy(rule)` / `needs_reverse_kl(rule)` tell the driver which
scoring stage a rule requires.

## Step-based driver (`apod/driver.py`)

The driver of the r1/r2/r3 experiments (ADR 0005, 0006; vocabulary in
CONTEXT.md). The budget is counted in **training steps** (one optimizer
update at `train.effective_batch` = 32 trajectories); a **refresh** every
`refresh_every` steps evaluates the current weights and rolls out the next
block of questions from them. Nothing is counted in rounds; on disk a
refresh's work lives in the stages' existing `round_<r>/` directory
(`apod.paths`), which is refresh r.

```
python -m apod.driver +experiment=r1_correctness_8k            # Hydra app
python -m apod.driver +experiment=r2_trajsel_8k driver.dry_run=true output_dir=/tmp/r2
python -m tests.test_driver                                     # CPU verification
```

Config: `conf/driver.yaml` (package `driver`) plus the experiment files,
which inherit the shared regime `conf/experiment/refresh_8k.yaml` (cap 8192,
100 steps, refresh every 10, all 500 MATH-500 questions + AIME 2025+2026 at
every refresh, peak LR from the kl50w probe, one cosine schedule over
`train.total_training_steps` with a 5-step warmup at step 0, Adam state
persisted, keep-last-2 weights):

| key | meaning |
|---|---|
| `driver.steps_total`, `driver.refresh_every` | 100, 10 -> 10 refreshes x 320 trajectories = 80 questions x `selection.k` |
| `driver.arms.<name>.question_source` | `pool_random` (seeded OpenThoughts sample), `bank_bucket:<bucket>`, `bank_top_entropy`, `bank_random` (bank = `driver.bank_path`, apod/bank.py) |
| `driver.arms.<name>.selection` | a rule of `apod.selection.RULES` |
| `driver.monitor_sets` | named eval sets beside `cfg.eval`; stamped as `eval_sets.<name>` from `conf/eval/<name>.yaml` |
| `driver.kl_estimator` | `scripts/oracle_kl.py --estimator` for the `kl_*` rules (`exact` / `mc`) |
| `driver.noise_band_points` | shaded band width (points) in the plot |
| `driver.dry_run` | CPU stubs instead of the GPU stages (`apod/dry_run.py`) |
| `rollout.num_prompts` | stamped by the driver = questions per refresh (the rollout stage's seed base) |

| experiment | run dir | arms (question source / rule) | rollouts |
|---|---|---|---|
| `r1_correctness_8k` | `r1-correctness-8k` | `teacher_right_student_wrong`, `both_right`, `both_wrong`, `mixed`: `bank_bucket:<arm>` / `all_k` | 4, all trained |
| `r2_trajsel_8k` | `r2-trajsel-8k` | `entropy_top4` (`entropy_top_k`), `kl_high`, `kl_mid`, `kl_low`, `random4` (`random_k`): all `pool_random` | 12, keep 4 |
| `r3_qentropy_8k` | `r3-qentropy-8k` | `uncertain_questions` (`bank_top_entropy`), `random_questions` (`bank_random`): `all_k` | 4, all trained |

Per arm (sequential, in `driver.arms` order), refresh r at step
`r * refresh_every`:

1. `rollout_eval` (sharded over `num_gpus`): MATH-500 eval of the current
   weights + this refresh's rollouts in one engine session; then
   `rollout_eval --eval-only --eval-dataset <name>` per monitor set. Step 0
   evaluates the same base model in every arm, so its eval rows are copied
   from the first finished arm (`reused_from.json`); rollouts are never shared.
2. scoring, only when the rule needs it: `apod.stages.entropy`
   (`entropy_top_k`) or `scripts/oracle_kl.py --student-path <weights>
   --estimator <kl_estimator>` (`kl_*`), sharded.
3. selection in-process -> `selected/selected.jsonl` (exactly 320 rows).
4. `apod.stages.train` under torchrun with `--global-step-offset <step>`
   (steps already trained for the arm, also after a resume; the stage
   continues the run-level schedule from there).
5. prune: weights older than the newest `keep_checkpoints` refreshes and the
   previous refresh's `optimizer_state.pt`.
6. `metrics.jsonl` row (upsert) and `apod.tracking.log_refresh` when the
   module is present (`tracking.init` / `finish` bracket each arm).

Refresh index `refreshes` (step `steps_total`) is the final eval only. Then
`apod.plotting.plot_refresh_curves` renders the plot.

Layout additions over the round-based layout above:

```
pool/questions_<arm>.jsonl      # the arm's questions in the rollout stage's row
                                # schema (prompt = question text, round = refresh
                                # index; bucket / question_entropy / bank_example_index
                                # for bank sources)
pool/prompts.jsonl -> questions_<arm>.jsonl   # symlink, re-pointed per arm (the
                                # stage's fixed path); the driver verifies the
                                # rollouts' question ids against the arm's block
pool/eval_problems_aime2526.jsonl
arms/<arm>/rounds/round_<r>/eval_aime2526/   eval.shard{K}.jsonl, summary.json
arms/<arm>/rounds/round_<r>/oracle/          oracle_kl[_mc].shard{K}.jsonl
dry_run_launches.jsonl          # dry runs: every stage command the driver issued
```

`metrics.jsonl` (driver runs) -- one row per (arm, step):
`{arm, step, refresh, model_path, trajectories_trained, eval: {<set>:
{strict_avg_at_n, strict_pass_at_n, avg_at_n, pass_at_n, cap_hit_rate,
mean_response_length, num_problems, num_samples}}, rollouts: {num_questions,
num_trajectories, cap_hit_rate, strict_accuracy, mean_response_length}|null,
selected: {num_trajectories, mean_entropy, mean_reverse_kl, cap_hit_rate,
strict_accuracy, mean_response_length}|null, train_loss_mean,
train_loss_final, tokens_trained, wall_clock: {rollout_eval_s,
monitor_eval_s, entropy_s, oracle_s, train_s}}`. The eval fields measure
the weights at `step`; the train fields belong to the block trained after
that eval (steps `step .. step + refresh_every`); the final row has null
rollout/train fields.

Resume: every stage leaves a done-marker (`done.shard{K}`, `selected.jsonl`,
`train/done.shard0` + `checkpoint/config.json`, oracle row counts); a
resumed run skips finished stages and continues from any refresh boundary
or mid-refresh state. An existing run dir keeps the `resolved_config.yaml`
it started with (the CLI composition only locates the run dir). `driver.log`
holds the driver's and the children's output.

Dry run (`driver.dry_run=true`): `apod/dry_run.py` replaces each stage
command with a stub writing that stage's files with synthetic, deterministic
values; `tests/test_driver.py` runs the three experiments this way and
checks step accounting, dispatch, sharding, pruning, metrics, the plot,
resume from every boundary, and the pool-symlink guard.

## File ownership (parallel agents — do not cross-edit)

- rollout/eval agent: `apod/stages/rollout_eval.py`
- scoring/selection agent: `apod/stages/entropy.py`, `apod/selection.py`
- driver agent: `apod/main.py`, `apod/plotting.py`; step-based driver
  (2026-09-01): `apod/driver.py`, `apod/dry_run.py`, `apod/selection.py`,
  `conf/driver.yaml`, `conf/experiment/{refresh_8k,r1_*,r2_*,r3_*}.yaml`,
  `tests/test_driver.py`
- question-bank agent: `apod/bank.py`, `conf/bank.yaml`, `tests/test_bank.py`
- train stage (main session): `apod/stages/train.py`, `conf/`
- `pyproject.toml`, `CONTEXT.md`, ADRs, existing `apod/` modules: main
  session only. Reuse existing helpers (`apod.models.generate_vllm`,
  `apod.datasets.io`, `apod.verification`) rather than duplicating them.
```
