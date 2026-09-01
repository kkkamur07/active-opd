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
- question-bank agent: `apod/bank.py`, `conf/bank.yaml`, `tests/test_bank.py`
- train stage (main session): `apod/stages/train.py`, `conf/`
- `pyproject.toml`, `CONTEXT.md`, ADRs, existing `apod/` modules: main
  session only. Reuse existing helpers (`apod.models.generate_vllm`,
  `apod.datasets.io`, `apod.verification`) rather than duplicating them.
```
