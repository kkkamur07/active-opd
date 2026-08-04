# Active OPD

A research prototype for active on-policy distillation on mathematical reasoning.

In standard on-policy distillation the student generates rollouts, the teacher
scores the states the student actually visited, and the student is trained to
match the teacher on those states. This repository adds a selection step: not
every student rollout is equally worth training on, so a verifier decides which
ones enter the training pool.

The research question is whether selection helps once every arm gets the same
training budget. `scripts/experiments/run_filter_ablation.py` is built to answer
that and nothing else.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

`math-verify` is a hard dependency. Without it the verifier falls back to a
conservative string comparator that reports `unverified` rather than guessing,
which shrinks the training pool instead of filling it with mislabelled rollouts.

## Running things

```bash
# Print the resolved Hydra config without loading any model or dataset.
python -m scripts.main

# End-to-end check on real models. Verifies that parameters actually move.
python -m scripts.diagnostics.smoke_test --run --max-new-tokens 256

# Measure the trace-length distribution of a corpus before picking a budget.
python -m scripts.data.profile_traces --limit 256

# The experiment.
python -m scripts.experiments.run_filter_ablation \
    --train-prompts 64 --rounds 20 --k 4 --select-budget 16 --eval-problems 200
```

Add `--dry-run` to the ablation to print the budget plan without touching a GPU.

## Method

Each round:

1. The student generates K rollouts per problem with the current weights.
2. A verifier grades each final answer against ground truth.
3. A selector keeps some subset.
4. The student is trained on the selected rollouts with the OPD loss.

The verification gate is kept separate from the loss. Answer matching decides
what enters the pool; it is never a reward or a differentiable target.

## Design notes

### The loss is the exact reverse KL, computed in chunks

The objective is the per-token reverse KL on student-visited states:

```
kl_t = sum_v pi_theta(v) ( log pi_theta(v) - log pi_T(v) )
```

This is computed exactly over the vocabulary rather than estimated from the
sampled token. The reason is that a sampled estimator like veRL's `k3` is
unbiased for the KL *value* but its pathwise derivative is not the derivative of
that value. With `r = log pi_theta(y) - log pi_T(y)` and the teacher detached,
`d k3/d log pi_theta(y) = 1 - exp(-r)`, so

```
E_{y~p}[ (1 - q/p) grad log p ] = sum_v (p_v - q_v) grad log p_v = grad KL(q || p)
```

which is the forward KL, the opposite direction. `k1` is worse: its expected
gradient is exactly zero. Both remain available through `sampled_diagnostics`
for logging and for comparison against veRL, and `OPDLossConfig` refuses to
train on either.

At a real thinking-trace length the naive computation does not fit. An
18k-token sequence costs about 5.1 GiB of bf16 logits per model and 10.2 GiB per
fp32 `log_softmax`, so over 30 GiB before weights or activations. The loss is
therefore chunked along the time axis, with each chunk's softmax recomputed
during backward. Measured on an H100: 13.9 GiB drops to 5.8 GiB at 4k tokens,
and 16k fits in 23 GiB.

### Parameters are kept in fp32

bf16 has an 8-bit mantissa. One ulp at a typical weight magnitude is around
8e-5, while an Adam step at `lr=1e-5` moves about 1e-5, so the update lands
below half an ulp and rounds away. Writing updates directly into bf16 weights
leaves roughly 85 to 90 percent of parameters bitwise unchanged while the loss
still decreases and the step counter still climbs.

Compute stays in bf16 under autocast; only the master copy is fp32. Set
`assert_params_move_after` to have training fail loudly if the weights are not
moving.

### Truncation is a separate outcome

The verifier emits six outcomes: `correct`, `wrong`, `malformed`, `truncated`,
`unverified`, and `skipped`. Only `correct` and `wrong` can ever be selected for
training.

`truncated` matters because a rollout that used its whole generation budget
without emitting EOS did not finish reasoning, and grading it produces a length
statistic rather than a reasoning verdict. The flag comes from the generator
rather than from inspecting the decoded text, since a trace can close `</think>`
and still be cut off before writing its answer. Answer extraction reads only the
region after `</think>`, so a mid-derivation "so the answer is ..." cannot be
scraped as a final answer.

`unverified` exists because `wrong` is the Active OPD retention rule. An
undecidable comparison that defaults to `wrong` does not merely get dropped, it
gets promoted into the training set as a student mistake.

### Budgets are matched, and the checks are enforced

A learning-efficiency claim is only readable if the arms consumed the same
resources. The ablation matches every arm to the same number of trained
rollouts per round and then validates three things before reporting a result:
equal optimizer steps, equal rollouts generated, and response-token counts
within 10 percent of each other. Failing any of them exits non-zero.

All arms generate identical rollouts, so selection saves training compute rather
than sampling compute. Results are written against four axes: optimizer steps,
response tokens backpropagated, rollouts generated, and wall clock.

`pass@1` is the headline metric, reported with Wilson intervals. `pass@k` is
reported separately. Arms are compared on shared problems with McNemar's test,
which is far more sensitive than comparing two independent proportions.

### Selection returns weights

Every selector returns per-rollout weights next to the selected records, and
`compute_opd_loss` takes a `weights` argument. A hard filter is the case where
weights are 0 or 1. This is the seam an acquisition score plugs into later
without a rewrite, which keeps a future filter-versus-score comparison
controlled: the only thing that differs is the weighting function.

## Layout

```
aopd/
  data/        answer extraction, verification, rollout records, dataset adapters
  losses/      OPD loss estimators and response masking
  models/      lazy student and teacher wrappers
  train/       rollout collection, selection, batching, the trainer
  evaluation/  held-out scoring and efficiency metrics
configs/       Hydra config groups
scripts/
  main.py        config inspection
  experiments/   the filter ablation
  diagnostics/   real-model smoke test
  data/          corpus profiling and trace generation
tests/
```

## Configuration

Hydra composes the run from `configs/config.yaml`. The groups that matter most:

`precision/h100` sets the micro-batch caps, the loss chunk size, and fp32 master
weights. `generation/qwen3_thinking` sets an 18k-token budget, which the trace
profiler suggests is roughly the p95 of a Qwen3 math trace. `filtering/` selects
the arm. `estimator/exact_reverse_kl` selects the loss.

Config parsing raises on unknown keys. A misspelled option fails at startup
rather than being silently dropped and leaving a default in charge.

Each run writes to its own timestamped directory under `outputs/`.

## Data

Training corpora need a real ground-truth answer column, and
`example_from_record` raises when one is missing rather than parsing an answer
out of a reasoning trace. `MATH_TRAIN_DATASETS` lists the presets.

OpenThoughts-114k is deliberately not among them. Its default split exposes only
`system` and `conversations`, has no answer column, and is ordered by domain, so
the first several thousand records are competitive programming rather than
mathematics.

## Tests

```bash
pytest -q
```

The suite pins the properties that are easy to break without noticing: that the
expected gradient of the loss matches the analytic reverse KL, that the response
mask is correct under left padding, that equivalent answers in different LaTeX
forms are not labelled wrong, that truncated rollouts never become training
signal, and that unmatched budgets fail the fairness checks.

## Status

`docs/idea.md` describes the intended method, `docs/code-review.md` records a
review of the codebase, and `todo.md` tracks what is not built yet. The
acquisition score is not implemented: the current arms are hard filters, and the
first question to answer is whether selection beats a budget-matched random
control at all.
