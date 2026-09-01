# 0006 — AIME 2025+2026 monitor: pooled n=60, avg@16, at the run cap

Date: 2026-09-01. Status: accepted.

## Context

MATH-500 avg@4 (ADR 0003) has been the only eval. A competition-level
number is wanted next to it, for every arm at every refresh (the eval +
re-rollout every 10 training steps), to put beside the OPD papers. The survey
(`docs/eval_benchmarks.md`) picked AIME 2025 (`MathArena/aime_2025`, standard
in every 2025-26 OPD paper) and AIME 2026 (`MathArena/aime_2026`, held after
the Qwen3.5 release, the clean number). Both are 30 integer-answer questions.

Two trade-offs had to be made:

- **avg@16, not avg@64.** Qwen evaluates AIME at avg@64; Rethinking OPD at
  avg@16; SEAD/DAPO at avg@32. The naive SE of avg@k shrinks with k, but the
  between-question term (`Var_i(p_i)/n`, ~6 pts at n=30) does not; past k~16
  extra samples buy nothing against it, only more questions do. Hence pool
  the two years (n=60, 960 trials per refresh) and stop at 16.
- **The run cap (8192), not 16k+.** Papers use 20-82k for AIME, and at 8k
  the student's AIME score is dominated by truncation. Running at the run
  cap anyway keeps the number on the same footing as the MATH-500 monitor
  (one engine, one cap, one strict rule, every refresh) and costs ~960
  generations per arm per refresh (~7 min on two A100s). A headline table at
  16,384 -- comparable to SEAD / Revisiting OPD, never to model cards -- is
  `scripts/terminal_eval.py --max-new-tokens 16384` on the final checkpoints.

## Decision

- Pooled AIME 2025+2026 (`aime2526`, 60 questions, ids keep the year prefix)
  at avg@16 / pass@16, evaluated in the same `rollout_eval` engine session
  as MATH-500 and the refresh's rollouts (`--eval-dataset math500 aime2526`;
  its own `eval_aime2526/` files and done-markers, never a second launch:
  design review D1, ~16 min saved per arm) at every refresh, at the run cap,
  same sampling as rollouts, strict
  scoring (no `\boxed` = incorrect, cap-hit included), Math-Verify grading
  (handles `\boxed{070}` vs `70`; no extra normalisation).
- Per-question seeds follow the MATH-500 rule
  (`seed + eval_seed_offset + refresh_index * num_problems + problem_index`).
- Every accuracy is reported with its cap-hit rate, the naive SE, and a
  question-level cluster bootstrap 95% interval (resample questions, keep
  all k samples of a question together); per-year splits under the pooled
  row.
- `scripts/terminal_eval.py` evaluates a chosen checkpoint under this
  protocol by default and takes `--max-new-tokens` / `--num-samples`
  overrides for a headline table. There is exactly one generation and
  grading path; the MATH-500 `eval/` layout is untouched.

## Consequences

- Never compare these AIME numbers to model-card numbers (38-82k caps).
  Changing the cap or k creates a new, incomparable series; the cap is in
  the derived dir's name for that reason.
- A 30-question set cannot carry an arm-vs-arm claim on its own: quote the
  bootstrap interval, and let MATH-500 and the pooled 60-question AIME carry
  comparisons. A single-seed 5-point AIME delta between arms is noise
  (`docs/eval_benchmarks.md` section 6).
- The driver has to materialize `pool/eval_problems_aime2526.jsonl` once per
  run and name the set on every `rollout_eval` launch; the stage refuses to
  load the set from the Hub itself so problem_index -> question can never
  drift between refreshes.
