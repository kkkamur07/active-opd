# 0003 — MATH-500 eval: avg@4 primary, pass@4 as diversity monitor

Date: 2026-08-14. Status: accepted.

## Context

The experiment's deliverable is a curve per arm (trajectories used vs accuracy),
so eval noise must sit below the per-round deltas between arms. MATH-500 has 500
problems; total Bernoulli trials = problems x samples controls the noise floor.
Qwen's own protocol is avg@64 on AIME (30 problems, 1,920 trials, ~±1.1%);
avg@4 on MATH-500 gives 2,000 trials — the same precision at 1/16th the depth.
Thinking-mode Qwen models are evaluated with sampling, never greedy.

## Decision

Every round, per arm, on `HuggingFaceH4/MATH-500`, graded by math-verify:

- **4 samples per problem**, sampling settings identical to training rollouts
  (`temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5`, 8192-token
  cap), seeds fixed per (round, problem, sample). Cap-hit samples grade incorrect.
- **avg@4** (mean accuracy over the 4 samples) is the primary metric — the
  unbiased, low-noise estimator of pass@1. No separate single-sample "pass@1"
  column; it is the same quantity with more noise.
- **pass@4** (fraction of problems with ≥1 correct sample) is computed from the
  same samples at zero extra cost, as the diversity/mode-collapse monitor:
  pass@4 falling while avg@4 rises is the reverse-KL collapse signature.
- **Round-0 anchor**: the untrained student is evaluated once; all arms share
  that point.
- Eval runs in the same vLLM engine session as the round's rollouts (same
  weights, one engine bring-up per round).

## Consequences

- Eval is the dominant per-round cost (~8M generated tokens vs ~4M for
  rollouts). If wall-clock binds, reduce rounds — never samples; a plot with
  noise above signal is a failed experiment.
- Absolute scores carry the math-verify grader ceiling (~97–98%); arm
  comparisons are unaffected. Changing any eval setting mid-experiment
  invalidates all curves.
