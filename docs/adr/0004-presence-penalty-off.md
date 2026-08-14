# ADR 0004: Presence penalty off (0.0) for rollouts and eval

Date: 2026-08-14. Status: accepted (user decision, relayed with the reasoning
below). Supersedes the presence_penalty: 1.5 setting in ADR 0002's sampling
block.

## Context

Rollouts were generated with `presence_penalty: 1.5` (via the token-identical
incremental logits processor) to suppress the base 2B model's degenerate
repetition loops. But entropy scoring (`apod/stages/entropy.py`) and the GKD
reverse-KL objective both use the model's RAW distribution — no penalty. Two
consequences:

1. **Length→entropy coupling.** The penalty subtracts 1.5 from every
   already-emitted token's logit, so the generation distribution drifts
   further from the natural one the longer the trajectory runs. Scored
   unpenalized, longer trajectories look systematically more surprising →
   higher measured entropy → entropy_top4 silently becomes a
   longest-trajectory arm. Invisible at the 1024 smoke cap (all lengths
   equal), active at 8192.
2. **Off-policy gap.** On-policy distillation's premise is training on
   samples from the student's own distribution. Penalized sampling is a
   different distribution, and the reverse KL is then computed between the
   unpenalized student and the teacher at states the unpenalized student
   would not have produced with the same probability.

## Decision

`presence_penalty: 0.0`, `fast_presence_penalty: false`. Generation, entropy
scoring, and the training objective now all use the same distribution: the
method is genuinely on-policy and the selection signal is uncoupled from
length by construction.

## Consequences

- Degenerate repetition is no longer suppressed. This was load-bearing for
  the base model (61% truncation without penalty in early probing). The
  smoke report measures repeated-n-gram rate, cap-hit rate, and response
  lengths with the penalty off; if repetition returns at damaging levels the
  penalty comes back as a *documented tradeoff* rather than a default.
- The incremental processor (`apod/models/presence_penalty.py`, verified
  token-identical to vLLM's native path) stays in the tree, dormant, with
  its config knobs; re-enabling is a two-line config change. The rollout
  stage's fast-path guard now fires only for a nonzero penalty.
- Without a logits processor vLLM may use its V2 model runner (the processor
  forced V1).
