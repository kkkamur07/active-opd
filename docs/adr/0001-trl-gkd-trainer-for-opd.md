# 0001 — TRL GKDTrainer for OPD training, not verl

Date: 2026-08-14. Status: accepted.

## Context

The training phase needs on-policy distillation of Qwen3.5-2B against a frozen
Qwen3.5-9B teacher with a reverse-KL objective, plus a nonstandard front end:
external vLLM rollouts, entropy scoring, and trajectory-level top-k selection
before any gradient step. Candidate frameworks were verl and TRL. verl was in the
repo previously and was removed in commit 686a593.

## Decision

TRL 1.10.0, using `GKDTrainer` configured for reverse KL. Rollouts stay outside
the trainer (vLLM 0.26.0 with the incremental presence-penalty processor);
selected trajectories are fed to the trainer as data, so the trainer's own
on-policy sampling fraction is not used for selection-arm runs.

## Reasons

- Single-machine scale (A100s, 2B student, frozen 9B teacher) does not justify
  verl's Ray/hybrid-engine machinery.
- The selection step is custom in either framework; TRL's GKD loop is the closer
  starting point to standard OPD, so less is patched.
- The dependency stack (torch 2.11.0+cu130, vLLM 0.26.0, driver R595) was resolved
  around TRL 1.10.0's `vllm<=0.26.0` ceiling (`[docs/guide.md](../guide.md#dependency-choices)`). Switching
  frameworks reopens all of it, including re-validating
  `apod/models/presence_penalty.py` against a different vLLM.

## Consequences

- vLLM is capped at 0.26.0 until TRL raises its ceiling.
- The exact `beta`/`lmbda` semantics of `GKDTrainer` must be verified against the
  installed 1.10.0 source before training code is written (TRL is not installed by
  the default extra).
- If the loop later becomes heavily iterative with frequent student-weight refresh
  of the rollout engine, the engine/trainer weight-sync cost is ours to manage;
  that is the scenario where verl would have earned its complexity.
