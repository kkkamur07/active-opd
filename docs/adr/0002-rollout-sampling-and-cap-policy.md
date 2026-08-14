# 0002 — Rollout sampling keeps presence_penalty=1.5 and the 8192 cap; monitor, don't gate

Date: 2026-08-14. Status: accepted.

## Context

Reverse-KL OPD is defined as an expectation over trajectories sampled from the
student π_S. Reference implementations (Thinking Machines' Tinker cookbook OPD,
TRL GKD) sample with plain temperature 1.0 and no penalties. Qwen3.5-2B's model
card, however, warns it is unusually prone to thinking loops, and this repo
measured the throughput collapse those loops cause. "Revisiting On-Policy
Distillation" (arXiv 2603.25562) shows repetition drift is exactly where
teacher supervision becomes unreliable, and its remedy is restricting rollout
sampling to teacher-reliable regions.

## Decision

All arms roll out with the model card's recommended settings:
`temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` (via the
incremental-mask processor in `apod/models/presence_penalty.py`), capped at
8192 new tokens.

- Cap-hit trajectories are **kept for training** (every token has a valid
  teacher target; the Tinker reference trains on partial rollouts explicitly)
  and **graded incorrect for reporting**.
- Selection is pure entropy or random — correctness never gates training data.
- Per round, per arm, we log: cap-hit rate, a repetition metric, and mean
  trajectory entropy. These are monitors, not gates. Entropy collapse across
  rounds (arXiv 2603.07079) would degrade the selection signal and must be
  visible if it happens.

## Consequences

- Trajectories are drawn from a penalized π̃_S, not π_S, so "on-policy" is
  approximate and the KL-gradient estimator is biased. The bias is identical
  across arms and cancels out of the arm ranking, which is the quantity the
  experiment measures. Record the sampling settings in the run manifest.
- The entropy arm has a known failure mode — preferentially selecting long,
  degenerate, or cap-hit traces. The monitors above are how we detect it; no
  automatic gating masks it.
- Changing any sampling setting mid-experiment invalidates all curves; a change
  means restarting every arm from round 0.
