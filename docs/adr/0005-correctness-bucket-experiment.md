# 0005 — Correctness-bucket experiment: four question-level arms at cap 8192, strict scoring, 100 training steps each

Date: 2026-09-01. Status: draft from the planning session; the user's decisions,
recorded with the reasoning given at the time.

## Context

kl50 showed trajectory selection matters (kl_mid > kl_high > random > kl_low,
strict MATH-500 avg@4). The supervisor's next question is question-level: does it
matter whether the teacher and the student are *correct* on the questions we
distil on? The pool's joint correctness is skewed — a 9B teacher is essentially
never wrong where the 2B student is right — and at any cap up to 16k most student
rollouts hit the cap.

## Decision

- Four arms, one per correctness bucket: TC/SW, TC/SC, TW/SW, and Mixed
  (everything in no other bucket). No random-question control arm in this run.
- Labels by strict correctness (no `\boxed` = wrong, cap-hit included) from 4
  rollouts per model: correct ≥ 3/4, wrong ≤ 1/4, at **cap 8192** — the cap the
  arms train and evaluate at. Labels, training and eval share one regime.
- Budget: 100 training steps per arm at effective batch 32 = 3,200 trajectories
  = 800 questions × 4 rollouts, all 4 trained (no trajectory selection), so the
  arms differ only in question correctness.
- Every 10 training steps (a *refresh*): eval the current weights on MATH-500
  avg@4 and AIME 2025+2026 avg@16 at 8192, strict; re-roll the next 80 questions
  from those weights. Per-step logs on the training batch: loss, top-16 overlap
  ratio, overlap-token advantage, mean |H_S − H_T|.
- Question bank built once: student 4 rollouts over ~14,500 questions, teacher 4
  rollouts only where its label decides a bucket (all student-correct questions,
  student-wrong questions until TC/SW, TW/SW and the (teacher-mixed, student-wrong)
  part of Mixed fill). Bank stored with per-rollout grade, truncation, length.

## Alternatives considered

- **Cap 16384 for labels** (teacher finishes 72% instead of 37%; TC/SC is 16%
  instead of 6% of questions). Rejected: labels would not describe the regime
  the arms train in; truncation is part of what OPD has to cope with, and strict
  scoring already treats it as wrong everywhere else.
- **Cap 32768.** Rejected: ~5 GPU-days and an untested trainer envelope.
- **Random-question control arm.** Deferred; the run answers how the buckets
  differ from each other, not whether any beats unselected OPD. Six more hours if
  wanted later.
- **Mixed = student 2-of-4 only.** Rejected: 3% of questions at 8k, needs the
  whole pool; the "none of the three" definition (9%) is fillable and also
  absorbs the empty teacher-wrong/student-correct cell.

## Consequences

- At 8192, most teacher-wrong is teacher cap-hit: TW/SW tests whether an
  unfinished teacher trace still transfers, and the slides must say so.
- TC/SC needs ~13,000 student-swept questions; the teacher sweep is narrow, so
  the bank costs ~17 GPU-hours, reusable for later question-selection work.
- Per-step overlap/entropy diagnostics require a chunked lm_head pass beside the
  Liger fused loss in the train stage (new code, after the live LR sweep ends).
- Changing the cap, the 3-of-4 rule or the eval protocol mid-run invalidates all
  four curves.
