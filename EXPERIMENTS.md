# Experiments: complete record

Every training and evaluation run of this project, what it tested, its configuration and where its data lives.
Generated 2026-11-19 from the run directories (`resolved_config.yaml`, `metrics.jsonl`) of the full results copy
(see "Where the data lives"). All accuracies are MATH-500 strict avg@4 (%) at the run's token cap unless stated.

Student Qwen3.5-2B, teacher Qwen3.5-9B, training data OpenThoughts-derived math (`siyanzhao/Openthoughts_math_30k_opsd`)
for every run except the 4B runs in Section 4.

---

## 1. Final program (reported in the paper, `papers/apod-acl/main.tex`)

Shared setup: 60 training steps, 4 refreshes of 15 steps, 480 answers trained per refresh, LR 3.1623e-6 (3 warmup steps,
cosine to 0.1x), effective batch 32, bf16, 8,192-token cap, MATH-500 (500 x 4) after every refresh (steps 0/15/30/45/60).
Configs: `conf/experiment/p2b_*.yaml`. Seeds: 42 / 1042 / 2042 with disjoint questions (question offset 0 / 1440 / 2880;
correctness seeds 2-3 exclude earlier seeds' questions). Hardware: 2x A100 80GB (seed 3 of most groups on a second VM).

| Run directories | Rules (arms) | Level | Status |
|---|---|---|---|
| `p2b-rollout-s{1,2,3}` | `random` (standard OPD), `kl_high`, `kl_mid`, `kl_low`, `ent_top4` | answer (120 q x 12, keep 4/q) | complete, 3 seeds |
| `p2b-qkl-s{1,2,3}` | `qkl_high`, `qkl_mid`, `qkl_low` | question (360 q x 4, keep 120 q), reverse KL | complete, 3 seeds |
| `p2b-qent1-s{1,2,3}` | `qent1_high` | question, student entropy of ONE answer | complete, 3 seeds |
| `p2b-qent-s{1,2,3}` | `qent_high`, `qent_mid`, `qent_low` | question, mean entropy of 4 answers | complete, 3 seeds; NOT in the paper (author decision: report the single-answer rule only) |
| `p2b-correct-s{1,2,3}` | `t_right_s_wrong`, `both_right`, `mixed`, `both_wrong` | question, correctness bucket (120 q x 4) | complete, 3 seeds |
| `p2b-{rollout,qkl,correct}-s1/terminal_eval` | held-out evaluation (otval200: 200 unseen OpenThoughts questions x 4) of every seed-1 checkpoint | | complete, seed 1 only; not in the paper |

Main results (mean over steps 15-60, mean +- SD over 3 seeds; source `experiments/p2b-analysis/three_seeds_meansd.txt`,
pass@4 `pass4_step60_meansd.txt`):

| Rule | avg@4 | Delta avg@4 vs OPD | Delta pass@4 vs OPD |
|---|---|---|---|
| Standard OPD (`random`) | 42.58 +- 0.37 | | |
| T right, S wrong | 43.80 +- 0.61 | +1.22 +- 0.31 | +1.77 +- 0.18 |
| Reverse KL high (question) | 43.67 +- 0.55 | +1.09 +- 0.84 | +1.68 +- 0.55 |
| Entropy high, 1 answer (question) | 43.65 +- 0.30 | +1.08 +- 0.26 | +1.38 +- 0.40 |
| Both right | 43.49 +- 0.28 | +0.91 +- 0.16 | +1.20 +- 0.25 |
| Mixed | 42.85 +- 0.40 | +0.28 +- 0.18 | +1.18 +- 0.50 |
| Reverse KL middle (question) | 42.33 +- 0.24 | -0.25 +- 0.13 | +0.55 +- 0.39 |
| Reverse KL low (question) | 41.52 +- 0.92 | -1.06 +- 1.10 | -0.57 +- 0.91 |
| Both wrong | 41.32 +- 0.63 | -1.26 +- 0.90 | -0.90 +- 0.52 |
| Reverse KL high (answer) | 42.31 +- 0.46 | -0.27 +- 0.10 | +0.25 +- 0.48 |
| Reverse KL middle (answer) | 42.57 +- 0.31 | -0.01 +- 0.47 | +0.48 +- 0.19 |
| Reverse KL low (answer) | 42.45 +- 0.77 | -0.13 +- 0.49 | +0.43 +- 0.64 |
| Entropy high (answer) | 42.45 +- 0.17 | -0.13 +- 0.20 | +0.25 +- 0.40 |

Supporting analyses (all in `experiments/p2b-analysis/`): between-question variance share (`variance_all.txt`),
single-answer entropy vs reverse-KL question overlap (`qent1_overlap.txt`), final-checkpoint results
(`pass4_step60_meansd.txt`), training-pool composition (`pool_composition.txt`), seed-1 held-out results (`otval_s1.txt`),
per-step curves (`curves_3seeds.json`).

### Shared infrastructure for the final program

| Directory | What |
|---|---|
| `bank-8k` | correctness labels: 14,500 sampled questions with 4 student answers; 4,641 also with 4 teacher answers (the training pool of every rule). Buckets: both wrong 2,387, T right S wrong 971, both right 866, mixed 417 |
| `bank-8k-w` | widened bank (25,000 questions) used by the correctness rules in seeds 2 and 3 |
| `teacher_eval` | Qwen3.5-9B on MATH-500 at 8k |
| `kl50w`, `kl50w_lrsweep` | learning-rate selection: per-rule search (random, kl_high/mid/low; all best at 5e-6 on 8-step training loss) and refinement on kl_mid -> 3.1623e-6 (`kl50w/lr_sweep.db`, `kl50w/lr_refine.json`) |

---

## 2. 2B pilot runs (before the final program; not in the paper)

These shaped the hypotheses. Their designs differ from the final program (constant LR 2e-5 with Adam reset per round, or
100-step runs with 10-step refreshes), and most have one seed.

| Directory | Rules | Regime | Notes / key result (source) |
|---|---|---|---|
| `apod` | `entropy_top4`, `random_top4`, `all` | 8 rounds, LR 2e-5, 8k | first run: answer-level entropy vs random vs all answers |
| `oracle8k` | `kl_high/mid/low`, `random` | 1 round, LR 2e-5, 8k | first exact-reverse-KL answer selection at 8k |
| `oracle16k`, `oracle16k_seed2` | same + `all` (seed A only) | 3 rounds, LR 2e-5, 16k | middle band led after round 1 by +3.75 / +3.70 (`experiments/apod-data-scan/results_tables.md`) |
| `oracle16k_lr2e5`, `oracle16k_lr5e6` | `kl_mid` | 16k, LR probes | learning-rate probes for the 16k regime |
| `oracle16k_teacher`, `kl50_teacher` | teacher only | 16k / 8k | teacher accuracy (8k: 61.15) |
| `kl50` (+ `kl50_warmup_backup`) | `kl_high/mid/low`, `random` | 2-3 rounds, LR 2e-5, 8k | middle band 47.45 vs random 38.80 at step 34, one seed (`results_tables.md`); did not hold over all ten checkpoints (mean +1.20) |
| `reeval32k-kl50` | kl50 checkpoints | eval only, 32k cap | re-evaluation at 32k; stopped before writing results |
| `r1-correctness-8k` | correctness buckets | 100 steps, LR 3.16e-6, fixed 800 questions | step 100: T right S wrong 45.50, both right 44.30, mixed 43.30, both wrong 41.00 (`r1-correctness-8k/metrics.jsonl`) |
| `r2-trajsel-8k` | `kl_high/mid/low`, `entropy_top4`, `random4` | 100 steps, LR 3.16e-6 | step 100: kl_mid 43.75, kl_high 43.55, kl_low 43.30, random4 43.10, entropy_top4 42.40 (`metrics.jsonl`) |
| `r3-qentropy-8k` | `uncertain_questions`, `random_questions` | 100 steps, 800 most uncertain bank questions | step 100: 42.25 vs 44.10 (`metrics.jsonl`) |
| `r4-qentropy-low-8k` | `certain_questions` | 100 steps, 800 least uncertain | step 100: 42.45 (`metrics.jsonl`) |

---

## 3. Abandoned 4B program (not in the paper)

The program was first designed for a Qwen3.5-4B student and switched to the 2B because the 4B left too little room below
the teacher at 8k (57.15 vs 61.15).

| Directory | What | Result |
|---|---|---|
| `p0-4b-base8k`, `p0-4b-base16k` | 4B base evaluation | 8k: 57.15; 16k: 74.25 (HANDOFF.md §5) |
| `spec-ab-4b` | MTP speculative decoding A/B | 2,367.6 vs 3,054.5 generated tokens/s: speculative decoding slower, not used (`mtp.json`, `plain.json`) |
| `train-probe-4b` | micro-batch memory/speed probe | micro 2 chosen (`probe.json`) |
| `bank-8k-4b` | correctness relabel with the 4B student | too few T-right-S-wrong questions (187 of 2,837) |
| `p4b-lr` | 4B LR sweep (2e-6, 5e-6, 1e-5, 2e-5) on held-out questions | step-0 eval used a 16k cap, later steps 8k; not comparable |
| `p4b-rollout-s1` | 4B answer-level program | stopped after the step-0 evaluation (57.15) when the program moved to the 2B |

---

## 4. Not run

AIME 2025/2026 on the saved checkpoints; held-out evaluation for seeds 2-3; a standard-OPD run with both-wrong questions
removed; the 4-answer entropy rule is run but unreported.

## Where the data lives

- Full results (metrics, evaluations, scores, rollouts; no checkpoints): local copy `~/Desktop/krrish/apod-results-20261118/runs`
  on the lab machine (20 GB).
- Everything including checkpoints (650 GB): GCP project `krrish-active-learning`, boot disk of `instance-20260813-175138`
  (us-central1-a) and snapshot `apod-results-20261118`.
- Per-step training logs: W&B project `apod`.
- Analysis scripts and derived tables: `experiments/p2b-analysis/`; job scripts: `experiments/p2b-jobs/`.
