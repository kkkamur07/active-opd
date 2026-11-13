# Pre-registered design (revised after the AC critique `outputs/apod-critique-2026-11-12.md` §3.3, before any new seed ran): does the aligned band beat uniform-k? (written 2026-11-13, before any new data)

## Design goal

One question decides the paper: **at equal optimizer steps, does training on the middle third of a question's rollouts by reverse KL (aligned band) beat a random four of twelve (uniform-k), and does that hold for new questions and new seeds?**

Everything in the current draft rests on one MidAlign-8k seed (+8.65 points at step 34) and on round 1 of two MidAlign-16k seeds that share their questions. The hypothesis was formed after MidAlign-16k seed A. This experiment is the first test designed after the hypothesis, with the analysis fixed in advance.

A second, cheaper question runs first because it needs no training: **is the step-34 gain solving or finishing?** We re-evaluate the surviving MidAlign-8k step-34 checkpoints at a 32k-token cap with the same per-sample seeds.

## Experiment A: 32k re-evaluation (running since 2026-11-13 00:00 UTC)

| item | value |
|---|---|
| models | MidAlign-8k (kl50) step-34 checkpoints: aligned (kl_mid), uniform-k (random), then base, divergent, redundant |
| eval | MATH-500, 500 × 4 samples, cap 32,768, same eval seeds as the original step-34 eval (round index 2; base at round 0) |
| run dir | `outputs/runs/reeval32k-kl50` on the VM |
| readout | strict accuracy, finish rate, acc given finished, at 8k-equivalent truncation and at 32k |
| validity check | samples that finished under 8k in the original eval should reproduce token-for-token (same seed); report the match rate |

Interpretation, fixed now: if the aligned − uniform-k gap at 32k is ≥ half the 8k gap (≥ 4.3 points), the step-34 gain is mostly solving. If it is < 2 points, the gain was mostly finishing within budget and the paper says so.

## Experiment B: MidAlign-8k replicate seeds

| item | value |
|---|---|
| regime | identical to MidAlign-8k: config copied from `kl50/resolved_config.yaml` (cap 8,192, constant LR 2e-5, Adam reset per round, effective batch 32, micro 8 × 2, 136 questions/round, 17 steps/round, 3 rounds) |
| what changes | `seed` = rollout and eval sampling seed, training seed, and question draw (`pool_seed`) |
| questions | fresh per seed; exclude kl50's pool, earlier replicate seeds, and any row overlapping MATH-500 (exact normalized match or shared 13-gram; filter flags 115/29,427 rows) |
| arms | aligned band (kl_mid), uniform-k (random) |
| evals | MATH-500 500 × 4 at 8k after steps 17, 34, 51 |
| checkpoints kept | steps 34 and 51 for every arm (for a later 32k re-evaluation); the chain pauses if free disk < 25 GB |
| logged | per-step `bf16_rounded_frac`, overlap and entropy diagnostics (on by default, `diag_every: 1`); never logged at 2e-5 before |
| seeds | 101–108 (n = 8 planned), run one after another |
| code | `scripts/bucket_experiment.py --midalign-seed S` |

Known difference from kl50: the fused flash-linear-attention kernels were installed after kl50 ran, so the new seeds are not bit-comparable with kl50. For this reason the **primary analysis uses only the new seeds**; kl50 is reported alongside, not pooled.

### Primary endpoint and test

- c(s) = mean over steps 17, 34 and 51 of [strict avg@4(aligned) − strict avg@4(uniform-k)] on MATH-500 at cap 8,192, seed s, in points. Pooled over steps because the contrast swings between checkpoints and no single step was justified in advance (step 34 was picked after seeing the test set).
- Test: one-sided one-sample t-test of mean c(s) > 0 over the 8 new seeds, α = 0.05. Robustness: exact sign-flip permutation test (2⁸ = 256 configurations).
- Reported with: every seed's c(s), the mean with a t-based 95% CI, and problem-level bootstrap CIs as supplementary only.

### Decision rules (fixed now)

- **Supported:** both the t-test and the permutation test give p < 0.05.
- **Headline size refuted:** the one-sided 95% upper bound on mean c(s) is below +3.7 points; the paper may not claim an effect of that size.
- **Practically null:** the 90% CI lies inside ±2 points.
- Every seed is reported, whatever the outcome.

### Secondary endpoints

1. Each step separately (Holm-corrected over 3 tests); step 34 alone is the old headline.
2. The termination decomposition (finish rate, acc given finished, acc given truncated) of Δ₃₄.
3. Training-set reverse KL and truncation of the selected rollouts per round.

### Power (from existing data, an estimate)

The between-seed SD of a contrast was 3.40 points in MidAlign-16k, where the seeds shared their questions. Fresh questions can only add variance, so this is optimistic. With σ = 3.4, one-sided α = 0.05 and 80% power:

| seeds | smallest detectable mean Δ₃₄ |
|---|---|
| 4 | 5.6 points |
| 6 | 4.0 points |
| 8 | 3.3 points |

(Exact noncentral-t power, computed 2026-11-13 with scipy.) The kl50 result was +8.65, but the mean over all ten earlier contrasts was +1.2. No affordable design confirms an effect near +1.2; if the truth is there, the honest result is "an effect below about 3 points cannot be ruled out". Pooling over three steps may lower the SD of c(s), which would help.

### Stopping rule

- Seeds run one after another. No stopping for efficacy.
- Non-binding futility stop after 4 seeds if mean c(s) ≤ 0 (stopping for futility cannot inflate the false-positive rate).
- No other look changes the plan.

### Cost (estimate from `docs/perf_review.md` round-arm timings)

| part | minutes |
|---|---|
| shared round 0 (rollouts 13.6 + KL scoring 27.5, plus base eval 15.2) | ≈ 56 |
| aligned arm, 3 rounds (3 × train 25.4 + 3 × eval 15.2 + 2 × (rollouts 13.6 + scoring 27.5)) | ≈ 204 |
| uniform-k arm (same, no scoring) | ≈ 149 |
| **per seed** | **≈ 6.8 h** |
| **8 seeds** | **≈ 55 h wall-clock on the 2×A100 VM (≈ 110 GPU-hours)**; interim after 4 at ≈ 27 h |

## What each outcome means for the paper

| outcome | paper |
|---|---|
| mean Δ₃₄ > 0, significant, and 32k gap holds | the main claim stands as a replicated result |
| mean Δ₃₄ > 0, significant, but 32k gap collapses | the claim becomes "aligned-band selection teaches the student to finish within budget faster" |
| not significant, mean > 0 | effect smaller than about 4 points or absent; report as an honest null with the estimate |
| mean ≤ 0 | the kl50 result was seed noise; reframe around the entropy/termination analysis |

## Not in this experiment (next, in order)

1. Divergent and redundant arms on 2 seeds (tests the inverted U).
2. Entropy-middle arm (the teacher-free version).
3. fp32 master weights for the 100-step regime (bf16 rounding discards ~97% of updates at 3.16e-6, `docs/decisions.md` 2026-09-01).
4. Qwen3.5-4B student.
