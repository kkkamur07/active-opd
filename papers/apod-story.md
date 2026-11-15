# Storyline v4: what makes a good rollout for on-policy post-training (2026-11-15)

Supersedes v3 (aligned-band headline). v3 lives in git history (`papers/apod-story.md` at `b7fc097`).
Evidence: the 2B selection program, seed 1 (`p2b-{rollout,qkl,correct}-s1`) plus seed 2 of the question-KL group
(partial, steps 0 to 45). Analysis code and outputs: `experiments/p2b-analysis/` (`analyze_s1.py` -> `s1_analysis.json`,
`curve_ci.py` -> `curve_ci.json`, `analyze_signals.py` -> `signals.json`, `finish_decomp.txt`). Raw data pulled from the VM
into `outputs/runs_results/p2b/`.

## 1. The question

On-policy post-training: the student samples, the teacher scores every token, the student minimizes reverse KL.
Each rollout costs a long generation plus a teacher pass, and the usual recipe trains on whatever was sampled.
**Which rollouts are worth training on?** Three properties are proposed in the literature and in our own notes
(CONTEXT.md, apod-story v3) as markers of a good rollout:

| Property | What it measures | Rule tested at rollout level | Rule tested at question level |
|---|---|---|---|
| Correctness | does the answer reach the right boxed result | (not separable: within a question 8.3 to 9.0% correct per KL band) | 4 buckets by teacher/student pass: T right S wrong, both right, mixed, both wrong |
| Distributional alignment | exact per-token reverse KL(student, teacher) on the student's own tokens | keep KL ranks 1-4 / 5-8 / 9-12 of 12 rollouts | keep the high / middle / low third of 360 questions by mean KL |
| Uncertainty | student's mean token entropy | keep the 4 highest-entropy of 12 (classic active learning) | (not run: within a question, entropy and KL rank rollouts alike, Spearman 0.69) |

12 arms, equal budget (480 trained rollouts per refresh, 15 steps per refresh, 60 steps), one shared random baseline,
Qwen3.5-2B student, Qwen3.5-9B teacher, 8,192-token budget, MATH-500 strict avg@4 at steps 0/15/30/45/60.

## 2. What we found (seed 1; "robust" = structural fact or replicated, "provisional" = waiting for seeds 2-3)

1. **Training mostly teaches the student to finish (robust).** When the 2B finishes inside 8,192 tokens it is right
   99.4 to 100% of the time; every error is a run-out-of-budget answer. Training lifts random from 26.5% to 42.9% and the
   finish rate from 20.5% to 36.8%. Across all 48 trained checkpoints, strict accuracy tracks finish rate (r = 0.80).
2. **Which rollout of a question you keep does not matter (robust in mechanism, provisional in size).** All four
   rollout-level rules land within 0.35 points of random over the curve (CIs include 0). The three KL bands are nearly
   the same rollouts: correctness 9.0 / 8.3 / 8.8%, truncation 95 / 94 / 91%, and 84.6% of KL variance is between
   questions (2B pilot: 76 to 86%). Entropy-top and KL-top pick the same rollouts 2.63 of 4 times (chance 1.33).
   **The aligned-band hypothesis (keep the middle KL third) is not supported**: -0.30 [-1.07, +0.47] (pre-registered
   primary contrast).
3. **Which question you train on matters (provisional, partly replicated).** Question-level rules spread 3.6 points
   (+1.7 to -1.9) against 0.6 for rollout-level rules.
   - Question KL is monotone: high +1.69 [+0.85, +2.53], middle -0.18, low -1.93 [-2.75, -1.12].
     Seed 2 (partial) repeats high > middle: +1.58 vs +1.80 in seed 1 (steps 15-45).
   - Correctness: T right S wrong +0.90 [+0.03, +1.76], both right +0.80, mixed +0.07, both wrong -1.71 [-2.59, -0.84].
     Both wrong hurt in the earlier 2B question-selection run too (-2.59), a different design.
   - The differences show up as finishing: qkl_high +2.0 of its +1.7 comes from finish rate, qkl_low -2.0 of -1.9.
4. **Good questions are not "where the teacher knows more" (robust, descriptive).** The high-KL third at round 0 is
   mostly both-wrong questions (86 of 120), with the lowest teacher pass rate (24.8% vs about 40% for the other thirds),
   the highest student entropy (0.66 vs 0.40) and teacher entropy (0.58 vs 0.34). Question KL barely correlates with
   pass rate (Spearman -0.15 teacher, -0.11 student). KL and correctness are different signals.

## 3. Intuition (label as hypothesis in the paper)

- The loss sees tokens, not outcomes. Reverse-KL post-training moves the student wherever its next-token distribution
  differs from the teacher's, whatever the final answer. So the question property that should matter is how much
  per-token disagreement a question carries, which is question KL. Low-KL questions give the student almost nothing to
  move toward (qkl_low trained on 0.081 nats per token vs 0.129 for qkl_high, mean of 4 refreshes) and waste a refresh.
- Rollouts of one question share the same prefix-level difficulty, so they carry about the same disagreement; picking
  among them moves the mean KL of the kept set by about 9% (0.112 to 0.134 nats at round 0) and nothing happens.
- High-KL questions are the ones where both models are uncertain and the student writes long, hesitant traces. At an 8k
  budget the most useful thing a 2B can learn is to commit and stop; we believe the teacher's corrections on these
  traces carry exactly that signal. Test: locate high-KL tokens (hesitation words, loop restarts) [need].
- Both-wrong questions have average KL (0.105 nats, the same as random) and 99% truncated rollouts: hard, but without
  extra teacher signal. Hard alone is not enough; disagreement is what teaches.

## 4. Abstract (draft)

See `papers/apod-acl/main.tex`.

## 5. Assumptions we write under (and what would break them)

| Claim in the paper | Status | Breaks if |
|---|---|---|
| Finished answers are ~always correct; training gains are mostly finishing | robust (2,000 samples per checkpoint, all 49 checkpoints) | n/a |
| Rollout-level selection ≈ random | mechanism robust; size provisional | seeds 2-3 show a rollout rule beyond +-1 pt consistently |
| Question KL high > low | provisional, seed 2 partial agrees | seeds 2-3 flip the sign or shrink it below 1 pt |
| Both-wrong hurts | provisional, agrees with pilot | seeds 2-3 put it at random |
| Aligned band not supported | provisional | seeds 2-3 put kl_mid clearly above random |

## 6. Runs that will settle it

1. Seeds 2-3 for every group (queued: qkl-s3, rollout-s2/s3; correct-s2/s3 after bank widening).
2. Held-out best-checkpoint pick on otval200 (queued: 900-p2b-otval-s1).
3. Re-evaluation at a 16k or 32k budget: are the gains finishing only, or also solving.
4. Token-level location of KL in high-KL questions (no GPU training; scores exist).
5. Question-KL selection with a cheaper score (mean entropy of 4 rollouts; ranks questions like KL?) [need].
