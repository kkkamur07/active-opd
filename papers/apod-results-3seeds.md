# Results (three training seeds)

Sources: `experiments/p2b-analysis/three_seeds.txt`, `three_seeds_summary.txt` (from the 15 p2b runs, copied 2026-11-18 to
`~/Desktop/krrish/apod-results-20261118/runs`). Metric: MATH-500 strict avg@4 at the 8k budget, averaged over steps 15, 30, 45 and 60.
"Minus random" is against the random arm of the same seed. Points, no intervals; the seeds are the noise measure.

Every arm starts from the base student (26.87% averaged over the three seeds) and trains on 480 answers per refresh for 60 steps.
Random selection reaches 42.58% over steps 15 to 60 and 42.70% at step 60. The teacher scores 61.15%.

| Rule | Mean accuracy (%) | Minus random | Seed 1 | Seed 2 | Seed 3 | Above random |
|---|---|---|---|---|---|---|
| Random | 42.58 | | | | | |
| **Choosing answers** | | | | | | |
| KL high | 42.31 | −0.27 | −0.34 | −0.33 | −0.15 | 0 of 3 |
| KL middle | 42.57 | −0.01 | −0.30 | +0.52 | −0.26 | 1 of 3 |
| KL low | 42.45 | −0.12 | +0.25 | −0.67 | +0.05 | 2 of 3 |
| Entropy high | 42.45 | −0.13 | −0.06 | +0.04 | −0.35 | 1 of 3 |
| **Choosing questions by uncertainty** | | | | | | |
| Entropy high (4 answers) | 43.60 | +1.02 | +1.35 | +1.02 | +0.68 | 3 of 3 |
| Entropy high (1 answer) | 43.65 | +1.08 | +1.34 | +1.06 | +0.83 | 3 of 3 |
| Entropy middle | 42.11 | −0.47 | −0.29 | −0.56 | −0.55 | 0 of 3 |
| Entropy low | 42.02 | −0.56 | −0.71 | +0.38 | −1.35 | 1 of 3 |
| **Choosing questions by alignment (teacher KL)** | | | | | | |
| KL high | 43.67 | +1.09 | +1.69 | +1.45 | +0.13 | 3 of 3 |
| KL middle | 42.33 | −0.25 | −0.18 | −0.16 | −0.40 | 0 of 3 |
| KL low | 41.52 | −1.06 | −1.93 | +0.17 | −1.42 | 1 of 3 |
| **Choosing questions by correctness** | | | | | | |
| Teacher right, student wrong | 43.80 | +1.22 | +0.90 | +1.24 | +1.53 | 3 of 3 |
| Both right | 43.49 | +0.91 | +0.80 | +1.10 | +0.84 | 3 of 3 |
| Mixed | 42.85 | +0.27 | +0.07 | +0.41 | +0.34 | 3 of 3 |
| Both wrong | 41.32 | −1.26 | −1.71 | −0.23 | −1.85 | 0 of 3 |

## 1. Choosing answers does not help

None of the four answer-level rules moves away from random. Their means sit between −0.27 and −0.01 points, and no rule is
above random in all three seeds. This includes the aligned middle KL band we set out to test (−0.01) and the classic
active-learning rule of keeping the most uncertain answers (−0.13). Once a question is chosen, which of its answers we train on
does not matter.

## 2. Choosing questions by the student's uncertainty helps

Training on the third of questions the student is most uncertain about beats random in all three seeds, by +1.02 points on
average. The rule needs no teacher: it uses the student's own entropy, and scoring each question from a single answer works
just as well (+1.08, above random in all three seeds). The
middle and low thirds lose 0.47 and 0.56 points. Question uncertainty picks almost the same questions as teacher KL (in seed
1, 99 to 106 of 120 per refresh), and the two rules give the same gain (+1.02 and +1.09). So the student can find the useful
questions itself, at the cost of one sampled answer per candidate.

## 3. Questions nobody can solve hurt, and questions the teacher can solve help

Questions that neither the teacher nor the student solves lose 1.26 points, below random in all three seeds. Every bucket
the teacher can solve helps: teacher right and student wrong gains 1.22 points (the largest and most consistent gain of any
rule), both right gains 0.91, and mixed questions 0.27. Half of our labeled pool is questions neither model solves, so
random selection spends about half of its budget on the one bucket that hurts.

## 4. Size of the effects

The useful rules gain about one point over random, against a 16-point gain from training itself. They are consistent rather
than large: each of the four best rules is above random in all three seeds. Single seeds vary by up to two points for the
same rule (teacher-KL low: −1.93, +0.17, −1.42), so we read the three-seed means and the sign counts, not any single run.

## What these results do not show

- We did not run a "random minus both-wrong questions" arm, so the size of the gain from that filter is inferred, not measured.
- One model pair (Qwen3.5-2B and -9B), one training set, one learning rate, one 8k budget, MATH-500 only.
- Correctness labels come from the base student and are not updated as it trains.
