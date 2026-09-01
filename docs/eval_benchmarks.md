# Math-reasoning eval benchmarks: survey and recommendation

Context: student Qwen3.5-2B, teacher Qwen3.5-9B, on-policy distillation on math.
Current eval: `HuggingFaceH4/MATH-500` test, avg@4, T=1.0 / top_p 0.95 / top_k 20,
cap 8192 (16384 in `oracle16k`), strict Math-Verify on `\boxed{}` (no box = wrong).
Budget: ~15 min per 2000 generations at 8k cap on 2xA100.

Sources surveyed (2025-2026): Qwen3 technical report (arXiv 2505.09388), Qwen3.5
model cards (2B/4B/9B/27B/397B), Qwen3-4B-Thinking-2507 card, Thinking Machines
"On-Policy Distillation" blog, Revisiting OPD (2603.25562), Rethinking OPD
(2604.13016), SEAD (2606.28562), FiRe-OPD (2606.02684), DAPO (2503.14476),
DeepSeek-R1 (model card), MathArena (2505.23281), "A Sober Look at Progress in LM
Reasoning" (2504.07086). HF dataset ids, splits, sizes and columns were verified
against the HF datasets-server API on 2026-09-01.

## 1. What the papers use

| Paper / card | Math benchmarks | Protocol | Cap (tokens) |
|---|---|---|---|
| Qwen3 tech report | MATH-500, AIME'24, AIME'25 (+ others) | thinking: T=0.6, top_p .95, top_k 20; **AIME avg@64** | 32,768; **38,912 for AIME** |
| Qwen3.5 cards (0.8B-397B) | **HMMT Feb 25, HMMT Nov 25** (small cards); AIME26, IMOAnswerBench (397B) | T=1.0, top_p .95, top_k 20, presence_penalty 1.5 (small cards); number of runs not stated on the cards | recommends **81,920** for competition math |
| Qwen3-4B-Thinking-2507 card | AIME25, HMMT25 | as above | 81,920 |
| DeepSeek-R1 (distill 1.5B/7B) | AIME 2024, MATH-500 (+ others) | T=0.6, top_p .95, **64 samples -> pass@1 (= avg@64)** | 32,768 |
| DAPO | AIME 2024 | T=1.0, top_p 0.7, **avg@32** | 16,384 + 4,096 overlong buffer = 20,480 |
| Thinking Machines OPD blog | AIME'24 (Qwen3-8B-Base student) | not stated | not stated |
| Revisiting OPD (2603.25562) | MATH-500, AIME24, AIME25, Minerva, OlympiadBench | T=1.0, top_p 0.9; pass@1 / avg@32 on the small sets | 16,384 |
| Rethinking OPD (2604.13016) | AIME 2024, AIME 2025, AMC 2023 | T=0.7, top_p .95; **avg@16** | 31,744 |
| SEAD (2606.28562) | MATH-500, Minerva, OlympiadBench, AMC23, AIME24, AIME25 | greedy pass@1 on big sets; **avg@32** on AMC/AIME; T=0.7 | 16,384 |
| FiRe-OPD (2606.02684) | AIME24, AIME25, MATH-500, AMC23, OlympiadBench, Minerva, HMMT Feb 25, HMMT Nov 25 | T=1.0, **avg@8** | not captured |
| MathArena | AIME 25/26, HMMT Feb/Nov 25, BRUMO 25, CMIMC 25, Apex | provider-recommended params, **4 runs averaged**, sympy + LLM fallback grading | provider max |
| Sober Look (2504.07086) | AIME24/25, AMC23, MATH-500, Minerva, OlympiadBench | mean +- std over **>=10 seeds** (3 for MATH-500) | 32,768 recommended |

The de-facto standard for a distillation paper is the six-set list used by SEAD /
Revisiting OPD / FiRe-OPD: **MATH-500, AIME24, AIME25, AMC23, Minerva, OlympiadBench**,
with HMMT Feb/Nov 25 added by the newest papers and by every Qwen3.5 card.

## 2. Comparison table

Naive SE = 100 * sqrt(p(1-p) / (n*k)) in points, at a plausible 2B-student p (see section 6 for the caveat).
"2B-scale reference" numbers are the closest published thinking-mode scores; caps differ and all are far above our 8k/16k.

| Benchmark | n | Answer format | Grader | HF id (split; columns) | Contamination / reuse | 2B-scale reference (cap) | Typical protocol | Naive SE @ k=4 / 16 | Cap needed |
|---|---|---|---|---|---|---|---|---|---|
| MATH-500 | 500 | free-form LaTeX | Math-Verify | `HuggingFaceH4/MATH-500` (test; `problem`,`answer`,`solution`,`subject`,`level`) | MATH test is in many SFT mixes; near-saturated for thinking models | Qwen3-1.7B 93.4, Qwen3-4B 97.0 (32k); **ours: Qwen3.5-2B strict ~0.47 at 8k** | avg@4-16 or greedy | 1.1 / 0.5 (p=.5) | 8k gives ~45-85% cap-hit here (oracle16k runs, strict); 16k halves that |
| AIME 2024 | 30 | integer 0-999 | exact int | `HuggingFaceH4/aime_2024` (train; `problem`,`answer`,`solution`) or `math-ai/aime24` (test; `problem`,`solution` -- no answer col) | **Known contaminated**: MathArena shows models 10-20 pts above the human-aligned line | Qwen3-1.7B 48.3, Qwen3-4B 73.8 (avg@64, 38.9k); R1-Distill-1.5B 28.9 (32k) | avg@16-64 | 4.2 / 2.1 (p=.3) | >=16k, papers use 20-39k |
| AIME 2025 | 30 | integer | exact int | `MathArena/aime_2025` (train; `problem_idx`,`problem`,`answer`(int64),`problem_type`) or `math-ai/aime25` (test; `problem`,`answer`) | Feb 2025; older than Qwen3.5 data; MathArena finds only small inflation | Qwen3-1.7B 36.8, Qwen3-4B 65.6 (avg@64); Qwen3-4B-Thinking-2507 81.3 (81.9k); R1-Distill-1.5B 20.0 (MathArena) | avg@16-64 | 4.0 / 2.0 (p=.25) | >=16k |
| AIME 2026 | 30 | integer | exact int | `MathArena/aime_2026` (train; `problem_idx`,`problem`,`answer`(int64)) | Held Feb 5/11 2026; Qwen3.5 small released Mar 2 2026 -- most likely post-cutoff, least contaminated | Qwen3.5-397B 91.3; no small-model numbers yet | MathArena 4 runs | ~4 / 2 | >=16k |
| AMC 2023 | 40 | integer | exact int | `math-ai/amc23` (test; `question`,`answer`(str int)) | 2023, in most 2024+ corpora; easier than AIME | no official Qwen3/3.5 number | avg@16-32 | 3.6 / 1.8 (p=.7) | 8k mostly OK, 16k safe |
| HMMT Feb 2025 | 30 | mixed: ints and LaTeX (`\frac{1}{576}`) | Math-Verify | `MathArena/hmmt_feb_2025` (train; `problem_idx`,`problem`,`answer`(str),`problem_type`) | **Qwen3.5 headline benchmark**; less prominent than AIME -> less contamination, but Qwen tuned for it | **Qwen3.5-2B 22.9**, Qwen3-1.7B 10.2, Qwen3-4B-2507 57.5, Qwen3.5-4B 74.0, Qwen3.5-9B 83.2 (81.9k); R1-Distill-1.5B 11.7 | avg@4-16 | 3.7 / 1.8 (p=.2) | >=16k, harder than AIME -> longer |
| HMMT Nov 2025 | 30 | mixed | Math-Verify | `MathArena/hmmt_nov_2025` (train; `problem_idx`,`problem`,`answer`(str)) | Nov 2025; **Qwen3.5-2B 19.6**, Qwen3-4B-2507 69.6, Qwen3.5-4B 76.8, 9B 82.9 | | avg@4-16 | 3.7 / 1.8 | >=16k |
| HMMT Feb 2026 | 33 | mixed | Math-Verify | `MathArena/hmmt_feb_2026` (train; `problem_idx`,`problem`,`answer`,`problem_type`) | Feb 2026, post-cutoff | none | 4 runs | ~3.5 / 1.8 | >=16k |
| BRUMO 2025 | 30 | free-form (`2^{99}`, `\frac{\sqrt{13}}{2}`) | Math-Verify | `MathArena/brumo_2025` (train; same cols as hmmt_feb) | 2025, obscure -> clean | none | 4 runs | ~3.5 / 1.8 | >=16k |
| CMIMC 2025 | 40 | free-form | Math-Verify | `MathArena/cmimc_2025` (train; same cols) | 2025, obscure -> clean | none | 4 runs | ~3 / 1.5 | >=16k |
| BeyondAIME | 100 | integer | exact int | `ByteDance-Seed/BeyondAIME` (test; `problem`,`answer`(int64)) | Original, manually rewritten problems -> contamination-resistant; difficulty >= AIME #11-15 | none for 2B; expect <10% -> floor effects | avg@8-32 | 1.8 / 0.9 (p=.15) | >=16k, likely 32k |
| Minerva Math | 272 | free-form numeric with units / sig figs (`4.5e33`, `41.8`) | Math-Verify (lossy on units) | `math-ai/minervamath` (test; `question`,`answer`) | old (2022), physics-flavoured; grading noise is the known problem | none official | greedy or avg@4 | 1.5 / 0.8 (p=.45) | 8k OK |
| OlympiadBench (OE_TO_maths_en_COMP) | 674 | free-form; `final_answer` is a **list** | Math-Verify | `math-ai/olympiadbench` (test; `question`,`final_answer`(seq),`answer_type`,`is_multiple_answer`) | 2024, in SFT mixes | none official | greedy or avg@4 | 1.0 / 0.5 (p=.55) | 8-16k |
| GSM8K | 1319 | integer, gold as `#### N` | parse `####` | `openai/gsm8k` (config `main`, test; `question`,`answer`) | saturated (>90) for any thinking model | Qwen3 does not even report it | greedy | 0.4 / 0.2 (p=.9) | 2-4k |
| Omni-MATH | 4428 | free-form, originally judged by Omni-Judge | LLM judge | `KbsdJames/Omni-MATH` (test; `problem`,`answer`,`solution`,`domain`,`difficulty`) | 2024; too big and judge-dependent for us | -- | -- | -- | 16k+ |
| PolyMath (en) | 4 x 125 | free-form | Math-Verify | `Qwen/PolyMath` (config `en`; splits `low`,`medium`,`high`,`top`; `id`,`question`,`answer`) | Qwen's own multilingual set; `top` level is olympiad | Qwen3 cards report it | avg@k | 1.8 / 0.9 per level | 16k+ |
| LiveMathBench | 238 (202412) | free-form | Math-Verify | `opencompass/LiveMathBench` -- **gated (`auto`)**, raw jsonl per source, no `datasets` config | 2024-12 vintage; "live" idea but now a year old | -- | -- | -- | 16k |
| Omni/AIMO validation | 90 / 83 | integer | exact int | `AI-MO/aimo-validation-aime` (train; 90 = AIME 22-24), `AI-MO/aimo-validation-amc` (train; 83) | AIME 2022-24 -> contaminated | -- | -- | -- | >=16k |

Integer-only sets (cheap, unambiguous exact-match on the boxed integer, no
Math-Verify ambiguity): **AIME 24/25/26, AMC23, BeyondAIME, GSM8K**. HMMT, BRUMO,
CMIMC, MATH-500, Minerva, OlympiadBench, PolyMath need Math-Verify (HMMT is
~2/3 integer, 1/3 fractions/radicals).

`math-ai/aime24` has no `answer` column (only `solution`); use `HuggingFaceH4/aime_2024`.
`MathArena/aime_2024` does not exist.

## 3. Token-length requirements

What the papers use for AIME-class problems: DeepSeek-R1 32,768; Qwen3 38,912 for
AIME (32,768 elsewhere); Qwen3.5 cards recommend 81,920; Rethinking OPD 31,744;
DAPO 20,480; SEAD and Revisiting OPD 16,384 (both report AIME numbers that are
clearly cap-limited: 10-23% for 7B students). The Sober Look paper's explicit
finding: reducing `max_new_tokens` "harms performance, especially on long-form
problems"; they standardise on 32,768.

Our own measurements (strict scoring, so a cap-hit *is* an incorrect answer):

- Qwen3.5-2B on the OpenThoughts training pool at cap 8192: **91% truncated**,
  mean length 7.9k (`outputs/runs/kl50/bucket_stats.jsonl`, round 0).
- Qwen3.5-2B on MATH-500 at cap 16384: cap-hit 0.45-0.85 across rounds
  (`oracle16k`, `oracle16k_seed2`), mean length 10-15k.
- Qwen3.5-9B teacher on MATH-500 at cap 8192: cap-hit 0.47 on 100 problems
  (`outputs/runs/teacher_eval`).

MATH-500 is the *easy* end. AIME/HMMT solutions from thinking models are
routinely 15-30k tokens; at 8k the student's AIME/HMMT strict score will be
dominated by truncation, not ability, and at 16k it will still be a
"score-under-a-16k-budget" number, comparable to SEAD / Revisiting OPD
(16,384) but not to Qwen or DeepSeek cards (32-82k). Always report cap-hit rate
next to accuracy, and never compare our AIME/HMMT numbers to card numbers.

Throughput implication: 2000 gens at 8k = 15 min. Long-cap generations cost
roughly proportionally more per token plus a decode slowdown from KV growth, so
budget ~1000 gens per 15 min at 16k and ~400-500 at 32k.

## 4. Recommended suite

### Per-round cheap monitor (~15 min, run every round on every arm)

Keep **MATH-500 avg@4 at 8192** as the only per-round metric. It is the
largest set (n=500 -> naive SE ~1.1 pt at p=0.5), it is the one metric all
existing rounds share, and every cross-arm comparison in `outputs/runs` is
anchored on it. Do not add AIME/HMMT per round: at 8k cap they are almost all
truncation, and 30-problem sets cannot separate arms round-to-round (section 6).

Optional +3 min if a harder integer signal is wanted mid-run: **AMC23 avg@8 at
8192** (320 gens, exact-int grading). Treat it as a trend line only.

### Terminal full eval (once per final checkpoint per arm, plus the base
student and the teacher; ~1.5-2 h per checkpoint at 16k)

| Set | k | Gens | Cap | Grader | Why |
|---|---|---|---|---|---|
| MATH-500 | 4 | 2000 | 16384 | Math-Verify | same set as the monitor, at the "real" cap; compare to the 8k number to quantify cap effect |
| AIME 2025 | 16 | 480 | 16384 | exact int | standard in every 2025-26 OPD paper; less contaminated than AIME24 |
| AIME 2026 | 16 | 480 | 16384 | exact int | post-cutoff for Qwen3.5; the clean number |
| HMMT Feb 2025 | 16 | 480 | 16384 | Math-Verify | Qwen3.5's own headline set; the one card number for Qwen3.5-2B (22.9 at 81.9k) |
| HMMT Nov 2025 | 16 | 480 | 16384 | Math-Verify | second Qwen3.5 card set (19.6) |
| AMC 2023 | 16 | 640 | 16384 | exact int | easier integer set where a 2B model is in the informative 40-80% band |
| AIME 2024 | 16 | 480 | 16384 | exact int | only for comparability with DeepSeek/Qwen3/TM-blog numbers; flag as contaminated |

Total ~5,000 generations at 16k ~ 1.3-1.7 h per checkpoint. Pool AIME 24+25+26
into one 90-problem "AIME" number (naive SE ~1.2 pt at k=16) and HMMT Feb+Nov into
one 60-problem number; report the per-year splits in an appendix.

Skip: GSM8K (saturated, no signal), Minerva (unit/sig-fig grading noise, old),
Omni-MATH (needs LLM judge, 4.4k problems), PolyMath (multilingual; `en/top`
duplicates the AIME/HMMT role), LiveMathBench (gated, non-standard loading, stale),
BRUMO/CMIMC (clean but no published reference points; add later if the
MathArena-style aggregate is wanted). BeyondAIME is clean and integer-graded but a
2B student at 16k will sit near the floor (<10%); add it only when a 32k terminal
eval exists.

If a 32k terminal pass is affordable for the final 2-3 checkpoints only (~2.5x the
cost), run AIME 25/26 + HMMT Feb 25 at avg@16 at 32768; that is the number to put
next to Rethinking OPD (31,744) and DeepSeek-R1 (32,768).

## 5. Exact loader entries for the recommended sets

All verified via `https://datasets-server.huggingface.co/info?dataset=...` on 2026-09-01.
`load_dataset(id, split=...)` with no config argument works for every id below.

| Key | HF id | split | problem col | answer col | answer dtype | notes |
|---|---|---|---|---|---|---|
| `math500` | `HuggingFaceH4/MATH-500` | `test` | `problem` | `answer` | str | already registered |
| `aime24` | `HuggingFaceH4/aime_2024` | `train` | `problem` | `answer` | str | 30 rows; `id` 60-89, `year` |
| `aime25` | `MathArena/aime_2025` | `train` | `problem` | `answer` | **int64** | `str()` it; `problem_type` list |
| `aime26` | `MathArena/aime_2026` | `train` | `problem` | `answer` | **int64** | `str()` it |
| `amc23` | `math-ai/amc23` | `test` | `question` | `answer` | str (int) | 40 rows |
| `hmmt_feb25` | `MathArena/hmmt_feb_2025` | `train` | `problem` | `answer` | str (LaTeX) | 30 rows |
| `hmmt_nov25` | `MathArena/hmmt_nov_2025` | `train` | `problem` | `answer` | str | 30 rows |
| `beyondaime` | `ByteDance-Seed/BeyondAIME` | `test` | `problem` | `answer` | **int64** | 100 rows; optional |

These drop straight into `DATASETS` / `COLUMNS` in `apod/datasets/load.py`
(`problem`, `answer`, `solution=None`, `cot=None`, `source=None`, `correct=None`);
`_example` already `str()`s the answer, so the int64 columns are fine. The
`BOXED` instruction matches the Qwen3.5 card's recommended math prompt.
MathArena ids are CC BY-NC-SA 4.0 (fine for research use).

Grading: for the integer sets, the strict rule is "boxed content parses to an
int and equals gold"; keep Math-Verify as the implementation so `\boxed{070}`
and `\boxed{70}` both pass, but the exact-int nature means no equivalence
ambiguity. For HMMT, Math-Verify handles `\frac{1}{576}` vs `1/576` vs
`0.001736...`; MathArena additionally used an LLM fallback, which we do not need
at 2B accuracy levels.

## 6. The tiny-benchmark noise problem

Naive Bernoulli standard error, in points, of avg@k = 100 * sqrt(p(1-p)/(n*k)):

| Set (n) | p | k=1 | k=4 | k=8 | k=16 | k=32 | k=64 |
|---|---|---|---|---|---|---|---|
| AIME (30) | 0.30 | 8.4 | 4.2 | 3.0 | 2.1 | 1.5 | 1.0 |
| HMMT Feb 25 (30) | 0.20 | 7.3 | 3.7 | 2.6 | 1.8 | 1.3 | 0.9 |
| AMC23 (40) | 0.70 | 7.2 | 3.6 | 2.6 | 1.8 | 1.3 | 0.9 |
| BeyondAIME (100) | 0.15 | 3.6 | 1.8 | 1.3 | 0.9 | 0.6 | 0.4 |
| MATH-500 (500) | 0.50 | 2.2 | 1.1 | 0.8 | 0.5 | 0.4 | 0.3 |
| OlympiadBench (674) | 0.55 | 1.9 | 1.0 | 0.7 | 0.5 | 0.3 | 0.2 |

Trials needed for a target SE under the naive model: SE = 2 pt needs 400-625
trials (p from 0.2 to 0.5); SE = 1 pt needs 1,600-2,500. So AIME at avg@16 (480
trials) is a "+-2 pt" measurement on paper, and MATH-500 avg@4 (2,000) a "+-1 pt"
one. This is why the papers use avg@16-64 on AIME and why Qwen uses avg@64.

The caveat: the n*k trials are not i.i.d. draws of one p. Each problem i has its
own p_i, and the k samples of a problem share it.

- If the estimand is "mean accuracy on *these* 30 problems", the sampling
  variance is sum_i p_i(1-p_i) / (n^2 k), which is <= the naive value and does
  shrink with k. The naive table is then conservative, and k is doing real work.
- If the estimand is "accuracy on AIME-style problems in general" (what a paper
  claim implies), there is an additional between-problem term Var_i(p_i)/n that
  **does not shrink with k**. With p_i spread across [0,1] (SD ~0.35 is typical
  for a model at 30% on AIME), that term alone is ~6 pt SE at n=30 no matter how
  large k is. Raising k past ~16 buys nothing against it; only more problems do
  (hence pooling AIME 24+25+26 -> n=90, SE ~3.7 pt from that term).
- There is a third, run-level term: the Sober Look paper measured 5-15 pt SD in
  single-run AIME24 pass@1 for 1.5B models across seeds and recommends >=10
  runs; our own seed replicate (`oracle16k` vs `oracle16k_seed2`, same config)
  shows 4-7 pt gaps on MATH-500 at a fixed round and one 20-pt within-run
  collapse tied to a length blowup. A single-seed AIME delta of 5 pts between
  two arms is therefore noise, not a result.

Practical rules: (a) compare arms at a fixed round on the same problem set and
same seeds, and quote a paired / problem-level cluster bootstrap CI (resample
problems, keep all k samples of a problem together), not the naive SE; (b) report
cap-hit rate with every accuracy; (c) treat any 30-problem set as a sanity check
against published numbers, and let MATH-500 (n=500) and the pooled 90-problem
AIME carry the arm-vs-arm claims; (d) if a claim rests on AIME/HMMT, run at least
2 seeds of the terminal eval.
