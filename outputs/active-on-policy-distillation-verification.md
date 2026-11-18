# Verification report: papers/active-on-policy-distillation.md

Date: 2026-11-19. Raw data: `~/Desktop/krrish/apod-results-20261118/runs/`. Recompute scripts (temporary): `/tmp/recomp.py` (MATH-500 arms), `/tmp/var.py` (variance shares), `/tmp/ov.py` (overlap), plus inline checks of the banks and question provenance described below. I did not use `experiments/p2b-analysis/*.txt` as inputs. I only compared them with my results afterwards.

## Summary

- Every result number (Tables 2 to 4, the abstract, intro, Sections 5 and 6, and the Figure 2/3 data) matches my recomputation from raw data at the precision printed. One caveat: seven three-seed or per-seed values fall exactly on a rounding tie (x.xx5), and the draft rounds them inconsistently. I did not edit them (see 1.2).
- **One substantive mismatch (data description, Section 4.2).** The correctness arms do not all draw from the 4,641 labeled bank-8k questions:
  - Seeds 2 and 3 use `bank-8k-w`, a widened bank of 25,000 questions with 7,317 teacher-labeled.
  - The training "mixed" bucket includes questions where the student is mixed and the teacher was never run.
  
  I corrected this in the text. The numbers 4,641 / 2,387 / 971 / 866 / 417 are correct for bank-8k.
- Citations: all 25 are on the allowed list, the IDs resolve, and the titles and first authors match. I softened three attributions (Hou, Xia, and the conclusion's "disagree" claim) and fixed two author-name forms (Song and Zheng; Ding and Zhang).
- Excluded topics: none found. Em/en dashes: none found, before or after the edits.

## JOB A: numbers

Metric: MATH-500 `strict_avg_at_n` x100, averaged over steps 15, 30, 45 and 60. "Minus random" is that mean minus the same seed's `random` arm, taken from p2b-rollout-sN. I spot-checked the strict definition (`correct AND has_boxed`) against the raw eval shards: p2b-rollout-s1 random step 0 recomputes to 0.265, which matches the summary. No duplicate (arm, step) rows. Every arm has steps 0, 15, 30, 45 and 60 in all three seeds.

### 1.1 Baselines and random

| Draft location | Value in draft | Recomputed | Source | Status |
|---|---|---|---|---|
| Table 2, Sec 5: 2B base (mean of the three random step-0 evals) | 26.87 | 26.8667 (26.5, 27.1, 27.0) | p2b-rollout-s{1,2,3}/metrics.jsonl random step 0 | MATCH |
| Table 2: 4B | 57.15 | 57.15 (also recomputed from raw p0-4b-base8k eval shards: 0.5715) | HANDOFF.md §5 line 133 | MATCH |
| Table 2: 9B teacher | 61.15 | 61.15 (recomputed from raw kl50_teacher eval shards: 0.6115) | todo.md kl50_teacher | MATCH |
| Tables 3/4, Sec 5: random mean | 42.58 | 42.5792 | metrics.jsonl | MATCH |

### 1.2 Question-level (Table 3; abstract; intro; Sec 5.1; Limitations)

| Rule | Draft mean / minus / s1 / s2 / s3 / above | Recomputed (exact) | Status |
|---|---|---|---|
| t_right_s_wrong | 43.80 / +1.22 / +0.90 / +1.24 / +1.53 / 3 | 43.8000 / +1.2208 / +0.900 / +1.2375 / +1.525 / 3 | MATCH |
| qkl_high | 43.67 / +1.09 / +1.69 / +1.45 / +0.13 / 3 | 43.6667 / +1.0875 / +1.6875 / +1.450 / +0.125 / 3 | MATCH |
| qent1_high | 43.65 / +1.08 / +1.34 / +1.06 / +0.83 / 3 | 43.6542 / +1.0750 / +1.3375 / +1.0625 / +0.825 / 3 | MATCH |
| both_right | 43.49 / +0.91 / +0.80 / +1.10 / +0.84 / 3 | 43.4917 / +0.9125 / +0.800 / +1.100 / +0.8375 / 3 | MATCH |
| mixed | 42.85 / +0.27 / +0.07 / +0.41 / +0.34 / 3 | 42.8542 / **+0.2750** / **+0.0750** / +0.4125 / +0.3375 / 3 | MATCH (ties*) |
| qkl_mid | 42.33 / -0.25 / -0.18 / -0.16 / -0.40 / 0 | 42.3333 / -0.2458 / -0.175 / -0.1625 / -0.400 / 0 | MATCH |
| qkl_low | 41.52 / -1.06 / -1.93 / +0.17 / -1.42 / 1 | 41.5208 / -1.0583 / -1.925 / **+0.175** / **-1.425** / 1 | MATCH (ties*) |
| both_wrong | 41.32 / -1.26 / -1.71 / -0.23 / -1.85 / 0 | 41.3167 / -1.2625 / -1.7125 / -0.225 / -1.850 / 0 | MATCH |

### 1.3 Answer-level (Table 4; abstract "-0.27 to -0.01"; Sec 5.2)

| Rule | Draft | Recomputed (exact) | Status |
|---|---|---|---|
| kl_high | 42.31 / -0.27 / -0.34 / -0.33 / -0.15 / 0 | 42.3083 / -0.2708 / -0.3375 / -0.325 / -0.150 / 0 | MATCH |
| kl_mid | 42.57 / -0.01 / -0.30 / +0.52 / -0.26 / 1 | 42.5667 / -0.0125 / -0.300 / **+0.525** / -0.2625 / 1 | MATCH (tie*) |
| kl_low | 42.45 / -0.12 / +0.25 / -0.67 / +0.05 / 2 | 42.4542 / **-0.1250** / +0.250 / **-0.675** / +0.050 / 2 | MATCH (ties*) |
| ent_top4 | 42.45 / -0.13 / -0.06 / +0.04 / -0.35 / 1 | 42.4542 / **-0.1250** / -0.0625 / +0.0375 / -0.350 / 1 | MATCH (tie*) |

\*Rounding ties. The exact values are multiples of 0.0125, so the bold ones sit exactly halfway between two 2-decimal values. The draft takes whatever binary-float `:+.2f` formatting returned. The result is inconsistent: kl_low and ent_top4 have the identical exact difference -0.125, but print as -0.12 and -0.13 in the same table. Elsewhere, +1.075 prints as +1.08 while +0.275 prints as +0.27. Both printed values are within ±0.005 of the truth, so I marked these MATCH and **did not edit them**. Changing them in the text alone would desynchronise Figure 2, whose printed means come from the same float formatting in `papers/active-on-policy-distillation_fig.py`. Recommendation: choose one rule (for example round-half-away-from-zero, which would give mixed +0.28 / s1 +0.08, qkl_low s2 +0.18 / s3 -1.43, kl_mid s2 +0.53, kl_low -0.13 / s2 -0.68) and regenerate the tables and Figure 2 together. Alternatively, print 3 decimals for the tied cells.

Derived claims checked: "three rules above random in all three seeds" MATCH. "about one point each" MATCH. "both right ... above random in all three" MATCH. "mixed seed 1 only +0.07" MATCH (tie). "KL middle below random in all three" MATCH. "KL low one seed above (+0.17), two below (-1.93, -1.42)" MATCH. "no answer-level rule above random in all three seeds" MATCH (max 2 of 3). "random reaches 42.58 from 26.87" MATCH. Conclusion and new text "both-wrong 41.32 vs base 26.87" MATCH.

### 1.4 Between-question variance share (Sec 5.3, intro, abstract, Figure 3)

Recomputed from p2b-rollout-sN/arms/random/rounds/round_0R. KL uses `oracle_kl.shard*.jsonl` `mean_reverse_kl` and entropy uses `entropy.shard*.jsonl` `entropy`, both keyed by (example_index, rollout_index). Every pool has 120 questions x 12 answers. Share = Σ_q n_q (mean_q − grand)² / Σ (x − grand)².

| Seed, refresh | KL % | Entropy % |
|---|---|---|
| s1 r0..r3 | 84.59, 85.15, 80.03, 82.05 | 85.81, 82.90, 75.56, 77.01 |
| s2 r0..r3 | 85.19, 83.43, 81.55, 81.59 | 87.18, 82.40, 78.35, 79.03 |
| s3 r0..r3 | 82.11, 77.15, 83.76, 80.67 | 84.00, 73.44, 82.55, 78.15 |

| Draft value | Recomputed | Status |
|---|---|---|
| KL range 77.2% to 85.2%, mean 82.3% | 77.15 to 85.19, mean 82.27 | MATCH |
| Entropy range 73.4% to 87.2%, mean 80.5% | 73.44 to 87.18, mean 80.53 | MATCH |
| Figure 3 bar values (fig script lists) | all 24 agree to 0.1 | MATCH |

Note: these pools are **the random arm's** pools. Every answer-level arm samples its own pool after refresh 0, so "every answer-level pool" overstated the coverage. I edited the text to "answer pools of the random rule".

### 1.5 Single-answer-entropy vs KL overlap (Sec 5.3)

Recomputed from p2b-qent1-sN/arms/qent1_high/rounds/round_0R. I took the top 120 of 360 questions by rollout-0 entropy and the top 120 by mean `mean_reverse_kl` over each question's 4 rollouts. As a check, the arm's own `selected.jsonl` question set equals the top-120-by-entropy set in all 12 rounds.

Overlap per refresh: s1 97, 93, 90, 95; s2 102, 92, 96, 97; s3 96, 99, 89, **66**.

| Draft value | Recomputed | Status |
|---|---|---|
| Chance 40 | 120·120/360 = 40 | MATCH |
| 11 of 12 refreshes 89 to 102 | 89 to 102 in 11 | MATCH |
| One refresh (seed 3, refresh 3) = 66 | s3 round_03 = 66 | MATCH. "Refresh 3" was 0-indexed; text now says "the last refresh" |
| +1.08 vs +1.09 | see 1.2 | MATCH |

### 1.6 Seed-1-only numbers (signals.json)

| Draft value | signals.json | Labeled seed 1? | Status |
|---|---|---|---|
| Within-question correctness of KL bands 9.0 / 8.3 / 8.8% | A_band_props correct 0.0896 / 0.0833 / 0.0875 | yes ("base student's first pool in seed 1") | MATCH |
| Both-wrong counts in KL thirds 86 / 63 / 64 of 120 | B_thirds buckets both_wrong 86 / 63 / 64 | yes ("In seed 1"); heading now also says "(seed 1)" | MATCH |
| Teacher pass rate 24.8 / 40.8 / 39.4% | teacher_pass_bank 0.2479 / 0.4083 / 0.39375 | yes | MATCH |
| Spearman(question KL, teacher pass) -0.15 | B_spear_qkl_teacherpass -0.1466 | yes (same paragraph; "this analysis covers seed 1 only") | MATCH |
| 84.6%, Spearman -0.11, entropy 0.66/0.40, 0.58/0.34 | present in signals.json | n/a | not used in the draft |

### 1.7 Data and settings

| Draft value | Recomputed / source | Status |
|---|---|---|
| 14,500 sampled, 4,641 labeled | bank-8k/questions.jsonl: 14,500 rows, 4,641 with teacher labels (R5 says the same) | MATCH |
| Buckets 2,387 / 971 / 866 / 417 | (W,W) 2,387; (W,C) 971; (C,C) 866; teacher-labeled mixed 367+43+7 = 417 | MATCH |
| "More than half of the labeled pool is beyond both models" | 2,387 / 4,641 = 51.4% | MATCH |
| "Every rule draws its questions from these 4,641" | **False for the correctness arms.** p2b-correct-s2/s3 `driver.bank_dir: outputs/runs/bank-8k-w` (25,000 questions, 7,317 teacher-labeled, labels consistent with bank-8k on the shared 14,500). The "mixed" bucket also includes 561 (M, no-teacher) questions in bank-8k (967 in bank-8k-w). These make up 153 / 193 / 446 of the 480 mixed questions in seeds 1 / 2 / 3. Random, qkl_* and qent1 draw only from the 4,641 labeled questions. | **MISMATCH, fixed** |
| "questions are never reused" | 0 repeats within any run; 0 overlap across seeds for every arm checked | MATCH (now stated as "within or across seeds") |
| 3 of 4 = solves, at most 1 = fails | resolved_config bank.correct_min 3, wrong_max 1 | MATCH |
| Answer level: 120 q x 12, keep 4 | rollout.num_prompts 120, num_rollouts 12, selection.k 4 | MATCH |
| Question level: 360 q x 4, keep 120 | qkl/qent1 configs num_prompts 360, num_rollouts 4; 120 selected | MATCH |
| Correctness: 120 q x 4 from one bucket | p2b-correct configs num_prompts 120, num_rollouts 4, selection all_k, bank_bucket:* | MATCH |
| 480 answers per refresh | metrics selected.num_trajectories 480; 480/15 = 32 per step | MATCH |
| Refresh every 15 steps, 60 steps, 4 refreshes | driver.refresh_every 15, steps_total 60 | MATCH |
| β = 1, λ = 0 | train.beta 1.0, lmbda 0.0 | MATCH |
| Temperature 1.0, top-p 0.95, top-k 20, 8,192 cap | sampling block | MATCH |
| bf16, effective batch 32 | train.bf16 true, effective_batch 32; log_history response_tokens 262,144 = 32 x 8,192 | MATCH |
| LR 3.1623e-6, 3 warmup, cosine to 0.1 | learning_rate, warmup_steps 3, cosine_with_min_lr min_lr_rate 0.1 | MATCH |
| Thinking mode, Qwen3.5-2B / 9B | enable_thinking true; student_id / teacher_id | MATCH |
| Eval 500 problems x 4, 8,192 | eval block | MATCH |
| Three training seeds | train.seed 42 / 1042 / 2042. Note: question_offset also changes per seed, so seeds differ in training questions as well as in the training RNG | MATCH (the word "training seed" undersells this; not edited) |
| TRL GKD trainer | apod/stages/train.py uses GKDTrainer/GKDConfig | MATCH |

## JOB B: citations

All in-text citations are on the allowed list in `notes/apod-draft-references.md`. I fetched the arXiv API records for all 23 arXiv IDs: titles, first authors and years match. I cross-checked 2609.05198, 2610.18011, 2608.16647, 2604.00626 and 2604.13016 in OpenAlex (DOI 10.48550/arxiv.*); title, year and first author match. Non-arXiv URLs (Settles PDF, Lu blog, TRL, Math-Verify, HF dataset, Qwen3.5 model cards) all return HTTP 200.

| Citation | Claim in draft | Abstract supports? | Action |
|---|---|---|---|
| Agarwal et al. 2024 (2306.13649) | student trained on self-generated sequences with teacher feedback | yes (GKD) | none |
| Gu et al. 2024 (2306.08543) | reverse-KL distillation in MiniLLM | yes | none |
| Lu 2025 (blog) | recipe pairing student's own samples with dense per-token teacher supervision | yes ("sample trajectories from the student ... grade each token"; per-token advantage = negative reverse KL) | none |
| Song 2026 (2604.00626) | survey of the area | yes. Two authors | renamed "Song and Zheng (2026)" |
| Li 2026a (2604.13016) | studies OPD mechanism | yes | none |
| Fu 2026 (2609.04172) | single training examples | yes (one-shot OPD, single query) | none. Note: this is a data study, so "these papers mostly change what the teacher provides or how the loss is formed" fits it least; "mostly" keeps the sentence defensible |
| Zhang 2026 (2605.09725) | teacher rollout selection | yes (BRTS, best-of-N teacher rollouts) | none |
| Ding 2026 (2608.31046) | noisy teachers | yes (teacher-supervision noise). Two authors | renamed "Ding and Zhang, 2026" |
| Ma 2026 (2606.30406) | multiple teachers | yes (MOPD) | none |
| Heo 2026 (2607.15161) | delta distillation | yes | none |
| Yang 2026b (2606.06021) | representations | yes (OPRD, hidden-state space) | none |
| Hou 2026 (2609.05198) | "hard examples help most, even ones beyond the teacher; small set matches much larger set" | partly. The abstract says harder examples "often yield superior performance gain", "unsolvable" examples "can be successfully used", and 8 hard examples match the 17K baseline. "Help most" is stronger than "often" | softened to the abstract's wording |
| Li 2026b (2608.16647) | difficulty barely matters; never-solved problems still useful | yes (near verbatim) | none |
| Yang 2026a (2610.18011) | training only on failed problems lowers accuracy in every configuration | yes ("All eight configurations show a decline") | none |
| Wang 2026 (2609.37377) | targeted selection does not consistently beat uniform random | yes (verbatim) | none |
| Zhu 2026 (2609.22109) | shared LR not neutral control for selective OPD | yes. Note: Zhu's selectors pick token positions, not rollouts or questions. The Limitations wording ("may rank differently") is suitably hedged. arXiv lists the published date as 2026-08-17 despite the 2609 ID; the ID resolves to the right paper | none |
| Settles 2009 | active learning, often by uncertainty | yes (standard survey; PDF live) | none |
| Kung 2023 (2311.00288) | task selection via prompt uncertainty | yes | none |
| Mindermann 2022 (2206.07137) | learnable, worth learning, not yet learnt | yes (verbatim title) | none |
| Xia 2024 (2410.09335) | "at scale, random selection is hard to beat" | yes, but the paper's scope is SFT on million-scale pools | added "for supervised fine-tuning on large data pools" |
| Guha 2025 (2506.04178) | training set derived from OpenThoughts | the paper is the OpenThoughts project. The actual set is HF siyanzhao/Openthoughts_math_30k_opsd | dataset URL added to the Sources entry |
| Hendrycks 2021 / Lightman 2023 | MATH / MATH-500 split | yes | none |
| Kwon 2023 (2309.06180) | vLLM | yes | none |
| von Werra et al., TRL | GKD trainer | yes (TRL repo; GKD trainer docs page live) | none |
| Hou 2026 + Li 2026b in the Conclusion | "Our results also disagree with reports that questions beyond the teacher are still useful" | overclaim. Those papers report usefulness in absolute terms. Here both-wrong training still lifts the student from 26.87 to 41.32; it is only worse than random selection | softened to "in tension with" plus the absolute-vs-relative note |

Sources section: filled with 25 numbered entries with direct URLs (24 author-year citations + TRL), plus model/tool links (Qwen3.5 cards, Math-Verify) that the text names without a citation. No orphans: every author-year citation in the body has an entry and every entry is cited. The draft uses author-year in-text style, so the entries are listed alphabetically and carry the author-year label rather than [N] markers.

## Overclaims, excluded topics, dashes

- "proves", "significant", "definitive", "confirmed": none in the draft.
- Excluded topics (truncation/finishing, bootstrap intervals, near-miss, diversity, prefix correction, self-training, budget sweep, SFT baseline, 4-answer entropy arm): none present. "Budget" appears only as the 8,192-token budget and the training budget, not a sweep. The Xia edit says "supervised fine-tuning" only to scope that paper, not as an SFT baseline.
- Em/en dashes: none before or after the edits.
- Unqualified seed-1 results: the "KL is not correctness" heading generalised a seed-1 analysis, so I added "(seed 1)". The other seed-1 statements were already labeled.

## Edits made to papers/active-on-policy-distillation.md

1. Intro finding 1 heading: "so selection at the question level matters" → "and selection at the question level is what helped" (a causal "so" from a variance share). "Across all 12 answer pools we scored" → "Across the 12 answer pools of the random rule (3 seeds by 4 refreshes)".
2. Intro finding 2 heading: "Questions both models fail reduce performance" → "do worse than random". The effect is relative to random, and the arm still improves well over base.
3. Related Work, Hou: softened to the abstract's wording (often larger gains; beyond-teacher examples can be used; 8 examples match 17K).
4. Related Work: "questions beyond both models hurt" → "did worse than random selection".
5. Related Work, Xia: scoped to supervised fine-tuning on large pools.
6. Related Work: "Song et al." → "Song and Zheng"; "Ding et al." → "Ding and Zhang" (two-author papers).
7. Section 4.2 (data MISMATCH): replaced "Every rule draws its questions from these 4,641, and questions are never reused." The new text says that the random/KL/entropy rules draw from the 4,641 with no reuse within or across seeds; that the correctness rules use the widened 25,000-question bank (7,317 teacher-labeled) in seeds 2 and 3; and that the mixed bucket includes student-mixed questions with no teacher label (561 in bank-8k), which make up 153 / 193 / 446 of the 480 mixed questions per seed.
8. Section 5.1 correctness: "This matches the proposal's case of filtering" → "This is consistent with ..., though we did not run the filter itself (Section 6)".
9. Section 5.2: "None of them moves away from random" → "moves far from random on average". "made no measurable difference" → "made little difference" (no test was run).
10. Section 5.3: "For every answer-level pool" → "For every answer pool of the random rule". Figure 3 caption: same correction, plus "refreshes numbered 0 to 3".
11. Section 5.3 overlap: "(seed 3, refresh 3)" → "(seed 3, the last refresh)", which removes the 0-index ambiguity.
12. Section 5.3 heading: "KL is not correctness." → "KL is not correctness (seed 1)."
13. Limitation 4: added a sentence on the widened bank and the teacher-unlabeled mixed questions.
14. Conclusion: "Questions both models fail hurt" → "did worse than random". "while a bucket of failed questions hurts" → "does worse than random". "disagree with reports ..." → "in tension with reports ...", plus "Training on both-wrong questions still raised the student well above its base (41.32% against 26.87%), so the gap is relative to random selection ...".
15. Sources: filled (25 entries + model/tool links).

No numeric values in the tables were changed.

## Residual risks

- **Rounding ties** (1.2/1.3): the draft and Figure 2 round exact .xx5 values inconsistently (kl_low and ent_top4 both have exact -0.125 but print -0.12 and -0.13). Fixing this needs the tables and the figure regenerated together under one rounding rule. Not edited.
- **Mixed-bucket interpretation**: in seed 3, 446 of 480 "mixed" questions have no teacher label (student mixed only). The mixed arm's +0.27 therefore does not measure the "teacher mixed or disagreeing" case the label definition suggests. The draft now discloses this, but the interpretation in Table 1 and Section 5.1 ("Mixed questions help a little") could be revisited by the author.
- **Correctness seeds 2 and 3 use a different (widened) question pool** than the random arm they are compared against (bank-8k labeled, offsets 0/1440/2880). The gap between the correctness rules and random therefore mixes the selection rule with a pool difference. This affects the +1.22 headline (t_right_s_wrong). Now disclosed in 4.2 and Limitation 4; not quantified.
- Variance shares come from the random arm's pools only. Other arms' pools after refresh 0 were not analysed.
- "Three training seeds" also change the training questions (question_offset); described but not reworded.
- The second-index (OpenAlex) cross-check was done for 5 of the 23 arXiv papers. The rest rely on the arXiv API record plus the references note's earlier verification.
- Section HTML comments (`<!-- src: ... -->`) still cite `experiments/p2b-analysis/*.txt`. Those files agree with my raw recomputation, but they are not themselves raw data.
