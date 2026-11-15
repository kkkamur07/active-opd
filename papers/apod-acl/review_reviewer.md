# Area-chair review: `papers/apod-acl/main.tex` ("[Title to be decided]")

Reviewed: `main.tex` (636 lines) and `main.pdf` (22 pages, compiled 2026-11-12 11:08). The main text ends on page 8, and Limitations starts on page 8, so the paper fits the 8-page limit. `main.tex` was not edited.
Cross-checked against: `papers/apod-story.md` (v3), `experiments/apod-data-scan/results_tables.md`, the `critic_checks.py` output (appendix of `notes/apod-critique-results.md`), `docs/decisions.md`, `docs/perf_review.md`, `figures/README.md`, and the raw shards under `outputs/runs_results/`. Section "Reviewer-computed checks" lists the extra numbers I computed, with the exact commands.

---

# Part 1: Structured review

## Summary

The paper studies **which rollouts to train on** during teacher-guided on-policy post-training. In this setup a Qwen3.5-2B student samples its own solutions to competition-math questions, and the student is trained to minimise the token-level reverse KL to a frozen Qwen3.5-9B teacher on those rollouts (TRL `GKDTrainer`, β=1, λ=0). For each question the method samples 12 rollouts and ranks them by exact mean per-token reverse KL. It then trains on ranks 5–8, the "aligned band". The paper motivates this with a "distributional alignment" hypothesis: the lowest-KL rollouts are redundant, the highest-KL rollouts are too divergent to absorb, and the middle is the most learnable. The comparison arms are divergent (ranks 1–4), redundant (9–12), uniform-k (4 at random) and full (all 12). The evaluation is MATH-500 strict avg@4 at the training cap (8k or 16k tokens).

The headline numbers come from one seed of MidAlign-8k at step 34: aligned 0.4745 against uniform-k 0.3880 (+8.65 [+6.55, +10.80]), with the four arms forming an inverted U. At 16k, round 1, the aligned-minus-uniform contrast is +3.75 and +3.70 in two seeds. The paper deserves credit for how much evidence against its own headline it puts in the main text:
- aligned trails uniform at step 17 (−3.65) and in the 8k pilot (−2.40);
- the effect is null at step 51 (+0.45, warmup-variant retrain) and in the 100-step low-LR regime (+0.05);
- seed-to-seed noise (±6.7 points) is 4–9× the bootstrap CI;
- across all ten high-LR contrasts the mean is +1.21 and the median +0.48.

The paper also shows that essentially the whole gain is a finish-rate gain (accuracy given finished is about 0.99 throughout). It shows that student entropy correlates with reverse KL (ρ≈0.73) but that entropy-top-4 selects the divergent third. Finally, it shows that only 14–24% of KL variance lies within a question.

## Strengths

- **[S1] Unusually honest noise accounting.** The paper separates evaluation noise (paired bootstrap, median half-width 1.66) from training noise (seed-replicate SD 3.40/7.68) and checks that runs are deterministic given the data (same-data retrain at +0.10). This is better than most selection papers. Table `tab:noise` (App. B) is a useful contribution in its own right.
- **[S2] The termination decomposition is exact and reported throughout.** Eq. 5 is an exact two-point product-rule split (I verified the algebra). Reporting finish rate and acc|fin next to every headline number (Table 2 caption; §6.1) stops readers from mistaking "finishes within 8k" for "reasons better".
- **[S3] Nulls and failed pre-specifications are reported.** The entropy arm misses its pre-specified 3-point MDE ("below the 3-point effect the run was planned to detect"). The MidAlign-long null, the question-level metric dependence, and the pilot's 512/544 row overlap are all in the main text. The origin of the hypothesis (16k seed A round 1) is disclosed.
- **[S4] Exact full-vocabulary KL scoring with estimator validation.** Table `tab:mc` shows that the free generated-token estimator is biased by −20% and that the middle third is the hardest to estimate (0.540 agreement at n=1). This is practical guidance for anyone who wants to make the method cheap.
- **[S5] A useful mechanistic negative result about entropy.** "Entropy correlates with KL, so entropy-*top* picks the divergent third" (overlap with KL-top 2.54–2.72/4 vs. 0.95–1.11/4 with the band, chance 1.33). This explains the entropy-arm null in a way other work can reuse.
- **[S6] Contamination and data audit beyond the usual 13-gram scan.** The audit adds an 8-gram scan and a trained-on check (App. D).
- **[S7] Reproducibility infrastructure.** Every number carries a source comment, and `make_figures.py` asserts every plotted number against the reference tables (`figures/README.md`, "check … OK" lines).

## Weaknesses

- **[W1] FATAL for the claim as currently worded (fixable by reframing plus seeds): the headline effect is one seed's best checkpoint, and the aggregate evidence does not support "the aligned band wins".**
  - Abstract, Contribution 2 and Fig. 1 lead with +8.65 at step 34 of a single seed. The same run gives −3.65 at step 17. Its planned final evaluation (step 51) "never ran" (`docs/decisions.md:60`); a step-51 number exists only for a warmup retrain (+0.45). So the headline is the best of three checkpoints of the first test, and the first evaluation point of that test went the wrong way.
  - The paper's own aggregate (§6.2 "Noise") is mean +1.21, median +0.48, 6/10 positive. Treating the ten contrasts as independent (they are not, which makes it weaker), I get SD 4.39, SE 1.39, t ≈ 0.87; the sign test 6/10 is uninformative.
  - `critic_checks` §H reports 102 arm-vs-control tests, 38 at p<0.05 of which only 14 are positive. The paper does not mention this multiplicity.
  - The +3.75/+3.70 agreement at 16k round 1 is the strongest evidence. But it is two seeds at one round, and the same two seeds disagree by 4.95–5.80 at rounds 2–3.
  - As written, Abstract sentences 4–5 and Contribution 2 claim more than the evidence supports. The paper's own Discussion ("Our results support a limited claim") is the right level, but the front matter does not match it.
- **[W2] MAJOR: "equal budget" hides a 1.6–3× compute difference, and the full baseline is not a fair efficiency comparator.**
  - The aligned band needs teacher+student forward passes over all 12 rollouts for scoring, plus training on 4. Per `docs/perf_review.md`, a KL round-arm costs about 83 min against about 55 min for uniform-k: scoring (27.5 min) is longer than training (25.4 min).
  - Against full, the aligned band saves only the optimizer steps. My estimate from the same table, excluding eval: about 68 min aligned vs about 91 min full. That is roughly 0.75× full, not one third.
  - At round 1, full is itself −3.05 below uniform-k. A baseline that underperforms uniform-k does not establish efficiency: 48 steps at a constant 2e-5 with Adam reset is plausibly over-training, at an LR that the authors' own LR search deems too high (App. B).
  - Neither a step-matched full baseline (16 steps at 3× batch) nor a teacher-pass-matched uniform baseline (uniform-k over 3× the questions) exists or is discussed.
  - Contribution 1 even lists "full" among the baselines "at equal budget".
- **[W3] MAJOR: the hypothesis's mechanism is untested, and the paper's own artifacts point against it.**
  - The hypothesis is about *learnability of the correction*: training on the band should move the student toward the teacher most effectively. The paper never measures this. The data exist: each KL arm's round-1/round-2 pool is re-scored with exact KL.
  - My check (below) shows the opposite pattern. After round 1, the aligned arm's own rollouts are the *farthest* from the teacher of the three KL arms in all three runs:
    - MidAlign-8k: 0.1202 vs 0.1168 divergent and 0.1154 redundant (round-0 0.1205);
    - 16k A: 0.1058 vs 0.1029 / 0.0973;
    - 16k B: 0.1001 vs 0.0996 / 0.0987.
  - In 16k these are the very checkpoints where the aligned arm has the best accuracy. This suggests that accuracy and distance-to-teacher decouple, and that the gain, if real, is not "distributional alignment" in the sense of Eq. 1.
  - Caveats: mean per-token KL is confounded with length, truncation and degenerate repetition, and I could not verify that the round-1 pools reuse the same questions. Still, the paper should either report this test and discuss it, or stop describing the mechanism as alignment.
  - §6.3 already notes that "The aligned band has the same mean KL as uniform selection (+0.2%)". The method is therefore close to "uniform minus the tails", and the paper should test that directly (W6).
- **[W4] MAJOR: the termination analysis is close to an accounting identity, but it is presented as a mechanism.**
  - With acc|fin ≥0.979 everywhere and acc|trunc around 0.1, any change in strict accuracy must load on Δf. So "+9.37 from finish rate" cannot distinguish (a) learning to stop from (b) solving more problems, faster.
  - Despite this, the abstract says the gain "operates mainly by teaching the model to finish" (causal), and §6.1 says "corrections teach the student to reach an answer and stop".
  - The training data contradicts a simple "learn to finish" story. The aligned band's selected rollouts were 88.0% truncated (round 1 of MidAlign-8k: 92.5%, vs 84.4% for uniform-k; reviewer check), so the band rarely sees an EOS target.
  - The paper should present the decomposition as *describing where* the difference appears, not as evidence of *how* it arises.
  - Per-level analysis (MATH-500 levels are recoverable from `id`) and the length distributions of finished answers are available offline and would sharpen the story.
- **[W5] MAJOR: the "budget/cap" explanation contradicts the headline.**
  - §6.2 "Token budget" concludes "the alignment signal needs rollouts that can finish", based on the 8k round-1 losses. But the headline result is itself at the 8k cap, where 88% of the band's training rollouts are truncated.
  - The supporting statistic is the *teacher's* finish rate ("even the teacher finishes only 55%"), which says nothing about the student's rollouts that are actually selected. The relevant numbers are student pool cap-hit: about 91% at 8k round 0 vs about 80% at 16k round 0 (reviewer check).
  - This paragraph needs to be reconciled with Table 2 or removed.
- **[W6] MAJOR: missing baselines and ablations, which should at least be discussed.**
  1. A *trimmed/stratified uniform* baseline: 4 random from ranks 3–10, or one per third. This separates "middle" from "not the extremes". The fact that uniform-k beats every third at step 17 suggests that diversity across thirds itself matters.
  2. Step-matched and teacher-pass-matched full and uniform baselines (W2).
  3. Off-policy SFT on teacher traces and an RL baseline (GRPO) as reference points for what on-policy post-training buys at this budget.
  4. Length-aware baselines, e.g. keep the shortest finished rollouts, or a length-penalised loss. If the gain is finishing (W4), a direct length intervention is the natural competitor.
  5. Wang et al. (2026), whose reverse-KL prompt selector did *not* beat random (36.94 vs 37.65). They are the closest negative result and deserve a direct "why ours differs" paragraph: rollout vs prompt level, middle vs top.
- **[W7] MAJOR: the seed yardstick is thin and favourably chosen.**
  - ±6.7 = 1.96×3.40 comes from n=6 seed-pair differences of a contrast at 16k (`critic_checks` §B), after excluding the redundant arm. Including it gives 7.68, i.e. ±15. On that band +8.65 is within noise.
  - The band is estimated in a different regime (16k, rounds with length collapse) and transferred to 8k.
  - Fig. 1 draws it around the uniform-k *level*, although it is a band for a *contrast*.
  - Either justify the exclusion before looking at the result, or show both bands, or use the conservative one.
- **[W8] MAJOR: the pre-specification and chronology are unclear, which matters for a paper that discloses a post-hoc hypothesis.**
  - Was the 8k pilot (oracle8k, −2.40) run before or after the hypothesis was formed after 16k seed A round 1?
  - MidAlign-8k was planned to 51 steps; why was the step-51 evaluation never run, and why is step 34 the primary endpoint?
  - MidAlign-long (r2) was designed after kl50 as the trajectory-selection follow-up (`docs/decisions.md:41–44`, with `persist_optimizer` added to fix kl50's restart spikes). If it was the intended confirmatory test, its null deserves equal weight in the abstract, not only a "scope" framing.
- **[W9] MAJOR: the external validity is narrow.**
  - One student–teacher pair, one training distribution, one benchmark (MATH-500), on which strict accuracy at 8k mostly measures finishing.
  - AIME carries no signal at 8k (App. E), and there is no 32k evaluation.
  - ACL readers will ask why thinking mode is evaluated at a cap where the base model truncates 77.7% of samples and the teacher 44.9%.
  - At minimum, motivate the cap (cost? matching the training cap?) and say clearly that "accuracy" here means "solved within 8k".
- **[W10] MINOR→MAJOR: related-work gaps on the paper's core idea.**
  - The hypothesis, rollouts "close enough … to be learnable but far enough to be worth learning", closely echoes **RHO-LOSS** (Mindermann et al., ICML 2022, *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt*). That paper is not cited.
  - Also missing:
    - reference-model-scored token selection, **Rho-1** (Lin et al., 2024);
    - the **small-model learnability gap** in reasoning distillation (Li et al., 2025);
    - the teacher–student capacity gap (**Teacher Assistant KD**, Mirzadeh et al., 2020);
    - **Speculative KD** (Xu et al., 2024), which tackles the same "student rollouts the teacher can't correct well" problem by intervening on tokens;
    - classic self-paced learning and online batch selection.
  - The claim "These works train on all rollouts" is too broad, because Zhang et al. (2026) already question full rollouts.
- **[W11] MINOR: the "divergent extreme at question level" is an analogy, not a measurement.** Question buckets are defined by correctness labels and H(q), not by reverse KL; no question-level KL is computed. The control comes from a different run. Soften §5.3's conclusion ("questions inside the student's reach matter, as rollouts do").
- **[W12] MINOR: presentation.**
  - Repo names leak into the main text ("apod: entropy-top-4 …", "r3's random-question arm", "r1 / r3 / r4" in Table 1). The story renames everything else.
  - Figure captions carry file paths and script names ("Data: `outputs/runs_results/kl50` … script `make_figures.py`"), which is unusual in ACL and wastes space.
  - Figures are numbered out of discussion order (Fig. 4 is cited in §6.1 before Fig. 3 in §6.2).
  - Table 1 (`table*`) costs about a third of a page for information an appendix can carry.
  - The Discussion is a bulleted "planned, not run" list, which reads as a to-do list.
  - The Ethics statement discusses figure-style licensing but omits GPU type and GPU-hours (A100 per `docs/perf_review.md`).
  - "Distributional alignment" and "aligned band" collide with the RLHF sense of "alignment" in an ACL venue, and the name presupposes the conclusion.
- **[W13] MINOR: release integrity.** The paper promises "configurations". But `outputs/runs_results/kl50/resolved_config.yaml` and `oracle16k/resolved_config.yaml` list `arms: [entropy_top4, random_top4, all]`, `rounds: 8` and `run_name: apod`, which do not describe those runs (they were driven by `scripts/bucket_experiment.py`). Fix this before release, or reviewers who open the artifact will distrust it.
- **[W14] MINOR: the bf16 explanation is stated more strongly in §6.2 than App. B supports.** §6.2: "so the late steps barely move the weights". App. B: "rounding alone does not explain why gains stop at step 10". Carry the appendix's counter-evidence into the main text.

## Questions for Authors

- **[Q1]** What was the pre-specified primary endpoint of MidAlign-8k (step 51?), and why did the step-51 evaluation of the original run never happen? Is the original round-3 checkpoint recoverable?
- **[Q2]** Chronology: was oracle8k (pilot, −2.40) run before or after the hypothesis was formed? Was MidAlign-long designed as the confirmatory replication of MidAlign-8k?
- **[Q3]** Both seed configs have `pool_seed: 42`. Did seed B actually draw different questions, as "different seed and question draw" states? If not, the seed yardstick measures sampling and training noise only.
- **[Q4]** Why exclude the redundant arm from the seed SD (3.40 vs 7.68)? Was that decision made before seeing the MidAlign-8k result?
- **[Q5]** After a round of training, which arm's student is closest to the teacher by mean reverse KL on fresh rollouts? (My check says the aligned arm is the farthest of the three KL arms in all three runs; see below.) How does this square with "learnable corrections"?
- **[Q6]** What is the wall-clock / FLOP cost per round of aligned vs uniform-k vs full, including scoring? Can scoring logits be cached and reused in the training forward?
- **[Q7]** What do the lowest-KL rollouts look like qualitatively? Low per-token KL at 91% truncation is consistent with degenerate repetition loops, which would make "redundant" a mislabel; the 16k seed-A redundant arm collapses to cap-hit 0.857.
- **[Q8]** Does the step-34 gain concentrate on easy MATH-500 levels (where finishing is the binding constraint) or on Levels 4–5?
- **[Q9]** Why evaluate thinking mode at 8k, where the base model truncates 77.7% of samples? Was a larger cap infeasible, or chosen to match training?

## Verdict

This is a carefully instrumented empirical study with an intuitive hypothesis and an evidence base that, by the paper's own accounting, supports only an early, regime-specific, seed-fragile effect. The strongest evidence is the two-seed agreement at 16k round 1 (+3.75/+3.70). The headline +8.65 is a single seed's best checkpoint, flanked by −3.65 and +0.45. The pooled evidence (mean +1.21, median +0.48 over ten contrasts) is consistent with a small or zero average effect. The mechanism section describes where the gain appears (finishing) but not why. The paper's own re-scored KL pools do not show the aligned arm moving closer to the teacher.

The writing is commendably candid in §6 and the Limitations. The main revision risk is the gap between the confident front matter (abstract, contributions, Fig. 1) and the scoped conclusion in §7. A skeptical reader will notice that the second half of the paper takes apart the first.

**ARR-style scores (my judgement; not a prediction of the venue outcome):**

| criterion | score |
|---|---|
| Soundness | 2.5 / 5 (claims as worded outrun the evidence; analyses themselves are careful) |
| Excitement | 2 / 5 (intuitive idea with close prior art, RHO-LOSS / learnability curricula; small, fragile effect) |
| Overall assessment | 2.5 / 5 (borderline) |
| Reproducibility | 4 / 5 (per-sample shards, asserting scripts; config files need fixing, W13) |
| Reviewer confidence | 4 / 5 (I checked the numbers against the tables and raw shards; I did not re-run training) |

What would move the score:
- reframing plus the no-experiment analyses below, about +0.5;
- adding ≥2 MidAlign-8k seeds that keep a positive aligned-minus-uniform contrast at a pre-specified endpoint, plus one fair compute-matched baseline, about +1 more.

If the new seeds come back null, the paper is still publishable as a careful negative/scoping study, but it would need the framing the story currently avoids.

## Revision Plan

### A. Edits that need no new GPU runs (ordered by expected impact)

1. **Make the front matter match §7 (W1, W2, W4).**
   - Abstract: say "at one checkpoint of one seed" for 47.5 vs 38.8 and "at round 1" for the 16k result.
   - Replace "exceeds training on all rollouts with a third of the optimizer steps" with "at round 1, matched or exceeded full training with a third of the optimizer steps (but the same number of teacher passes); full overtakes it by round 2".
   - Replace "operates mainly by teaching the model to finish" with "appears almost entirely as a higher finish rate within the token budget".
   - Add the pooled number: "across ten contrasts the mean advantage is +1.2 points".
   - Contribution 1: remove "full" from "at equal budget".
   - Contribution 2: add "at round 1" and the seed band.
2. **Add one forest plot** (it can replace Fig. 1 or its right half) of all aligned-minus-uniform contrasts: 10 high-LR plus MidAlign-long pooled, each with its bootstrap CI and the seed band, and the mean and median marked. Also add the **within-thirds test**, which is the direct test of the hypothesis and does not depend on uniform-k's dip: the aligned band is the best of the three KL thirds at 7 of 10 high-LR checkpoints (8k s17/s34/s51; 16k A r1/r2/r3; 16k B r1; not 16k B r2/r3 or the pilot). Under a 1/3 null with independence that is p≈0.02 (binomial), and it is weaker under the real dependence. Present it as post hoc. This is the most hypothesis-relevant evidence the paper has, and it is currently not reported.
3. **Add a compute table (W2)** giving, per arm and round: rollouts generated, teacher-scored positions, trained tokens, optimizer steps and wall-clock. Use `docs/perf_review.md` (rollouts 13.6 min, scoring 27.5, training 25.4, eval 15.2). Rename "equal budget" to "equal training budget" everywhere.
4. **Report the mechanism check from existing artifacts (W3):** mean reverse KL and truncation of each KL arm's re-scored pool at rounds 1–2, from `arms/*/rounds/round_0{1,2}/oracle`, and the per-arm training-loss trajectories from `train/log_history.jsonl`. Then either explain the decoupling of KL and accuracy, or drop "alignment" as the claimed mechanism.
5. **Reconcile §6.2 "Token budget" with the 8k headline (W5).** Use the student's selected-rollout truncation (88% aligned vs 87.4% uniform; 92.5% vs 84.4% in round 1), not the teacher's finish rate. Either remove the "needs rollouts that can finish" reading or restate it as conjecture.
6. **Disclose the chronology and endpoints (W8):** the pilot's timing, the planned step-51 endpoint and why it is missing, and the design intent of MidAlign-long.
7. **Justify or replace the seed band (W7).** State n=6, show both 3.40 and 7.68, and draw the band on contrast plots only (Fig. 3/forest), not around the uniform-k level in Fig. 1. Report the multiplicity (102 tests; 38 at p<0.05, 14 positive).
8. **Add offline analyses that sharpen §6.1 (W4):**
   - the gain by MATH-500 level (join shard `id` with the MATH-500 level field);
   - the length distributions of finished answers per arm (`response_length` in the shards);
   - the share of the aligned-vs-uniform difference on problems that uniform-k finishes but gets wrong (acc|fin is about 0.99, so this should be near zero; state it).
9. **Related work (W10):**
   - add RHO-LOSS, Rho-1, the small-model learnability gap, Teacher-Assistant KD and Speculative KD;
   - write one explicit paragraph contrasting with Wang et al. (2026)'s negative reverse-KL prompt-selection result;
   - narrow "These works train on all rollouts";
   - cut the long one-line list of 2026 OPD papers to the 6–8 that bear on selection.
10. **Presentation (W11–W14):**
    - rename "apod" (e.g. "Entropy-8k") and "r1/r3/r4" in the main text;
    - strip file paths and script names from captions;
    - move Table 1 to the appendix, with a one-line regime summary in §4;
    - fix the figure order;
    - fold the "planned, not run" bullets into a 3-line future-work sentence;
    - add GPU type and GPU-hours to the Ethics/Reproducibility statement;
    - bring App. B's bf16 counter-evidence into §6.2;
    - soften the question-level "divergent extreme" analogy;
    - consider a less loaded name than "alignment" (e.g. "mid-KL band"), or define it in the first paragraph as distinct from preference alignment.
11. **Fix the released configs (W13)** so that each run's `resolved_config.yaml` records its actual arms, k, rank windows and rounds.

The space freed by moving Table 1 and condensing the Discussion is about 0.5 page, enough for the forest plot and the compute table.

### B. Experiments that would most raise the score (ordered)

| # | Experiment | What it settles | Cost note |
|---|---|---|---|
| 1 | **MidAlign-8k, aligned band and uniform-k, ≥2 new seeds (better 3), fresh question draws, pre-registered endpoint (step 51) and analysis (pooled over steps 17/34/51)** | Whether +8.65 is the method or noise; addresses W1, W7, W8 | ≈83 min per aligned round-arm, ≈55 min uniform (`docs/perf_review.md`); story §7 estimates ≈9 GPU-h for 2 seeds |
| 2 | **Fair efficiency baselines:** full at 16 steps with 3× batch (step-matched); uniform-k over 3× questions (teacher-pass-matched); optionally full at LR/3 | Whether the "a third of the steps" claim survives a tuned full baseline; W2 | one arm each, 16k round 1 only |
| 3 | **32k re-evaluation of surviving checkpoints** (8k s17/s34; 16k r1) | Finishing vs. solving; W4, W9 | eval only |
| 4 | **Trimmed/stratified uniform arm** (4 random from ranks 3–10, or one per third plus one random) | Whether "middle" matters or just "not the tails"; W3, W6 | one arm |
| 5 | **Entropy-mid arm** | Teacher-free version of the method; removes the 1.6× scoring overhead | one arm |
| 6 | **Second benchmark at a non-binding cap** (e.g. OlympiadBench or AIME at 32k) and, if feasible, a second student–teacher pair | External validity; W9 | eval plus one pair |
| 7 | **MidAlign-long with fp32 master weights** | Whether the long-regime null is numerics; W14 | unmeasured |
| 8 | **Between-question KL-band arm** | Tests the hypothesis where 76–86% of the variance lies | one arm |
| 9 | Reference points: SFT on teacher traces and GRPO at a matched budget | Positioning for ACL readers | larger |

---

# Part 2: Inline Annotations

> "In one run, the aligned (middle) third raises MATH-500 accuracy from 27.9\% to 47.5\%, against 38.8\% for uniform selection; such a contrast varies by about $\pm$6.7 points across seeds."

**[W1] FATAL (wording) / [W7] MAJOR:** Good that "in one run" and the band are present. Still missing: (a) this is one checkpoint (step 34) of three, flanked by −3.65 and +0.45; (b) the ±6.7 band excludes the redundant arm, and with it the band is ±15. Add "at one checkpoint", and add the pooled mean (+1.2 over ten contrasts).

> "At a 16k-token budget its advantage agrees across two seeds (+3.7 points) and exceeds training on all rollouts with a third of the optimizer steps."

**[W2] MAJOR:** This needs "at round 1". It violates story integrity condition 3 ("Efficiency is claimed only early"). It is one seed for the full comparison. Full at round 1 is itself −3.05 below uniform-k. Teacher passes are equal or higher for the aligned band. Suggested: "…at round 1 agrees across two seeds (+3.7) and, in one seed, exceeds full training with a third of the optimizer steps; full overtakes it by round 2."

> "The gain operates mainly by teaching the model to finish within its token budget, appears early in training, and fades under long low-learning-rate training."

**[W4] MAJOR:** "teaching the model to finish" is a causal mechanism claim. The decomposition is an accounting identity in a regime where acc|fin≈0.99. Rephrase as "appears almost entirely as a higher finish rate".

> "tested against divergent, redundant, uniform and full baselines at equal budget."

**[W2] MINOR:** Full is 3× the data and steps, and the aligned band uses about 1.6× the wall-clock of uniform-k because of scoring. Say "equal training budget", and list full separately.

> "beats full training at round 1 with one third of the optimizer steps."

**[W2]:** Add "in one seed" and "the same teacher-scored rollouts". The comparator underperforms uniform-k at this point, so it is weak evidence of efficiency.

> "At step 34 the aligned band reaches 0.4745, ahead of divergent (0.4320), uniform-k (0.3880) and redundant (0.3835); this is the inverted U the hypothesis predicts."

**[W1] MAJOR:** At step 34 the divergent third also beats uniform-k by +4.40. The "U" depends on uniform-k dipping from 0.4300 to 0.3880. The cleaner test of the hypothesis is the ranking *among thirds* (aligned best at 7 of 10 checkpoints). Report that, and call the uniform-k comparison secondary.

> "Pooled over steps 17 and 34, the aligned band is +2.50\ci{+1.15}{+3.85} points ahead."

**[Q1]:** This is the more defensible single number for MidAlign-8k; consider promoting it next to +8.65. State which endpoint was pre-specified.

> "Early in training, then, keeping a third of the rollouts and a third of the steps is enough."

**[W2] MAJOR:** "keeping a third of the rollouts" saves backward passes only; all 12 rollouts are still generated and teacher-scored. Add the compute table.

> "a per-run standard deviation of 3.40 points for an arm-minus-uniform contrast (7.68 if the redundant arm is included), that is, a 95\% band of about $\pm$6.7 points."

**[W7] / [Q3] / [Q4]:** This comes from n=6 differences at 16k, and the exclusion of the redundant arm is favourable. Both runs' configs show `pool_seed: 42`; confirm the "question draw" differs. Justify the exclusion or adopt the conservative band.

> "Grey band: uniform-k $\pm$6.7 points, the 95\% seed-to-seed spread of a single arm-minus-uniform contrast (1.96 $\times$ 3.4)."

**[W7] MINOR:** A contrast band drawn around an arm *level* is a category error that readers will misread. Move it to a contrast plot (Fig. 3 or a forest plot).

> "Our reading is that the alignment signal needs rollouts that can finish."

**[W5] MAJOR:** This contradicts the headline. MidAlign-8k step 34 is at 8k, and the aligned band's training rollouts were 88.0% truncated (92.5% in round 1, vs 84.4% for uniform-k). Remove the reading or reconcile it.

> "At 8k even the teacher finishes only 55\% of samples, against 77\% at 16k."

**[W5] MINOR:** The teacher's finish rate on MATH-500 is not the relevant quantity. Use the student's selected-rollout truncation (about 91% at 8k round 0 vs about 80% at 16k round 0).

> "Our reading is that the aligned band's corrections teach the student to reach an answer and stop within the budget."

**[W4] MAJOR:** If the band trains on 88% truncated rollouts with no EOS target, how does it teach stopping more than uniform-k (87.4%)? Offer this as one hypothesis among others (e.g. avoiding degenerate low-KL loops, or shorter reasoning), and test it with the length and per-level analyses (Revision A8).

> "The aligned band has the same mean KL as uniform selection; if it helps, it does so through which rollouts it picks, not through how much KL they carry."

**[W3] MAJOR:** This sentence undercuts the paper's own name for the method. If mean KL is the same as uniform, the operative difference is excluding both tails. Test that directly (trimmed-uniform arm), and report the post-round KL check (aligned arm farthest from the teacher after round 1 in all three runs).

> "Scoring needs a teacher pass over all 12 rollouts, so selection saves optimizer steps, not teacher passes."

**[S] / [W2]:** Good. But this disclosure sits in §3, while the abstract and contributions imply compute savings. Echo it in the abstract and quantify it (scoring 27.5 min vs training 25.4 min per round-arm).

> "With bf16 weights and no fp32 master copy, 97.0--99.8\% of Adam updates round away at this learning rate, so the late steps barely move the weights \citep{gao2026bf16,singh2026fp32}."

**[W14] MINOR:** App. B says refreshes 1–3 had similar or higher LR than refresh 0 "but added no gain, so rounding alone does not explain why gains stop at step 10". Carry that sentence here. Also note that 2–3% of updates do land, concentrated on small-magnitude weights.

> "MidAlign-long changes five things at once relative to MidAlign-8k (rate, schedule, Adam persistence, kernels and number of questions), so it is not a clean replication attempt."

**[W8] / [Q2]:** Fair, but `docs/decisions.md` describes this run as the planned trajectory-selection experiment, with Adam persistence added specifically to fix kl50's restart spikes. If it was the intended confirmatory run, say so.

> "These are the divergent extreme at question level, and 98.0\% of their training rollouts are truncated."

**[W11] MINOR:** No question-level KL was measured; "divergent" here is an analogy based on correctness labels. Say "the hardest extreme (neither model solves them)".

> "The reading we draw is narrow: questions inside the student's reach matter, as rollouts do, and the clearest harm comes from the divergent extreme."

**[W11]:** "as rollouts do" assumes the rollout-level result that §6 qualifies heavily. Drop the clause.

> "These works train on all rollouts; we ask which rollouts to keep."

**[W10] MINOR:** Too broad; Zhang et al. (2026) ("Are full rollouts necessary…") is in this very list. Narrow it to "most of these works".

> "The idea mirrors ``zone of proximal development'' curricula in reinforcement learning \citep{tzannetos2023proximal,rutherford2024noregrets,foster2025learnability}, but it acts on rollouts, not on tasks."

**[W10] MAJOR (positioning):** The closest conceptual precedent is RHO-LOSS (Mindermann et al., 2022: "points that are learnable, worth learning, and not yet learnt"), and Rho-1 (Lin et al., 2024) for reference-model-scored token selection. Cite both, and say what differs: on-policy, teacher-referenced, within-question.

> "\citet{wang2026beyondprompt} report that targeted prompt selection does not consistently beat uniform random sampling; their reverse-KL-based selector did not beat random (36.94 vs.\ 37.65 $\pm$ 0.38)."

**[W6]:** This is the closest prior negative result. Add 1–2 sentences explaining why rollout-level, middle-band selection should behave differently, and note that your own pooled evidence (+1.2 mean) is not far from theirs.

> "\paragraph{apod: entropy-top-4 vs.\ uniform-k vs.\ full.}"

**[W12] MINOR:** "apod" is a repo name, and every other run was renamed. Use a paper name.

> "Data: \texttt{outputs/runs\_results/kl50}, \texttt{kl50\_teacher}; script \texttt{make\_figures.py}."

**[W12] MINOR:** File paths and script names in captions are non-standard for ACL and cost lines. Move them to the appendix or the README.

> "All of the following are planned, not run:"

**[W12] MINOR:** A six-bullet to-do list in the Discussion invites "come back when done". Condense it to one sentence of future work, and keep the scope statement.

> "We release the code, configurations, per-sample evaluation shards, reverse-KL scores and training logs of every run in this paper"

**[W13] MINOR:** The current `resolved_config.yaml` for kl50 and oracle16k records `arms: [entropy_top4, random_top4, all]`, `rounds: 8` and `run_name: apod`, which do not describe those runs. Fix this before release.

> "Figures are produced by one script from the raw shards and follow the figures4papers style (CC BY-NC 4.0)."

**[W12] MINOR:** The figure-style licence does not belong in the Ethics statement; put it in the acknowledgements. Add the GPU type (A100) and total GPU-hours, as the Responsible NLP checklist asks.

> "We state its origin plainly: the hypothesis was formed after the first round of our 16k-token experiment (MidAlign-16k, seed A)"

**[S3] / [Q2]:** Commendable. Add whether the 8k pilot preceded or followed this, and the planned endpoint of MidAlign-8k.

---

# Reviewer-computed checks (raw artifacts; no training re-run)

1. **Ten aligned-minus-uniform contrasts** (pilot r1, 8k s17/s34/s51, 16k A r1–r3, 16k B r1–r3), from `results_tables.md` and CC §A:
   - values: −2.40, −3.65, +8.65, +0.45, +3.75, −0.15, +6.30, +3.70, −5.10, +0.50;
   - mean +1.205, SD 4.39, SE 1.39, t = 0.87 (assumes independence, which does not hold);
   - this matches the paper's +1.21 / median +0.48.
2. **Aligned band best among the three KL thirds** at 7/10 of those checkpoints (from `results_tables.md`, CC §A). Binomial P(X≥7 | n=10, p=1/3) ≈ 0.020 under independence. This is a post-hoc analysis.
3. **Selected training rollouts** (`arms/*/rounds/round_0*/selected/selected.jsonl`, de-duplicated by `(example_index, rollout_index)`). Fraction correct / truncated per round:

   | run | arm | round 0 | round 1 | round 2 |
   |---|---|---|---|---|
   | kl50 | aligned | .121 / .914 | .180 / .925 | .206 / .801 |
   | kl50 | uniform-k | .110 / .915 | .211 / .844 | .151 / .862 |
   | kl50 | divergent | .132 / .908 | .145 / .947 | .186 / .868 |
   | kl50 | redundant | .118 / .917 | .145 / .868 | .151 / .871 |

   Mean truncation over rounds: aligned 0.880, uniform-k 0.874, which matches the paper.
4. **Re-scored pool mean reverse KL after training** (`arms/{kl_high,kl_mid,kl_low}/rounds/round_0{1,2}/oracle/oracle_kl.shard*.jsonl`, de-duplicated; uniform-k and full pools are not scored). Values are round-0 → round-1 per arm:

   | run | round 0 (shared pool) | divergent r1 | aligned r1 | redundant r1 |
   |---|---|---|---|---|
   | kl50 | 0.1205 | 0.1168 | **0.1202** | 0.1154 |
   | 16k A | 0.1071 | 0.1029 | **0.1058** | 0.0973 |
   | 16k B | 0.1076 | 0.0996 | **0.1001** | 0.0987 |

   In every run the aligned arm is the farthest of the three from the teacher after round 1. Caveats: per-token KL is confounded with length and truncation, and I did not verify that the questions are identical across rounds.
5. **Compute** (`docs/perf_review.md`, kl50 KL round-arm): rollouts 13.6 min, scoring 27.5 min, training 25.4 min (+~1), eval 15.2 min, total ~83 min; uniform-k ~55 min. My estimate excluding eval: aligned ≈68 min, uniform-k ≈41 min, full ≈13.6 + 3×25.4 ≈ 90 min (no scoring). This is an estimate, not measured.
6. **Planned endpoint:** `docs/decisions.md:60`: "kl50 headline is at 34 training steps (eval after two refreshes; the post-51-step eval never ran)".
7. **Config mismatch:** `outputs/runs_results/kl50/resolved_config.yaml` lines 54–62 contain `run_name: apod`, `arms: [entropy_top4, random_top4, all]`, `rounds: 8`. The same holds for `oracle16k`.

Commands (run from `outputs/runs_results/`):

```
python3 - <<'EOF'   # check 3
import json,glob
for run in ['kl50','oracle16k','oracle16k_seed2','oracle8k']:
  for arm in ['kl_high','kl_mid','kl_low','random']:
    for r in sorted(glob.glob(f"{run}/arms/{arm}/rounds/round_*/selected/selected.jsonl")):
      seen={(d['example_index'],d['rollout_index']):d for d in map(json.loads,open(r))}
      v=list(seen.values()); print(run,arm,r.split('/')[4],len(v),
        sum(bool(x.get('correct')) for x in v)/len(v), sum(x['truncated'] for x in v)/len(v))
EOF
python3 - <<'EOF'   # check 4
import json,glob
for run in ['kl50','oracle16k','oracle16k_seed2']:
  for arm in ['kl_high','kl_mid','kl_low']:
    for r in ['round_00','round_01','round_02']:
      fs=glob.glob(f"{run}/arms/{arm}/rounds/{r}/oracle/oracle_kl.shard*.jsonl")
      if not fs: continue
      seen={}
      for f in fs:
        for l in open(f): d=json.loads(l); seen[(d['example_index'],d['rollout_index'])]=d
      v=list(seen.values()); print(run,arm,r,len(v),sum(x['mean_reverse_kl'] for x in v)/len(v))
EOF
```

---

# Sources

- RHO-LOSS: Mindermann et al., 2022, *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt* (ICML 2022). https://arxiv.org/abs/2206.07137
- Rho-1: Lin et al., 2024, *Not All Tokens Are What You Need*. https://arxiv.org/abs/2404.07965
- Li et al., 2025, *Small Models Struggle to Learn from Strong Reasoners*. https://arxiv.org/abs/2502.12143
- Mirzadeh et al., 2020, *Improved Knowledge Distillation via Teacher Assistant* (AAAI 2020). https://arxiv.org/abs/1902.03393
- Xu et al., 2024, *Speculative Knowledge Distillation* (ICLR 2025). https://arxiv.org/abs/2410.11325
- Local artifacts inspected:
  - `papers/apod-story.md`, `papers/apod-acl/{main.tex, main.pdf, figures/README.md, refs.bib}`
  - `experiments/apod-data-scan/results_tables.md`, `notes/apod-critique-results.md` (CC appendix)
  - `docs/decisions.md`, `docs/perf_review.md`
  - `outputs/runs_results/{kl50,oracle8k,oracle16k,oracle16k_seed2,apod,r2-trajsel-8k}/` (resolved configs, selected and oracle shards)
