# CHANGES: revision of `main.tex` (2026-11-12)

**Inputs.** `review_verify.md` (verifier, 30 items) and `review_reviewer.md` (area chair, W1–W14 and plan A1–A11). The
storyline (`papers/apod-story.md` v3) was kept: the names, the seven beats, the main-text placement of Beat 6, and
`[Title to be decided]`.

**Build.** `tectonic main.tex` builds cleanly. There are no undefined references or citations and no overfull boxes;
only underfull-box warnings remain.

**Final page count.** 23 pages in total. The main body (Abstract through §7 Discussion) ends on **page 8**. Limitations
starts on page 8, followed by Ethics, references from page 9, and the appendix. The original draft had 22 pages and
also ended its body on page 8.

**Files changed.**

| File | Change |
|---|---|
| `main.tex` | Revised and humanized |
| `refs.bib` | 5 entries added |
| `bibkeys.md` | Rows added for the 5 new entries |
| `make_appendix_tables.py` | Run names and one caption changed |
| `appendix_tables.tex` | Regenerated. All decimal numeric tokens are identical before and after (sorted multiset diff, empty) |
| `CHANGES.md` | New |

**Untouched.** Figures were not regenerated: no plotted number changed, and the figure labels contain no repo
names. `papers/apod-story.md` was not edited (see "Declined", item 1).

---

## Evidence I re-checked or newly computed (no GPU runs)

- **Chronology (`git log`).**
  - 2026-08-15: oracle16k driver.
  - 2026-08-17: `--replicate` (seed B).
  - 2026-08-18: `--at8192` (the 8k pilot).
  - 2026-08-31 / 09-01: kl50 (MidAlign-8k).
  - `docs/decisions.md:60`: "the post-51-step eval never ran".
- **Seed B uses the same questions as seed A.** This is a new finding and it contradicts the notes.
  - In `--replicate` mode, `scripts/bucket_experiment.py:139-160` copies the pool. Only `seed` changes; `pool_seed`
    stays 42.
  - The selected `example_index` sets of oracle16k and oracle16k_seed2 are identical at rounds 0, 1 and 2 (128/128
    each). Only 171 of 512 round-0 aligned `(example_index, rollout_index)` keys coincide, so the rollouts do differ.
  - `notes/apod-critique-results.md` ("fresh question pool") and the old paper text ("different seed and question
    draw") were wrong.
  - The paper now says "same questions, different sampling and training seed". It also adds that the ±6.7 band
    therefore leaves out question-sampling noise.
- **Teacher logits are recomputed in training.** `docs/perf_review.md:186-189`: "the train stage recomputes teacher logits
  for the 544 selected trajectories … Caching … Rejected". This confirms the 12 + 4 teacher-pass accounting (verifier
  #5).
- **Reviewer check 2 (aligned band best of the three KL thirds at 7 of 10 checkpoints).**
  - Recomputed from `results_tables.md` and CC §A. The checkpoints where it is not best are the pilot r1 (divergent
    best) and seed B r2/r3.
  - Binomial P(X ≥ 7 | n = 10, p = 1/3) = 1161/59049 = 0.0197.
- **Reviewer check 3 (selected-rollout truncation).** Re-run on `outputs/runs_results/*/arms/*/rounds/*/selected/selected.jsonl`,
  de-duplicated: kl50 round 1, aligned 0.925 vs uniform-k 0.844.
- **Reviewer check 4 (re-scored pool KL after round 1).** Re-run on `arms/{kl_high,kl_mid,kl_low}/rounds/round_0{1,2}/oracle`,
  de-duplicated. The question index ranges are identical across arms within a round (e.g. 136–271 for kl50 round 1).
  Values (aligned / divergent / redundant):

  | run | round | aligned | divergent | redundant |
  |---|---|---|---|---|
  | kl50 | r1 | 0.1202 | 0.1168 | 0.1154 |
  | 16k A | r1 | 0.1058 | 0.1029 | 0.0973 |
  | 16k B | r1 | 0.1001 | 0.0996 | 0.0987 |
  | kl50 | r2 | 0.1164 | 0.1179 | 0.1197 |

  In kl50 round 2 the order flips; I added this to the paper. The round-0 pool truncation (cap-hit) is 0.913 for kl50,
  0.804 for oracle16k and 0.807 for oracle16k_seed2.
- **New citations.** All five new entries were checked against the export.arxiv.org API (titles, authors, venue
  comments).

---

## Verifier issues (`review_verify.md`)

| # | Status | What changed |
|---|---|---|
| 1 | Fixed | Intro: hypothesis formed at 16k A round 1. "According to our commit history", seed B came next (+3.70), then the 8k pilot (negative, −2.40), then MidAlign-8k about two weeks later, reusing all 512 pilot rows. The phrase "MidAlign-8k is its first test" was removed. Limitations has an "Origin and order of the tests" item. The story's §6.5 still says "first test"; see Declined 1 |
| 2 | Fixed | Abstract, Contribution 2 and the Efficiency paragraph now say round 1, one seed, and that full overtook at round 2. The Efficiency paragraph adds that the 6.80-point gap is about the ±6.7 band, that full is a weak comparator (−3.05 vs uniform-k), and that no step- or compute-matched baseline exists |
| 3 | Fixed | Abstract: "MATH-500 accuracy within an 8,192-token budget". §4 Evaluation: "'accuracy' means problems solved within the token budget" |
| 4 | Fixed | The "Reading" paragraph was replaced. The Discussion now says the effect is clearest at 16k r1 (both seeds) and MidAlign-8k s34, and that at 8k it was negative after one round. Token budget: "needs rollouts that can finish" is now labelled a conjecture that s34 partly contradicts |
| 5 | Fixed | New paragraph "What 'equal budget' means": equal trained rollouts and optimizer steps, not compute. Teacher passes are 16 per question vs 4 (uniform-k) and 12 (full). Wall-clock is 83 vs 55 min per round-arm. The intro's "every trained token costs a teacher pass" motivation was removed |
| 6 | Declined (kept, reworded, sourced) | `notes/apod-lit-review.md` §3.2 says the exact prompt string is Li et al.'s *teacher-aligned template*, observed in their HTML, and `apod/datasets/load.py:12` cites it. The sentence now reads "which matches the teacher-aligned template of Li et al." and its `% src:` points to the lit review |
| 7 | Fixed | "Among the nine 100-step arms not flagged in advance (the control included) … 43.74 against a mean of 43.64 (SD 0.62); excluding the certain arm from that set was partly post hoc" |
| 8 | Fixed | "…acts where most of the KL variance lies (76–86%), so it may be the larger lever. We have not tested it." |
| 9 | Fixed | "In this run, early in training, a third … was enough" |
| 10 | Fixed | "Retraining the second 16k round from the round-1 checkpoint at a 4× lower rate (5e-6, one retrain)…" |
| 11 | Fixed | Ethics: release covers "every run except the four smoke runs (Appendix B), whose numbers come from the project log" |
| 12 | Fixed | Quotes the r2 shift (+9.3 / −0.0 / −9.8%) and the oracle16k maximum (−16.2%), with range 10–16%. The kl50 [S-C] row is no longer used (duplicated pool) |
| 13 | Fixed | App. D: the MidAlign-8k round-0 oracle file holds two scoring passes (128 questions twice, 1,293 pairs differ). Which pass selection used is unknown. Analyses keep the first copy, and the overlaps are the same either way |
| 14 | Fixed | The control is "the first 800 bank questions in the pool's seeded shuffle order", matching the bank on H(q), accuracy and truncation. App. E adds "all teacher-labelled (452 both-wrong)". The shuffle is confirmed in `notes/apod-paper-scan-method.md:73` |
| 15 | Fixed | "hardest questions (each model solves at most one of four rollouts) … can be read as the divergent extreme at question level, although we did not score question-level KL" |
| 16 | Fixed | "The one same-data retrain … which suggests near-determinism given the data" |
| 17 | Fixed | Abstract: "is absent under long low-learning-rate training" |
| 18 | Fixed | Contribution 2 carries the ±6.7 band |
| 19 | Fixed | Discussion: "In the one seed with a full arm it beat full training at round 1 … before full overtook it at round 2" ("matched" removed) |
| 20 | Fixed | Regime table: "16 per round, 2 rounds; only round 1 evaluated" |
| 21 | Fixed | "All 512 of the pilot's round-1 training rows are among MidAlign-8k's 544" |
| 22 | Fixed | "We ran entropy-based selection as a full arm twice: in Entropy-8k, an 8-round run made before we had KL scores, and in MidAlign-long" |
| 23 | Fixed | "None of these works selects among a question's rollouts, which is the choice we study" |
| 24 | Fixed | "their selector based on sampled teacher-student log-ratios did not beat random (36.94 vs. 37.65 ± 0.38)" |
| 25 | Fixed | "a major part of the cost" |
| 26 | Fixed | OPSD is now "a self-distillation variant uses one model as both teacher and student". Dodge et al. is cited for fine-tuning seed variance, and Hochlehnert et al. for math benchmarks. madaan2024variance was dropped from that sentence for space |
| 27 | Fixed | "termination collapses (spikes in the cap-hit rate)" / "as termination collapses and recovers" |
| 28 | Fixed | Noise-table caption: "computed by the released `critic_checks.py`". bf16: "In the uniform-k arm of MidAlign-long…" (App. C and §6.2) |
| 29 | No change needed | Section order kept, per the verifier |
| 30 | Fixed | Fig. 4 (time course) caption: "a separate bootstrap draw, so endpoints can differ from the text by up to 0.1" |

---

## Reviewer edits (`review_reviewer.md`), no-new-experiment items

### Applied

- **A1 / W1 / W2 / W4: front matter.**
  - Abstract: "At one checkpoint of one run", the token budget, the round-1 and one-seed efficiency claim with
    "full training overtook it at round 2", and the pooled +1.2 mean.
  - "Shows up almost entirely as a higher finish rate" replaces "operates by teaching the model to finish".
  - Contribution 1 now says "equal training budget", with full listed separately. Contribution 2 adds "round 1" and
    the band.
  - The intro adds that the same comparison was −3.65 at step 17, plus the pooled mean +1.21.
- **A2 (partial): within-thirds test.** The 7/10 count, p = 0.020 under independence, is reported in §6.2 Noise and
  labelled post hoc and non-independent. A separate forest plot was not added (see Declined 3).
- **A3: compute.** The "equal budget" paragraph gives teacher passes per question and measured wall-clock per stage
  (from `docs/perf_review.md`), and states that selection saves optimizer steps, not teacher passes.
- **A4: mechanism check.** §6.3 reports that after round 1 the aligned arm's fresh pool is the farthest from the
  teacher in all three high-rate runs. It also reports the flip at kl50 round 2 and the length and truncation
  confound, and concludes that accuracy and distance to the teacher "may simply not move together". I kept the
  storyline's term "aligned band" and did not rename the method.
- **A5 / W5: token budget.**
  - The paragraph now uses student pool truncation (91.3% at 8k vs 80.4% and 80.7% at 16k) and keeps the teacher
    finish rates.
  - It adds that the aligned band's selected rollouts were 92.5% truncated vs 84.4% for uniform-k in the round that
    produced s34.
  - The "needs rollouts that can finish" reading is now a conjecture.
- **A6 / W8: chronology and endpoints.**
  - Pilot timing and the never-run step-51 evaluation are stated (intro and Limitations).
  - MidAlign-long is described as "planned after MidAlign-8k as a 100-step trajectory-selection run, with persisted
    Adam state to remove MidAlign-8k's loss spikes". This is in Limitations, moved there for space.
- **A7 / W7: seed band.**
  - §4 states n = 6 (aligned and divergent contrasts, rounds 1–3) and gives both bands (±6.7 and ±15). It also says
    the seeds share questions.
  - Multiplicity is reported: 38 of 102 tests at p < 0.05, 14 of them positive.
  - The Fig. 1 caption explains how to read the band.
- **A8 (partial).** The decomposition is now described as "where the difference shows up, not how training produced
  it". The "teach to finish" idea is an untested hypothesis, and the caption notes that the band rarely sees an EOS
  target.
- **A9: related work.**
  - Added RHO-LOSS (also in the intro), Rho-1, Li et al. 2025 (small-model learnability gap), TAKD and Speculative KD.
  - Added a "how we differ from Wang et al." sentence, including that our pooled +1.21 is not far from their null.
  - "These works train on all rollouts" was narrowed.
  - The OPD list was cut to the most relevant works.
  - Dropped citations (still in `refs.bib`, uncited): zhao2026semiopd, hong2026dynamics, li2026demystifying,
    ko2025distillm2, yu2026mismatch, dong2026rlstarts, lee2026sead, liu2026sweetspot, wang2025oneshot,
    kung2023active, wu2025whenmore, jiang2026crater, madaan2024variance.
- **A10 / W11–W14: presentation.**
  - Repo names in the main text were replaced: apod → **Entropy-8k**; r1/r3/r4 → **Q-buckets / Q-uncertain /
    Q-certain**. The mapping is in the appendix and in the header comment. The story names in §0 are unchanged.
  - File paths and script names were removed from figure captions.
  - Table 1 (regimes) moved to Appendix A.
  - Figure floats were reordered so numbering follows citation order. Fig. 3 (termination) is cited in §6.1 and
    Fig. 4 (time course) in §6.2.
  - The Discussion bullet list became a short paragraph.
  - Ethics now gives GPU type (A100, from the measured timings) and states that total GPU-hours were not tallied.
    The figure-style licence moved from Ethics to the appendix-table preamble.
  - App. B's bf16 counter-evidence (refreshes 1–3) is now in §6.2.
  - The question-level "divergent extreme" is now framed as an analogy.
  - "As rollouts do" was removed.
  - "Alignment" is defined in the intro as distributional closeness, distinct from preference alignment.
- **W13 (config mismatch).** The paper does not fix the files, but App. A states that the released
  `resolved_config.yaml` of MidAlign-8k and MidAlign-16k do not list their actual arms and rounds, and that the regime
  table describes those runs.

### Declined, and why

1. **Story §6.5 wording, "MidAlign-8k is its first test".** The verifier showed this is chronologically wrong (the
   negative 8k pilot came first). I followed the verifier, which matches the integrity commandment not to smooth away
   evidence, and left `apod-story.md` unedited because the storyline is owned upstream. **Action for the authors:**
   confirm the dates from run timestamps and update story §6.5.
2. **Renaming "aligned band" / "distributional alignment" (W12 suggestion).** These are fixed story names. I added a
   one-sentence definition separating them from preference alignment.
3. **New forest plot (A2) and compute table (A3 as a table).** Fig. 4 already shows 9 of the 10 contrasts with CIs and
   the seed band, and its caption now names the tenth (pilot, −2.40). A new figure and a table would not fit in 8
   pages. The compute numbers are given in prose instead.
4. **A8 offline analyses (per-MATH-level gain, finished-length distributions) and A11 (fixing the released
   configs).** These are new analyses or artifact changes, not text edits. They are listed for the authors.
5. **All Part-B experiments** (new seeds, matched baselines, 32k re-eval, trimmed-uniform arm, entropy-mid, fp32,
   between-question arm, SFT/GRPO references). They need GPUs. The Discussion names the top two, and the others are
   mentioned in §5.1 ("no step- or compute-matched baseline") and §5.2 (entropy-mid).
6. **Moving Fig. 1's grey band to a contrast plot (W7).** Story §4 asks for "Fig 1 with seed band". I kept the band
   and rewrote the caption so it reads correctly as the region within which an arm differs from uniform-k by less
   than one contrast's seed spread.

---

## Humanizer pass (embedded/file mode)

- **Removed.**
  - Signposting ("We also find where…", "Two qualifications … are developed in…", "That result needs care").
  - Staccato and announcement openers ("This paper asks…", "We state its origin plainly").
  - The stacked contribution phrasing.
  - The bulleted to-do list.
  - Excess bold on arm names in the Method section.
  - Some repeated "we", plus "therefore" and "Our reading is" constructions.
- **Varied.** Sentence lengths were varied and some short sentences merged. Clauses such as "giving an inverted U"
  were rewritten.
- **Dashes.** There are no em or en dashes in the prose. Compound hyphens that were typed as `--` (train–inference,
  student–teacher, teacher–student) are now single hyphens. Every remaining `--` in the LaTeX source is a numeric
  range. Table placeholders (`---`) became "(ref.)" and "n/a".
- **Unchanged.** No number, citation key (other than the related-work cuts and additions listed above) or claim was
  changed by the humanizing step. All `% src:` comments were kept, and 17 more were added for new statements (180
  in total, up from 163).

## Residual risks

- **Chronology.** The ordering rests on commit dates and messages, not on run timestamps.
- **Seed band.** It now correctly excludes question-sampling noise, so the true training-noise band for a new
  question draw may be wider. The earlier notes (`apod-critique-results.md` §1) still say "fresh question pool" and
  should be corrected.
- **Wall-clock.** The 83 / 55 min figures are from MidAlign-8k only. The 16-teacher-pass count assumes that the 16k
  runs had the same recompute behaviour, which is documented for the kl50 loop.
- **Reviewer-derived numbers.** These are the 7/10 count and the post-round KL values. I recomputed them from raw
  shards and tables, but they are not in `results_tables.md` or the CC output. The verifier may want them added to a
  scripted check. The `% src:` tag is "RV-check".
- **"Every run used two GPUs" and A100.** The A100 type is documented only for the measured MidAlign-8k loop.
