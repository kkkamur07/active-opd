# CHANGES v2: Task B revision of `main.tex` (2026-11-12)

Inputs: `papers/apod-story.md` §0, §1, §6, §8; `CRITIQUE_language_flow.md` §4; evidence files listed in the
`main.tex` header. Snapshot before this pass: `main_v3_before_taskB.tex`.
No result was added, removed or changed. Numbers only changed units (fraction to percent, exact x100) or moved
location; every moved number kept its `% src:` comment, with the source fraction added where the unit changed.

## Build and page layout

`tectonic main.tex` builds with no errors, no undefined references or citations, and no overfull boxes (underfull
warnings only). 23 pages in total.

| Page | Content |
|---|---|
| 1 | Title, abstract, §1 Introduction |
| 2 | Figure 1 (method schematic, `figure*`, top of page), §1 cont., Figure 2 (MidAlign-8k bars), contributions, §2 Related Work |
| 3 | §2 cont., §3 Method |
| 4 | §3 cont., §4 Experimental Setup (incl. new "Runs" paragraph) |
| 5 | Tables 1 and 2, §5.1 |
| 6 | Figure 3 (entropy), §5.2, §5.3, start of §6 |
| 7 | Figure 4 (termination), §6.1, §6.2 |
| 8 | Figure 5 (time course, `figure*`), §6.2 cont., §6.3, **§7 Discussion ends on page 8** |
| 9 | Limitations, Ethics and Reproducibility, References |
| 12-23 | Appendices A-G (regimes, smoke runs, LR search/probes/noise, KL estimator, data, AIME, full tables) |

Main body ends on page 8 (last line of the Discussion is at the bottom of the right column, so there is no slack:
any addition to §1-§7 will push the Discussion to page 9). Limitations starts on page 9.

## Item 1: counter-evidence into the MidAlign-8k argument
- Removed from the end of §6.3 the paragraph "The aligned arm also does not end up closer to the teacher ..." and the
  follow-up sentence about the order flipping at round 2.
- Added a second paragraph to §5.1 "MidAlign-8k" that leads with the claim ("Distance to the teacher does not follow these
  accuracy differences"), then gives the round-1 re-scored pool KL in nats (0.1202 vs 0.1168 / 0.1154 for MidAlign-8k;
  0.1058 vs 0.1029 / 0.0973 seed A; 0.1001 vs 0.0996 / 0.0987 seed B, noting the aligned band led in accuracy at 16k
  round 1), the flip after the second MidAlign-8k round (0.1164 vs 0.1179 / 0.1197), and the length/truncation confound
  ("accuracy gains and distance to the teacher need not move together"). Sources: `CHANGES.md` reviewer check 4.
- §6.3 now ends on the between-question variance point (76-86%, untested).

## Item 2: noise
- §6.2 "Noise" now holds only: the seed-to-seed spread of one contrast (SD 3.40, about ±6.7) against the median
  bootstrap half-width 1.66 (four times; nine times with the redundant arm, which supports the Discussion's
  "four- to ninefold"), and the post-hoc 7-of-10 count (p = 0.020 if independent, which they are not). It points to
  Appendix C.
- New paragraph "Noise and multiplicity" (`\label{app:noise}`) in Appendix C (Learning-Rate Search, Probes and Noise
  Yardsticks): the 24.40-point same-arm seed gap, the +5.10 [+3.10, +7.10] earlier-8k-run vs step-17 uniform-k gap, the
  ten-checkpoint mean +1.21 / median +0.48 / 6 of 10 positive, and the 38/102 multiplicity count (14 positive).
- CIs in Results/Analysis prose: dropped where the CI is in Table 1 or an appendix table, with a pointer to the table
  (MidAlign-8k step 17/34/51 contrasts, 16k round-1 contrasts, entropy run round 8, MidAlign-long pooled, question-level
  pooled and step-100, full vs uniform-k round 1, LR-probe retrains). CIs not in any table were kept, at most two per
  sentence (pooled steps 17+34; full rounds 1-7 pooled; metric-dependent question-level results, now in Appendix E).
  Where a claim depends on the CI excluding zero, the text says so ("both paired CIs exclude zero").
- To make room (item 5 and the 8-page limit), the metric-dependence of the question-level results and the
  nine-arm control check moved to a new Appendix E paragraph "Question-level robustness" (`\label{app:qlevel}`), and the
  per-arm mean-KL shifts of §6.3 (+9.3 / -0.0 / -9.8%, max -16.2%) moved to a new Appendix D paragraph "Mean KL of the
  selected thirds". The 83-minute round-time breakdown (13.6 / 27.5 / 25.4 / 15.2) moved to Appendix A; §3 keeps the
  83 vs 55 minutes. The contamination sentence in §4 was shortened (details unchanged in Appendix E).

## Item 3: names
- New §4 paragraph "Runs" defines once: MidAlign-8k; MidAlign-16k with seeds A and B (shared questions, only A has full);
  "an earlier 8k run, evaluated after one round", whose 512 first-round training rows are all among MidAlign-8k's 544;
  "the entropy run" (8 rounds, entropy-top-4 vs uniform-k vs full, before KL scores existed); MidAlign-long; and the three
  "question-selection runs" (by whether teacher and student solve the question; 800 most uncertain; 800 least
  uncertain). The 2e-5 / Adam-reset regime is stated once for the first four runs. The old "Training regimes" paragraph
  is replaced by it.
- §4 "Training data" now defines "solves" (>= 3 of 4 correct) and "fails" (<= 1 of 4); the text then uses
  "questions both models fail", "questions only the teacher solves", "questions both solve", "the mixed group".
- Removed from text and captions: Q-buckets, Q-uncertain, Q-certain, Entropy-8k, pilot, high-rate / high-learning-rate
  as a label (now "at learning rate 2e-5" where needed), seed band (now "seed-to-seed spread"), teacher-right/
  student-wrong, both-wrong / both-right as identifiers. One exception: the Figure 4 caption quotes the figure's own row
  label once ("questions both models fail (``both-wrong'')") so the reader can match caption and plot.
- Appendix: regimes table rows renamed ("Earlier 8k run (oracle8k)", "Entropy run (apod)", "Question selection
  (r1 / r3 / r4)"); noise-table row renamed; Appendix G intro and the bank paragraph use plain descriptions.
- `make_appendix_tables.py`: run captions renamed (entropy run, earlier 8k run, the three question-selection runs,
  LR-search run instead of "kl50w"), arm labels for the correctness groups renamed ("only teacher solves",
  "both solve", "neither solves"), the question-level caption no longer says "Q-uncertain", and every generated table now
  carries a `\label{tab:app-...}` so the text can cite it. `appendix_tables.tex` regenerated; the sorted multiset of all
  decimal numbers is identical before and after.
- `make_figures.py`: Figure 3 (entropy) legend label "MidAlign-8k pilot" changed to "earlier 8k run"; all figures
  regenerated, every built-in number check printed OK. No plotted value changed.

## Item 4: units
- Tables 1 and 2: strict accuracy, cap-hit, base and teacher rows in percent (exact x100; cap-hit with one decimal as in
  the 3-decimal source, e.g. 0.777 -> 77.7). Captions say "in %". Table 1 caption: acc|fin 98.8-99.4%, 100% for base and
  teacher.
- Prose converted (exact x100): abstract (27.85%, 47.45%, 38.80%), intro, §4 (8.7% bank accuracy, 10.8% of truncated
  samples), §5 (all accuracies; 16k efficiency 58.15 / 51.35 / 55.75 / 50.10 / 53.15 / 53.80%; entropy run
  36.50 / 35.10 / 38.55%), §6 (97.9% acc|fin floor; 63.3% cap-hit at 16k; 85.7% cap-hit; 26.10 / 44.95 / 43.10%;
  teacher finish 55.1% and 76.8%, previously rounded to 55% and 77%), Figure 2 caption (27.85%, 61.15%).
  Appendix prose: smoke runs (4.3 / 33.5 / 24.0%), teacher gate (60.0%, 47%), bf16 shares (97.0-99.8%), LR probes
  (50.20 / 50.10 / 51.15%), MATH-500 levels (66.9 / 8.0 / 93.6 / 32.1%), bank (43.5 / 54.9%, 1.2 / 15.8 / 8.7%), AIME
  (99.82%, 0.19%, at most 0.73%), step-51 evaluation (35.60 / 32.65 / 34.15 / 35.15%, cap-hit 76.9 / 74.2 / 77.4 / 84.9%).
- Removed "Tables give accuracies as fractions, and differences are in percentage points." Replaced by defining
  "points" once in §4 Evaluation. Reverse KL stays in nats; Spearman, r, H(q), losses, overlap counts and the
  estimator-agreement shares of Appendix Table (KL estimator) stay as numbers.

## Item 5: method schematic
- `figures/fig0_method.pdf` added as `figure*` (width `\textwidth`) at the start of §1; it lands at the top of page 2 as
  Figure 1. Two-sentence caption. Referenced from §1 ("train on ranks 5-8 (Figure 1)") and §3.
- The procedure box in §3 was removed; its content (12 samples, one teacher and one student pass, keep four by the rule,
  one epoch, re-score every round) is now one two-sentence paragraph that cites Figure 1.
- Figure numbering is now: 1 schematic, 2 MidAlign-8k bars, 3 entropy, 4 termination, 5 time course; figures and tables
  are first referenced in this order. Figure 5 is set at `0.92\textwidth` to fit the page limit.

## Item 6: abstract
- Scope sentence now reads "The advantage depends on when it is measured (+1.2 points on average over all ten
  aligned-versus-uniform checkpoints), is absent in a long low-learning-rate run, and appears almost entirely as more
  answers finished within the budget." Gains still come first. Abstract trimmed elsewhere ("teacher scoring is costly";
  "equal budget"; "its top third is the divergent band, not the aligned one") to 219 words by a crude whitespace count
  that also counts symbols, so the real count is a little lower.

## Item 7: polish
- Topic sentences now lead with the claim: §5.1 MidAlign-8k ("The three KL thirds form the inverted U ..."), the
  distance paragraph, MidAlign-16k ("The first-round lead reproduces across the two seeds"), efficiency ("After one
  round, a third of the rollouts was enough"), the entropy paragraphs ("Entropy-top-4 did not clearly beat uniform-k"),
  §5.3 ("The hardest questions hurt"), §6.2 Over time / Training regime / Noise.
- Repeated caveats removed: "every gain is a gain in problems solved within the token budget" (kept once in §4);
  the "saving is in optimizer steps" caveat in the efficiency paragraph (kept in §3 and Limitations); the termination-
  collapse sentence in §5.1 (kept in §6.2); the seed-spread note in the Table 2 caption (kept in §4, Table 1, Figures 2
  and 5, §6.2, Limitations).
- Transition sentence added at the start of §6. The Discussion was rewritten to match the abstract: gains first, then
  "not stable" with the +1.2-point mean, the round-2 overtake by full, the negative first 8k round, the absent long-run
  effect, the finishing account, two recommendations and the untested next experiments.
- Humanizer pass: the remaining "Nor ..." opener replaced, no -ing tack-ons, no AI vocabulary, no em or en dashes
  in prose (`--` only in numeric ranges), tense: past for what we ran, present for what the data show.

## Residual risks
- Page 8 has no slack.
- The Figure 4 plot still labels one row "both-wrong vs random questions" (not in the banned list; the caption explains it).
- Appendix table labels contain underscores (`tab:app-diff-oracle16k_seed2`); they compile, but rename if a later tool
  dislikes them.
- Title still `[Title to be decided]`.

## Verifier fixes (Task D, from `review_verify_v2.md`)

Backup of the pre-Task-D source: `/tmp/main_before_taskD.tex` (not committed). No result was changed or added.

- **M1** Abstract: "lead of about 3.7 points held across two seeds, and in the seed with a full arm it beat training on
  all 12 rollouts after one round with a third of the optimizer steps". Discussion: "...after one 16k round in both
  seeds; in the seed with a full arm it also beat full training with a third of the optimizer steps." `% src:` comments
  note "full arm in seed A only".
- **M2** Seed spread put back next to the 16k headline: §5.1 MidAlign-16k paragraph ("both leads are within the
  ±6.7-point seed spread"), Table 2 caption ("Seed-to-seed spread of a contrast: about ±6.7 points") and Contribution 2
  ("against a seed-to-seed spread of about ±6.7 points"). Abstract left without the band (body meets story §6.2).
- **M3** §4 Runs: 3.16e-6 "was within 2% of the lowest 8-step training loss in a short learning-rate search" (src:
  critique-verify M16, 0.05154 vs 0.05104), matching Appendix C.
- **L1** `make_figures.py` row label "both-wrong vs random questions" → "both fail vs random questions"; figures
  regenerated (47/47 built-in checks OK); quoted ``both-wrong'' removed from the Figure 4 (termination) caption.
- **L2** `make_appendix_tables.py`: `both_wrong` label "neither solves" → "both fail"; `appendix_tables.tex` regenerated.
- **L3** §4 Training data now defines the mixed group: "in mixed questions either model gets 2 of 4 or only the student
  solves" (src: notes/apod-paper-scan-method.md:78, bank.py:94-104; consistent with bank_scan 978 = 417 teacher-labelled
  + 561 student-mixed).
- **L4** No change (author call: story §8.2 taken to override §6.1 for the 24.40 maximum and median).
- **L5** Appendix E: "Among the nine 100-step arms other than the three lowest (two of which were flagged in advance;
  the control included)".
- **L6** Teacher gate: 60.00% and 47.0%. `make_appendix_tables.py` `pct()` now keeps the source precision (4-decimal
  fraction → 2 decimals, 3-decimal → 1 decimal), still refusing any value that would need rounding; checked that every
  column of every appendix table has a single precision.
- **L7** Table `tab:mc` agreement shares and per-third overlaps converted to percent (64.6, ...; chance 33.3%); median ρ
  kept as a correlation. `% src:` comments say "shares x100".
- **L8** Figure 3 (entropy) pool colours no longer reuse method colours: 8k pools blue_secondary (earlier 8k run
  hatched), 16k seeds green_3 (seed B hatched), MidAlign-long neutral.
- **L9** "3.7 points" → "about 3.7 points" in the abstract.
- **L10** `% src: notes/apod-critique-verify.md L9 ([27] Table 4)` added to the Wang et al. numbers.
- **L11 / page limit** To absorb the added lines, wording-only trims in §6.2–§7 (no content removed): "Training noise
  limits how far these contrasts can be read:" dropped; "We keep the finishing account as a conjecture" merged into the
  preceding sentence; "so it may be the larger, as yet untested, lever"; "bootstrap CIs over test problems"; "Next
  steps, none yet run, are ...". Discussion again ends at the bottom of page 8; Limitations starts page 9 (23 pages,
  no errors, overfull boxes or undefined references).

### Residual risks after Task D
- Page 8 still has no slack.
- Appendix G rates now mix 2-decimal (accuracy) and 1-decimal (cap-hit, pass@n) columns, by source precision.
