# CHANGES (cut pass): shortening `main.tex` (2026-11-12)

Snapshot before this pass: `main_v4_before_cut.tex` (unchanged). Storyline and integrity rules: `papers/apod-story.md` §0, §1, §6, §8.
No result was added, changed or re-unitised. Every number that stays in the paper keeps its `% src:` comment. Numbers that
left the main body now sit in the appendix with the same comment. A few numbers were removed outright (listed below).
Figures were not regenerated; all five PDFs in `figures/` are still used.

## Build

`tectonic main.tex` builds cleanly: no errors, no undefined references or citations, no overfull boxes. The only warnings
are underfull boxes and the existing `lineno.sty` UTF-8 notice from the template. 21 pages in total (previously 23).

**Main body ends on page 6, left column** (§6 Discussion; previously page 8). Limitations and Ethics follow on page 6, and
the references start on page 6, right column.

| Page | Content |
|---|---|
| 1 | Title, abstract, §1 |
| 2 | Fig. 1 schematic (`figure*`), Fig. 2 bars, end of §1, §2, start of §3 |
| 3 | §3 Method and Setup (Eqs. 1-3, Table 1 runs), start of §4 |
| 4 | Tables 2 and 3, §4.1-§4.3, start of §5 |
| 5 | Fig. 3 time course (`figure*`), Fig. 4 termination, §5.1-§5.2 |
| 6 | §5.2 noise, §5.3, §6 Discussion, Limitations, Ethics, References |
| 9-21 | Appendices A-I |

## Word counts

Counted with `wordcount.py` (now in this folder): prose plus figure and table captions; tabular contents, equations,
LaTeX comments and citation keys excluded. With this script the "before" numbers sit close to the counts in the task
(Intro 726 vs 710; Results 1089 vs 1095). They differ for Analysis (1133 vs 889): this script counts the time-course
caption under Analysis, where that figure was defined.

| Section | Before | After | Target |
|---|---|---|---|
| Abstract | 227 | 170 | <= 170 |
| 1 Introduction | 726 | 448 | <= 450 |
| 2 Related Work | 376 | 188 | <= 250 |
| 3 Method + 4 Setup (now one section, "Method and Setup") | 434 + 598 = 1032 | 589 | <= 650 |
| Results (now §4) | 1089 | 658 (includes the time-course caption, which moved here; prose about 590) | <= 800 |
| Analysis (now §5) | 1133 | 533 | <= 600 |
| Discussion (now §6) | 180 | 109 | <= 120 |
| **Main body, Abstract to Discussion** | **4583** | **2695** | **<= 2,900** |
| Limitations | 355 | 178 | <= 200 |
| Ethics and Reproducibility | 104 | 55 | <= 80 |

## Structure changes

- §3 Method and §4 Experimental Setup are merged into **§3 Method and Setup** (`\label{sec:method}`; `sec:setup` is gone,
  and all references now point to `sec:method`). It keeps Eq. (D_t), the score Eq. and the selection rule, the
  schematic, and a new compact **Table 1 (runs)**: run, arms, questions per round, steps, cap, LR, seeds. The row set is
  MidAlign-8k, MidAlign-16k A/B, the earlier 8k run, the entropy run, MidAlign-long and the question-selection runs (x3).
  Every cell comes from appendix Table `tab:regimes`, and each row has a `% src:` comment.
- The seed spread is defined once (§3 Evaluation: SD 3.40, about ±6.7, ±15 with the redundant arm). Everywhere else the
  text says "the seed spread": Fig. 2 caption, Table 2 caption, §4.1, Fig. 3 caption, §5.2 Noise and Limitations.
- Results: §4.1 "The aligned band" (Tables 2-3, MidAlign-8k paragraph, the counter-evidence paragraph on distance to the
  teacher kept inside the MidAlign-8k argument as story §8.1 requires, the MidAlign-16k paragraph, and "Against full
  training" in two sentences); §4.2 entropy (one paragraph); §4.3 questions (one paragraph).
- Analysis: §5.1 Termination (equation, figure, short text); §5.2 When alignment helps (over time, regime, token budget,
  noise; two to three sentences each); §5.3 fragility (three sentences).
- The time-course `figure*` is now defined before §4.2, so it floats to the top of page 5 instead of falling after the
  Discussion. It is therefore Figure 3, and termination is Figure 4.
- The entropy figure moved to the new Appendix D, with a one-line reference in §4.2.
- Limitations is now an itemised list (Seeds, Test order, Metric, Regime, Approximation, Scope and cost).

## Said once (repeats removed)

| Fact | Kept at | Removed from |
|---|---|---|
| +8.65 (aligned − uniform, step 34) | Table 2 (with CI), §4.1, §5.2 time course | Intro, contributions, §5.1 prose (now "the aligned band's lead"), Discussion |
| ±6.7 seed spread | §3 Evaluation (defined) | Fig. 2/Table 2/Table 3 captions, contributions, §4.1, §5.2, Limitations (all now "the seed spread") |
| Full overtook the aligned band at round 2 | §4.1 "Against full training" | Abstract, Discussion |
| "One seed" | Fig. 2 caption, Table 2 caption, §4.1 full comparison, §4.3, Limitations | Repeats in §4.1 and §5.2 |
| 512-of-544 overlap | §3 Runs and Limitations (story §6.6 asks for it; the Limitations item covers test order) | §5.2 token budget, noise paragraph |
| Finish-rate decomposition (+9.37 / +0.01 / −0.72) | §5.1 | Other three contrasts moved to App. C (shown in Fig. 4) |
| −3.65 at step 17 | §5.2 over time | §4.1 (now "The lead is recent (§5.2)"), token budget ("in both 8k runs") |
| −2.40 earlier 8k run | §5.2 token budget | Fig. 3 caption, Limitations |
| +1.2 mean over ten checkpoints | Abstract, §6 Discussion | App. E keeps the full statistic (+1.21, median +0.48, 6 of 10) |
| 4-9x CI understatement | §5.2 Noise | Discussion |

## Moved to the appendix (paragraph or figure → destination)

1. Loss equation `eq:loss`, the GKDTrainer β=1/λ=0 detail and the fixed-prefix/no-gradient remark (old §3 "On-policy
   post-training objective") → **App. A "Objective, Budget and Training Regimes"**, paragraph "Objective".
2. Exact prompt string, no presence penalty, gradient clip 1.0, no weight decay (old §4 Models) → App. A "Objective".
3. "What equal budget means": 16/4/12 teacher passes and the 83 vs 55 min wall-clock with the 13.6/27.5/25.4/15.2
   breakdown (old §3 + old App. A) → App. A "Equal budget". §3 keeps one sentence (16 vs 4 vs 12) with a pointer.
4. Resolved-config caveat, MidAlign-long's planning rationale, MidAlign-8k never evaluated at its planned step 51 (old
   Limitations and App. A) → App. A "Configuration files".
5. KL-scoring estimator remark (old §3 "KL scoring") → §3 keeps one sentence pointing to App. F.
6. Training-data statistics: mean reference length 2,897 tokens and max under 5,000; bank accuracy 8.7%; the definition of
   "mixed" questions; the shuffle-order control and how it matches the bank (old §4 Training data, old §5.3) →
   **App. G "Data Statistics and Contamination"**. §3 keeps one sentence each for data and contamination.
7. Contamination detail (two same-answer duplicates) → App. G (it was already there in full; §3 keeps one sentence).
8. Counter-evidence nats for 16k seeds A/B round 1 and for the MidAlign-8k round-2 flip → **App. F, new paragraph
   "Distance to the teacher after training"**. §4.1 keeps the MidAlign-8k round-1 values (0.1202 / 0.1168 / 0.1154 nats)
   and the confound.
9. Termination details: aligned vs base (+19.60 = +19.49 − 0.34 + 0.45), entropy-run and both-fail splits, the 10.8%
   truncated-but-correct rate, the 4.75-point "unsolved at 16k" split and its 63.3% caveat, teacher finish rates at 8k/16k,
   and the seed-A redundant collapse (−16.65, 85.7% cap-hit) → **new App. C "Termination Details"**.
10. Entropy figure (`fig4_entropy.pdf`), entropy-run truncation 92.4% vs 86.5%, full-vs-uniform pooled rounds 1-7
    (−2.54 [−4.50, −0.64]) → **new App. D "Entropy as a Proxy"**.
11. LR-probe detail (4× lower rate, −7.00 and +0.10 vs round 1) and the bf16 rounding numbers (97.0-99.8%), plus uniform-k at
    step 100 (43.10%) → **App. E "Learning-Rate Search, Probes and Noise"**. §5.2 keeps one sentence with a pointer.
12. Question-level: both-fail at step 100 (−3.10 [−4.75, −1.50]) and the shared-control description → App. G
    "Question-level robustness" and "Question bank".

## Removed (no longer anywhere in the paper)

- Related work: Wang et al.'s selector numbers (36.94 vs 37.65 ± 0.38) and descriptive clauses; citations kept and grouped.
- §4 / §6: the "mean of about 0.12" nats aside in the fragility paragraph (App. F still gives the pool mean 0.1205).
- Ethics: the A100 sentence (now in App. A as "Every run used two GPUs").

## Appendix trimming

- App. A merges the old regimes text with the moved method material. The duplicated "Seed B uses the same questions" and
  Adam-reset sentences are dropped from the `tab:regimes` caption; §3 states them.
- App. B (smoke): shorter; the "validated the plumbing" sentence is dropped. The project-log provenance moved here from Ethics.
- App. E: the noise paragraph no longer repeats §5.2; it lists which ten checkpoints count and which three the aligned band
  loses.
- App. G "Question-level robustness" no longer repeats the 1.38/1.63 strict numbers that §4.3 gives.
- App. I (full tables): the intro drops the strict definition already given in §3.
- All tables are kept: regimes, noise yardsticks, MC estimator and every generated table in `appendix_tables.tex`.

## Language

- No em or en dashes in prose (checked by script; `--` appears only in numeric ranges).
- Filler and signposting removed ("We now ask", "In this section", "Note that"); short sentences, claim first, "we" voice.
- Results prose gives at most one CI per sentence; the other CIs are in tables or the appendix.

## Residual risks

- Word counts depend on the counting method; with captions excluded the main body is about 2,300 words.
- The Results/Analysis split in the table assigns the time-course caption to Results because that is where the figure is
  now defined in the source.
- The figure order in the text: §5.1 cites Fig. 4 (termination) before §5.2 cites Fig. 3 (time course), because Fig. 3 had
  to be defined earlier so it would float onto page 5.
- The full-vs-aligned efficiency claim is now two sentences, and the "one seed, no compute-matched baseline" caveat is
  folded into the second sentence.

## Verifier fixes (review_verify_cut.md, 2026-11-12)

- **M1 (seed spread next to headline numbers, story §6.2).** Intro: "against 38.80% for uniform selection (seed spread about ±6.7 points, §3)". §4.1 MidAlign-8k: "leads uniform-k by 8.65 points (seed spread about ±6.7)". §4.1 Against full training: "(51.35%), a gap about the size of the seed spread, though …".
- **M2 ("post-hoc" qualifier).** §5.2 Noise: "a post-hoc count independent of uniform-k" restored before the p = 0.020 statement.
- Regime sentence (§5.2) softened: "a single lower-rate retrain at 16k argues against simple overshoot of the round-2 drop (Appendix E)".
- Beat-5 interpretation restored in App. C (not in the word count): "One untested hypothesis is that the aligned band's corrections teach the student to reach an answer and stop, although the band rarely shows the student an end-of-sequence target."
- `tab:noise` referenced again: App. E "Noise and multiplicity" opens with "Table~\ref{tab:noise} collects the noise yardsticks."
- Abstract grammar: "…held across two seeds; in the seed with a full arm, the band beat training on all 12 rollouts using a third of the steps." To stay at 170 words, "exact" was dropped from "by exact reverse KL" in the abstract (the method section still says exact).
- Figure order: §4.1 now cites Figure 3 ("The lead is recent (§5.2, Figure 3)") before §5.1 cites Figure 4.
- 90.9% scope: "…OpenThoughts; on a 14,500-question sample the student hits the 8,192-token cap in 90.9% of rollouts" (bank_scan.txt measures the 14,500-question bank).
- No result, unit or `% src:` comment changed; no figure regenerated; no em/en dashes added (script check).

**Final word counts (wordcount.py; prose + captions, tables/equations/comments excluded):** Abstract 170; Introduction 453; Related Work 188; Method and Setup 593; Results 672; Analysis 539; Discussion 109. Main body (Abstract to Discussion) = **2,724** (target ≤ 2,900). Limitations 178, Ethics 55 (not counted).

**Build:** `tectonic main.tex` clean: 21 pages, 0 undefined references/citations, 0 overfull boxes (only underfull-box warnings, as before). **The main body (§6 Discussion) ends on page 6, left column**; Limitations and Ethics follow on page 6, references begin on page 6.
