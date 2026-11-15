# Verification of the cut: `main.tex` vs `main_v4_before_cut.tex` (2026-11-12)

Inputs: `main.tex`, `main_v4_before_cut.tex`, `CHANGES_cut.md`, `papers/apod-story.md` §0/§1/§6/§8,
`experiments/apod-data-scan/results_tables.md`, `selection_tables.md`, `bank_scan.txt`, `notes/apod-critique-results.md`.
`main.tex` was not edited. The build was done on a copy in `/tmp/vc`.

## What was checked and how

| Check | Method | Result |
|---|---|---|
| Build | `tectonic main.tex` (copy) | Built, 21 pages. `main.log` has no undefined or multiply defined references or citations |
| Page limit | `pdftotext` per page | §6 Discussion ends on **page 6**, left column. Limitations and Ethics are on p.6, references start on p.6 right column. OK |
| Word targets | `wordcount.py` (prose + captions; tables, equations, comments excluded) | Abstract **170** (≤170). Main, Abstract to Discussion: 170+448+188+589+658+533+109 = **2,695** (≤2,900). OK |
| Lost numbers | Every number in the non-comment text of v4 (before `\input{appendix_tables}`) compared with main.tex | Missing from main.tex: 0.12, 0.13, 0.46, 1.45, 1.75(prose), 2.55, 3.05, 4.00, 6.80, 35.10, 36.50, 36.94, 37.65. All but three are in `appendix_tables.tex` (see below). 6.80 is derivable from Table 3 (58.15 − 51.35). 0.12 and 36.94/37.65 were removed on purpose and are listed in CHANGES_cut "Removed" |
| Numbers vs. source | Tables 2 and 3 cell by cell against `results_tables.md`. Spot checks on all main-body prose numbers (abstract, §3, §4.1–4.3, §5.1–5.3, Discussion) and on the counter-evidence nats | All match. Units unchanged: % for accuracies and rates, points for differences, nats for KL. Derived values recomputed: 58.9/32.9% gap closed (19.60/33.30, 10.95/33.30); −0.15/+6.30/−5.10/+0.50; +5.10 uniform-k gap (43.00 − 37.90); 55.1/76.8% teacher finish (1 − 0.449, 1 − 0.232); −16.65 (33.60 − 50.25) |
| `% src:` | Script: every non-comment line after `\begin{document}` that contains a digit and has no `% src` | Only design constants are left without one (n=12, k=4, ranks 5–8, temperature 1, 8-step loss, tabcolsep, `\multicolumn`). Every result keeps its `% src:` |
| Counter-evidence placement | Read §4.1 | Paragraph "Distance to the teacher does not follow accuracy" sits between "MidAlign-8k" and "MidAlign-16k" in §4.1. It gives the 0.1202/0.1168/0.1154 nats values and the length/truncation confound. The 16k seed A/B values and the round-2 flip are in App. F. They match the claims "same order in both seeds" and "flips" (0.1058>0.1029>0.0973; 0.1001>0.0996>0.0987; 0.1164<0.1179<0.1197). **OK (§8.1)** |
| References | Script over `\label`/`\ref` in main.tex + appendix_tables.tex | No dangling refs. All main-body figures (1–4) and tables (1–3) are referenced in the text. Unreferenced: `tab:noise` (appendix; it **lost** its reference in the cut) and `eq:select` (harmless) |
| Dashes | Script: U+2013/U+2014 in non-comment source; `--`/`---` outside numeric ranges; `pdftotext` scan | No Unicode dashes and no `---`. Every `--` and every rendered en dash is a numeric range (ranks 5–8, rounds 2–3, 14–24%, pages 633–638) |
| Internal names | grep over source and full PDF text for Q-bucket, Q-uncertain, Q-certain, Entropy-8k, pilot, high-rate, seed band, teacher-right, student-wrong, bucket | None in prose, captions or figures. Repo names appear only in parentheses in the appendix tables and the regimes table (the disclosed mapping). OK |
| Figures | `ls -l figures/`, includegraphics list | The same five PDFs, timestamped before the v4 snapshot, so none was regenerated. Style unchanged |

## Where each story §6 / §8 item and each Beat-6 number now lives

| Item | v4 location | Now |
|---|---|---|
| §6.1 Over time: −3.65 / +8.65 / +0.45 (8k); −0.15, +6.30 / −5.10, +0.50 (16k r2–3) | §5 When | §5.2 Over time (main) |
| §6.1 16k swings: redundant −16.65, cap-hit 85.7% | §5 When | App. C |
| §6.1 Regime: +0.05 pooled, 40 tests, saturation 26.10→44.95 | §5 When | §5.2 Training regime (main) |
| §6.1 Regime: divergent −0.46, redundant +0.13; uniform-k 43.10 at step 100 | §5 When | App. Table 29 (`tab:app-long-pooled`); App. E |
| §6.1 bf16 97.0–99.8% of updates rounded away | §5 When | App. E "Saturation and bf16 rounding". The main text keeps the qualitative clause |
| §6.1 LR probe (4× lower, −7.00; same-data retrain +0.10) | §5 When | App. E "Learning-rate probes at 16k". The main text keeps one clause |
| §6.1 Budget: earlier 8k run −2.40, step 17 −3.65, 512/544 | §5 When, Limitations | §5.2 Token budget; §3 Runs; Limitations (Test order) |
| §6.1 Pool cap-hit 91.3 / 80.4 / 80.7; selected 92.5 vs 84.4 | §5 When | §5.2 Token budget (main) |
| §6.1 Noise: 4× / 9× CI understatement | §5 Noise | §5.2 Noise (main) |
| §6.1 Noise: up to 24.40 seed difference; mean +1.21, median +0.48, 6/10 positive; 38/102 multiplicity; +5.10 uniform-k gap | §5 Noise, App. E | App. E "Noise and multiplicity" + Table `tab:noise`. +1.2 is in the Abstract and Discussion (§8.2 allows this) |
| §6.1 7-of-10 best third, p = 0.020 | §5 Noise | §5.2 Noise (main), §8.2 OK |
| §6.1 Reading + next step Qwen3.5-4B | Discussion | Intro (early, higher LR), §5.2, Discussion (2e-5; Qwen3.5-4B) |
| §6.2 Seed-spread definition (SD 3.40, ±6.7, ±15) | §4 Setup | §3 Evaluation (defined once); "the seed spread" in Fig. 2, Table 2, Fig. 3 captions, §4.1 16k, Limitations. **See M1** |
| §6.3 Full overtakes at 16k r2/r3; full beats uniform at entropy-run r8 | §4.1, Discussion | §4.1 "Against full training" (qualitative; values in Table 3); §4.2 (+3.45); App. D (−2.54 pooled r1–7) |
| §6.4 "within the token budget" | throughout | §3 Evaluation definition; Abstract "within an 8k-token budget"; §5.1 32k caveat |
| §6.5 Hypothesis origin (16k seed A r1) | Limitations | Limitations, Test order. The ordering detail (aligned > uniform > divergent > redundant) was dropped; the fact is kept |
| §6.6 Earlier 8k run overlap 512/544 | Limitations, §5 | §3 Runs; Limitations |
| §8.1 Counter-evidence in the MidAlign-8k argument | §4.1 | §4.1 (main) + App. F |
| §8.2 Noise: seed spread + 7/10 in main, rest in appendix | done in v4 | kept |
| §8.3 No internal names | | kept (see check above) |
| §8.4 Units | | kept |
| §8.5 Method schematic | Fig. 1 | Fig. 1 (unchanged) |
| §8.6 Abstract +1.2 clause, gains first | Abstract | Abstract (kept) |
| Beat 5: acc\|fin ≥97.9%, r = 0.92/202, +9.37/+0.01/−0.72, 88.0 vs 87.4, 32k need | §5.1 | §5.1 (main) |
| Beat 5: aligned vs base +19.60 split, 10.8% truncated-correct, 4.75 "unsolved at 16k" + 63.3% caveat, teacher finish rates | §5.1, §5 When | App. C |
| Beat 7: ρ 0.73, overlaps 2.54–2.72 / 0.95–1.11 / chance 1.33, +1.40 [−0.45, +3.25], −0.26, entropy-mid need | §4.2 | §4.2 (main); figure in App. D |
| Beat 7: entropy run 92.4 vs 86.5 truncated; estimator convergence, middle third hardest | §4.2, §3 | App. D; App. F |
| Beat 7: question level −2.59 [CI], 98.0%, +0.93/+0.80/+0.02, 1.38/1.63 | §4.3 | §4.3 (main); −3.10 at step 100, metric dependence and control in App. G |
| Beat 7: within-question 14–24%, 0.027–0.033 nats, 10–16%, 76–86% | §5.3 | §5.3 (main) |

## Findings by severity

| Severity | Issue | Location | Fix |
|---|---|---|---|
| **Major** | **M1. Story §6.2 ("seed-noise band next to every headline number") has regressed.** v4 put ±6.7 next to the headline in the Contributions ("+8.65 … +3.75/+3.70 … against a seed-to-seed spread of about ±6.7 points") and next to the full-training comparison ("The 6.80-point gap is about the size of the seed-to-seed spread"). After the cut, the Intro headline (47.45 vs 38.80; "about 3.7 points"), the §4.1 MidAlign-8k sentence (leads by 8.65) and §4.1 "Against full training" (58.15 vs 51.35) have no seed-spread mention. Table 2 and the Fig. 2 caption still carry it, and the 16k sentence says "inside the seed spread". | §1 para 3; §4.1 "MidAlign-8k" sentence 2; §4.1 "Against full training" | Add short clauses, about 20 words in total, which keeps the main body under 2,900. Intro: "… for uniform selection, against a seed spread of about ±6.7 points (§3), …". §4.1: "… leads uniform-k by 8.65 points (seed spread about ±6.7) …". Against full: "… (51.35%), a gap about the size of the seed spread, though …" |
| **Major** | **M2. A caveat was dropped entirely: "post-hoc".** v4: "A post-hoc count that does not depend on uniform-k favours the hypothesis … 7". The new §5.2 Noise and App. E both drop "post hoc", so a p = 0.020 now appears without its post-hoc qualifier anywhere in the paper. | §5.2 Noise, sentence 2 (and App. E) | Reinstate it: "The aligned band is the best KL third at 7 of the ten checkpoints, a post-hoc count independent of uniform-k; …" (+2 words) |
| Minor | The regime sentence is stronger than v4 and loses its qualifier in the main text. "a lower-rate retrain at 16k rules out simple overshoot" (v4: "is not simple overshoot either … one retrain"). The probe is about the 16k round-2 drop and uses one retrain per rate, while the sentence sits in the MidAlign-long paragraph. | §5.2 Training regime, last sentence | Change to "a single lower-rate retrain at 16k argues against simple overshoot of the round-2 drop (Appendix E)" |
| Minor | The Beat-5 interpretation is gone from the paper. v4 had "One untested hypothesis is that its corrections teach the student to reach an answer and stop, although the band rarely shows the student an end-of-sequence target." It appears in neither the main text nor App. C. The underlying facts are all kept, and §3 still says only finished rollouts get an EOS target. | §5.1 / App. C | Optional: add the sentence to App. C, which does not count toward the word limit |
| Minor | `tab:noise` (noise yardsticks) lost its only reference (v4: "Table~\ref{tab:noise} collects these …"). It is now an orphan appendix table. | App. E "Noise and multiplicity" | Add "(Table~\ref{tab:noise})" to the first sentence of that paragraph |
| Minor | Abstract grammar: "its first-round lead (about 3.7 points) held across two seeds and, in the seed with a full arm, beat training on all 12 rollouts". The subject of "beat" is "lead". The scope is still first-round only, so §6.3 is respected. | Abstract, sentence 5 | "…held across two seeds; in the seed with a full arm, the band also beat training on all 12 rollouts after one round using a third of the steps." (keep ≤170 words) |
| Minor | Figure order: Fig. 4 (termination) is cited in §5.1 before Fig. 3 (time course) is first cited in §5.2. CHANGES_cut says this was done on purpose, for float placement. | §5.1 / §5.2 | Optional: cite Fig. 3 once in §4.1 ("The lead is recent (§5.2, Figure 3)") |
| Minor (not new in the cut) | "on which the student hits the 8,192-token cap in 90.9% of rollouts". The source (`bank_scan.txt`) measures the 14,500-question bank, not the whole 29,434-row set. The v4 wording had the same scope. | §3 Models and data | "…; on a 14,500-question sample the student hits …", or move the clause after the bank sentence |
| Info | `eq:select` is labelled but never referenced | §3 | None needed |
| Info | Removed outright, as CHANGES_cut says: Wang et al. selector numbers (36.94 vs 37.65 ± 0.38), "mean of about 0.12 nats" (App. F keeps 0.1205), and the A100 sentence in Ethics (moved to App. A) | §2, §5.3, Ethics | None. These are not story §6/§8 items |

No blockers: no result is changed, no unit is changed, nothing required by §6 or §8 is missing from the paper, the counter-evidence paragraph is where §8.1 puts it, the word and page targets are met, there are no dangling references, no dashes in prose, and no internal names.

## Verdict

**FAIL (conditional).** Two major integrity regressions need fixing: M1 (seed spread no longer next to the Intro and §4.1 headline numbers, story §6.2) and M2 (the "post-hoc" qualifier on the 7-of-10 count, p = 0.020, is gone from the paper). Both fixes are short clauses, about 25 words in total, so the main body stays near 2,720 of 2,900 words and should still end on page 6 (check after rebuilding). Once M1 and M2 are fixed, this is a PASS. The minor items are optional.
