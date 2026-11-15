# CHANGES (voice pass): author voice + humanizer, `main.tex` (2026-11-12)

Snapshot before this pass: `main_v5_before_voice.tex` (unchanged).
Voice source: `notes/voice/krrish-voice-profile.md`, anchored on the Nov 2025 posts "Distillation >> Finetuning" and
"Major Architecture Change around ?" in `notes/voice/substack_posts.txt` (the 2023 "Less is more" post was ignored).
Register (a): plain and direct, formal enough for ACL. Humanizer skill (`~/.agents/skills/humanizer/SKILL.md`) run in
file mode with voice calibration.

## Integrity checks (script, before vs after)

- Numbers: no number was removed or changed. The only "new" digits are the list labels (1)-(5) in the Discussion and
  "after round 1", which restates the round already shown in Table 3.
- Citations: the multiset of `\cite` keys is identical.
- `% src:` comments: all 183 kept, byte-identical and in the same order.
- `\ref`/`\label`: identical.
- No em or en dashes in prose; `--` appears only in numeric ranges.
- No question marks remain in the main body prose (the rhetorical question in the Introduction is gone).

## Build and length

- `tectonic main.tex`: clean (no errors, no undefined references or citations, no overfull boxes). 21 pages, the same
  as before.
- Main body: §6 Discussion now starts on page 6, left column, line 350 (before: 352) and ends on page 6, left column.
  Limitations starts at line 365 (before: 367). Ethics and References are on page 6, right column, as before.
- Word count (`wordcount.py`, prose + captions): main body 2,743 words (before: 2,722 with the same script, so +21 words or +0.8%). The
  layout is one line shorter, because several sentences got shorter in characters even where they gained a word.
  Abstract stays at 170.

| Section | Before | After |
|---|---|---|
| Abstract | 170 | 170 |
| Introduction | 453 | 457 |
| Related Work | 188 | 188 |
| Method and Setup | 591 | 592 |
| Results | 672 | 678 |
| Analysis | 539 | 544 |
| Discussion | 109 | 114 |
| Limitations | 178 | 172 |

- Three trial edits were reverted because each pushed a line in a full column and added a page to the appendix:
  "The split shows where a difference comes from, not why." (§5.1), "MidAlign-long differs from MidAlign-8k ..." (§5.2),
  and the variant "The split locates a difference but does not explain it." The original §5.1/§5.2 wording is kept there.

## Before / after: six representative sentences

1. **Abstract (owning the hypothesis; one honest scope statement)**
   - Before: "We hypothesise that the most useful rollouts lie in an *aligned band*, ..." / "The advantage averages +1.2
     points over all ten aligned-versus-uniform checkpoints, vanishes in a long low-learning-rate run, and shows up almost
     entirely as more finished answers."
   - After: "We believe the most useful rollouts lie in an *aligned band*, ..." / "The advantage is unstable: it averages
     +1.2 points over the ten aligned-versus-uniform checkpoints, vanishes in a long low-learning-rate run and appears almost
     entirely as more finished answers."
2. **Introduction (problem first, then the fix; no rhetorical question)**
   - Before: "This is costly: every long rollout must be generated and scored by the teacher. In our main experiment the
     teacher scores 1,632 rollouts per round and we train on 544. Which ones deserve the training step? We expect the
     answer to depend on distributional alignment, ..."
   - After: "The problem is cost: every long rollout must be generated and scored by the teacher. In our main experiment
     the teacher scores 1,632 rollouts per round and we train on 544, so we must choose. We believe the choice depends on
     distributional alignment, ..."
3. **Results (concession stated once; plain scope sentence)**
   - Before: "..., a gap about the size of the seed spread, though full also trailed uniform-k at that round. Full overtook
     the aligned band at round 2 and drew level at round 3; this rests on one seed and no compute-matched baseline."
   - After: "..., a gap about the size of the seed spread. To be fair, full also trailed uniform-k at that round. The lead did
     not hold after round 1: full overtook the aligned band at round 2 and drew level at round 3. This rests on one seed,
     with no compute-matched baseline."
4. **Analysis (§5.3; split the stacked adjectives, state what was not done)**
   - Before: "Selection between questions would act on the other 76--86% and may be the larger, untested lever."
   - After: "Selection between questions would act on the other 76--86%. It may be the larger lever, and we have not tested
     it."
5. **Discussion (numbered, concrete next steps)**
   - Before: "Next steps, none yet run, are more MidAlign-8k seeds, a 32k re-evaluation, an entropy-middle rule, selection
     between questions and a larger student such as Qwen3.5-4B."
   - After: "Next steps, none yet run, are (1) more MidAlign-8k seeds, (2) a 32k re-evaluation, (3) an entropy-middle rule,
     (4) selection between questions and (5) a larger student such as Qwen3.5-4B."
6. **Limitations (numbered points instead of bold inline headers; first person, said once)**
   - Before: "\textbf{Test order.} The hypothesis was formed after MidAlign-16k seed A round 1, which is therefore not an
     independent test."
   - After: "2. We formed the hypothesis after MidAlign-16k seed A round 1, so that round is not an independent test."

Other edits of the same kind:
- Introduction: "The gain does not hold everywhere: it is clearest early and at the higher learning rate, and absent in a
  100-step run at a lower rate." (scope statement stated once, plainly); "Almost all the gain shows up as more answers
  finished within the token budget." ("shows up as" kept on purpose: the paper says the split locates a difference and does
  not explain it, so "comes from" would overclaim).
- Method: "To be fair, equal budget means equal trained rollouts and optimizer steps, not equal compute: ..." (the second
  and last "To be fair" in the paper). "Scores are exact; Appendix F compares sampled estimates." (active, shorter).
- Results: "After round 1, the aligned arm's fresh rollouts are the furthest from the teacher of the three KL arms" (removes
  the dangling participle "Re-scoring ..., the aligned arm's are ..."); "The same logic applies to whole questions."
  (replaces the vaguer "one level up").
- Abstract: "own" dropped from "a small model's own rollouts" and "within" → "at" an 8k-token budget, only to keep the
  abstract at 170 words and the page layout unchanged.
- Limitations: all six items are now plain numbered sentences (`enumerate`), with no bold lead-ins. Item 6 is now a full
  sentence ("We test one model pair, one training set and only MATH-500 ...").

## US spelling changes (prose and captions; bibliography and citation keys untouched)

| Location | Before | After |
|---|---|---|
| Abstract | hypothesise | (sentence now uses "We believe") |
| Fig. 2 caption (`fig:main`) | Grey band | Gray band |
| Fig. 3 caption (`fig:timecourse`) | Grey: the seed spread | Gray: the seed spread |
| Ethics | licence | license |
| App. A, Objective | minimise | minimize |
| App. E, Search | favouring | favoring |
| App. G, Training pool | normalised | normalized |
| App. G, Question bank | labelled | labeled |
| App. G, Question bank | teacher-labelled | teacher-labeled |
| App. G, Contamination | normaliser | normalizer |

A grep of the prose (comments stripped) and of `appendix_tables.tex` for -ise/-ised/-isation, -our, -lled/-lling,
-yse, centre, grey, licence, towards finds no remaining British form. Words already in US form were left as they were
("optimizer" was already US; "behavior", "modeling" and "center" do not occur in the prose).

Not changed: two labels drawn inside the figure PDFs still say "grey" (`fig1_main.pdf`: "grey: seed spread";
`fig2_timecourse.pdf`: "grey band: seed-to-seed spread"; strings at `make_figures.py:392` and `:455`). Figures were not
regenerated in this pass (the time-course CIs come from a separate bootstrap draw, so a rebuild would need a check of the
endpoints). Changing those two strings and rerunning `make_figures.py` is a separate, small task.

## Humanizer audit: "what still reads as AI-generated?" per section

- **Abstract.** Tells before: a hedged "We hypothesise that", a long list sentence with no stance. Fixed with "We
  believe" and one plain verdict ("The advantage is unstable:") ahead of the list. Remaining: the semicolon in the first
  sentence is the shortest way to keep 170 words; acceptable.
- **Introduction.** Tells before: a rhetorical one-line question as a paragraph closer ("Which ones deserve the training
  step?"), "We expect the answer to depend on" (hedged). Fixed: problem, then "so we must choose", then "We believe".
  Scope sentence stated once. Remaining: the three-sentence redundant/divergent/aligned triplet is a rule-of-three
  pattern, but it is the actual three-way split of the method, so it stays.
- **Related Work.** Already plain, citation-dense, "We know of no work that ..." is in his voice. No change.
- **Method and Setup.** Passive "sampled estimates are compared" made active. The equal-budget caveat now reads as an
  owned concession ("To be fair, ..."). Remaining: dense parameter sentences, which is normal for a setup section.
- **Results.** Dangling participle fixed; the "though ..." clause split into a separate concession; the full-training
  scope stated once as "The lead did not hold after round 1: ..." (his "I was wrong"-style honesty without stacked
  hedges). "One level up" made concrete.
- **Analysis.** Already plain; "Our largest effect argues against this" is in his voice. The only kept edit is §5.3,
  where the stacked "larger, untested lever" became two plain clauses. "The split locates a difference; it does not
  explain it." is a short closer, but it is a real methodological caveat, and every rewrite tried added a line, so it stays.
- **Discussion.** Next steps numbered. "On-policy post-training need not train on every rollout." stays as the plain
  opening claim.
- **Limitations.** Bold inline-header list (humanizer pattern 16) replaced by a numbered list of full sentences; the test
  order item is now first person ("We formed the hypothesis after ..."). Each limitation is stated once.
- **Appendices.** Prose left as written apart from the spelling fixes. These sections read as dry technical notes, not AI
  patterns.
- **AI vocabulary scan** (crucial, notably, highlight, underscore, key, landscape, showcase, additionally, moreover,
  delve, pivotal, furthermore, "serves as", "not only"): no hits in the prose before or after. "To be fair" appears twice,
  "We believe" twice, "essentially" zero times.

**Fact check of the rewrite:** no new fact, number, citation or claim entered. "Unstable" in the abstract repeats the
Discussion's existing "The advantage is unstable"; "fresh rollouts" repeats App. F's wording for the same measurement;
"The lead did not hold after round 1" restates the round-2/round-3 numbers already in that sentence and in Table 3.

## Verifier fixes (review_verify_voice.md, 2026-11-12)

All four items in `review_verify_voice.md` were addressed. No result, number, unit, citation, `\ref`/`\label` or
`% src:` comment changed (checked by script against the pre-fix copy: prose number multiset, cite/ref/label multiset
and the `% src:` sequence are all identical).

- **"To be fair" (second use), §4 "Against full training".** "To be fair, full also trailed uniform-k at that round."
  became "Full also trailed uniform-k at that round." The first use (§3 Method, equal budget is not equal compute) is
  kept, since it introduces a real caveat.
- **"so we must choose", Introduction.** Now "so selection is needed."
- **Abstract/Discussion repetition.** The abstract keeps "The advantage is unstable: it averages +1.2 points over the
  ten aligned-versus-uniform checkpoints, ...". The Discussion now reads "Over the ten aligned-versus-uniform
  checkpoints this advantage is unstable and averages +1.2 points; where it appears, it takes the form of more
  finished answers." Same claim, same number, same `% src:` comment.
- **Figure "grey" carry-over.** `make_figures.py` lines 392 and 455 (in-figure labels) now say "gray"; the README
  strings (l.415 print, l.643-644 table) and the l.521 code comment were also switched to American spelling.
  Regenerated with matplotlib 3.11.2 (the version that produced the previous PDFs; 3.10.8 dropped a bold font in
  fig0, so it was not used). Only the label word changed in fig1/fig2 (pdftotext diff); fig0/fig3/fig4 PNGs are
  byte-identical and their PDFs differ only in CreationDate. All 47 embedded data checks print OK. Fig. 2 bootstrap
  CI endpoints differ from the tables by at most 0.10 points (e.g. kl_high step 34 upper +6.35 vs +6.45; seed B
  round 3 upper +2.55 vs +2.45), consistent with the caption's "up to 0.1". Point estimates match exactly.

Build: `tectonic main.tex` clean (no errors, no undefined refs/citations, no overfull boxes; underfull warnings as
before). 21 pages. Pages 1-6 have the same pdftotext line counts as before the fixes; Limitations, Ethics and
References stay at the same positions on page 6. No em/en dashes or `---` in main.tex.
Backups: /tmp/main_before_voicefix.tex, /tmp/figures_before_voicefix/, /tmp/make_figures_before_voicefix.py.
