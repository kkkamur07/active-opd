# Critique of the draft: language and flow (2026-11-12)

Draft reviewed: `main.tex` as left by the revise pass (snapshot kept in `main_v1_before_humanizer.tex` / `.pdf`).
Method: full read of the main body, the humanizer checklist (`~/.agents/skills/humanizer/SKILL.md`), and simple counts.

## 1. Measured before the pass

| Check | Before | After |
|---|---|---|
| Abstract length (words, LaTeX-stripped) | 241 | 214 |
| Em or en dashes in prose | 0 | 0 |
| Filler vocabulary (crucial, pivotal, delve, notably, ...) | 0 | 0 |
| "±6.7" in main body | 6 | 4 (tables, Fig 1, setup, noise) |
| "+1.21" in main body | 3 | 1 (§6.2, where it belongs) |
| "full overtook at round 2" | 6 mentions | 4 |
| Sentences over 40 words | 16 | fewer (not recounted) |
| Main body ends | page 8 | page 8 |

The draft had almost none of the classic AI tells (no slop vocabulary, no em dashes, no rule-of-three padding).
Its problem was a different one, and a more serious one for this paper.

## 2. The main problem: caveat stacking

Nearly every positive sentence in the abstract and introduction was followed immediately by its caveat
("At one checkpoint of one run, ... a single contrast varies by ±6.7"; "+3.7 ... the mean advantage is +1.2").
The effect on a reader:
- the paper's own storyline (distributional alignment helps) was argued against in the first 200 words;
- the same caveat (±6.7 band, +1.21 mean, "full overtook at round 2") was repeated 3 to 6 times, so it read as defensive;
- the introduction contained a paragraph of commit-history chronology ("According to our commit history ..."), which is
  disclosure material, not motivation.

Reviewers punish both underselling and overselling. The fix is not to drop caveats but to state each one once, in
the place a reader looks for it.

## 3. What I changed (applied, compiled)

1. **Abstract** rewritten: problem, hypothesis, setup, three results, one sentence of scope, entropy proxy.
   It now says "in our main run" (single seed) and "after one round" (efficiency), and states the timing and regime
   dependence once. The "+1.2 mean over ten contrasts" moved out of the abstract to §6.2.
2. **Introduction** reorganised into four moves: cost of training on everything, the distance hypothesis, what we did
   and found, then "where the gain comes from and when it holds". The commit-history paragraph moved, complete, to
   Limitations ("Origin and order of the tests").
3. **Contributions** shortened; contribution 2 no longer repeats the full-training caveat.
4. **Figure 1 caption** cut from five clauses to three; it points to §4 for the noise estimate.
5. **Related work**: removed our own result from the related-work paragraph; softened "None of these works selects
   among a question's rollouts" to "We know of no work that selects ... by their distance to the teacher", because the
   lit review found related rollout-level selectors (PG-OPD, FiRe-OPD).
6. **Discussion** rewritten from three apologetic sentences into: the claim, its scope, two practical lessons
   (report finish rate; estimate seed noise), and the concrete next experiments, including the 4B student.
7. **Humanizer edits in Results/Analysis**: split the 60-word parenthetical sentence in §6.3; replaced the two
   "Nor ..." openers; defined units once ("tables give fractions, differences are in points").
8. **Fixed a factual slip**: Limitations said MidAlign-8k was "never evaluated *after* step 51"; the correct statement
   is that it was never evaluated *at* its planned final step 51 (step-51 numbers come from a warmup retrain).

No number, citation or result was removed from the paper; every caveat still appears at least once.

## 4. Remaining issues (not applied; need your call or a larger edit)

### Flow
- **The paper ends its analysis on counter-evidence.** §6.3 closes with "the aligned arm does not end up closer to the
  teacher", followed directly by the Discussion. Either give it its own short paragraph in §6.2 with the length and
  truncation confound, or turn it into a forward-looking question. As placed, it is the last thing the reader sees.
- **§6.2 "Noise" is a statistics dump** (seven numbers, multiplicity count, binomial post-hoc). Keep the seed-spread and
  the 7-of-10 count in the text; move the multiplicity count and pilot/step-17 comparison to Appendix C.
- **Results paragraphs carry three or four CIs per sentence** (§5.1, §5.3). Readers get these from Tables 1–2; prose
  should give the one or two that matter.
- **Units**: prose uses both percentages (abstract, intro) and four-decimal fractions (results). I added a definition;
  a cleaner fix is percentages everywhere, with one decimal.
- **No method figure.** A small schematic (12 rollouts, ranked by KL, three thirds, keep the middle) would carry the
  idea faster than Eq. 4 and the procedure box. Figure 1 could become the schematic plus the bars.

### Language
- **Internal names leak**: "Q-buckets", "Q-uncertain", "Entropy-8k", "high-rate runs", "seed band", "pilot",
  "teacher-right/student-wrong". Define each at first use in one place (setup) or replace with plain descriptions.
- **Number-heavy openings**: several paragraphs start with a figure rather than a claim. Lead with the point, then the
  number.
- **Long sentences**: about a dozen still exceed 40 words, mostly in §5.3 and §6.2.

### Integrity checks to keep in view
- The abstract no longer states the +1.2 mean; it now says the advantage "depends on when it is measured". That is
  accurate, and the mean is in §6.2, but if a reviewer finds the abstract too favourable, put it back.
- "After one round it also beat training on all 12 rollouts" is one seed at one round, and full overtook it at round 2
  (stated in §5.1 and the Discussion).

### Title (still TBD)
Options consistent with the story:
1. *Not Every Rollout Needs a Teacher: Distributional Alignment for On-Policy Post-Training*
2. *Which Rollouts Should a Teacher Correct? Distributional Alignment in On-Policy Post-Training*
3. *The Aligned Band: Selecting Rollouts for On-Policy Post-Training by Distance to the Teacher*
