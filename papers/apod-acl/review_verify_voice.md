# Verify: voice pass (`main.tex` vs `main_v5_before_voice.tex`), 2026-11-12

`main.tex` was not edited. Checks: `git diff --no-index --word-diff`, script comparisons on prose with comments stripped, a fresh `tectonic` build of both versions (v5 built in /tmp/v5), and `pdftotext` page positions.

## Mechanical results

| Check | Result |
|---|---|
| Numbers (prose, comments stripped, multiset) | Nothing removed or changed. New digits: the list labels (1)-(5) in the Discussion and one "after round 1" ("The lead did not hold after round 1"), which restates the round-2/3 numbers already in the same sentence. |
| `\cite` keys (multiset) | Identical |
| `\ref` / `\label` | Identical |
| `% src:` comments | 183, byte-identical and in the same order |
| Caveats | All kept: one seed / no compute-matched baseline; equal budget is not equal compute; test order (now Limitations item 2, first person); metric = finishing; regime has no replicate, LR from 8-step loss; sampling/softmax/bf16 approximation; one model pair, one training set, MATH-500 only, AIME no signal; selection does not save teacher passes. Limitations went from bold-header `itemize` to plain `enumerate`, and no item was lost. |
| British spelling (main.tex + appendix_tables.tex prose: -ise/-ised/-isation/-yse, -our, -lled/-lling, grey, centre, licence, towards, whilst, amongst) | No hits. All 10 changes listed in CHANGES_voice.md are present in the diff. |
| Em/en dashes | No U+2013/U+2014 characters. No `---`. `--` appears only in numeric ranges (and `$-$` for minus signs). |
| Build | Clean: no errors, no undefined refs or citations, no overfull boxes (only underfull warnings, as before). 21 pages, same as v5. |
| Main body length | Same pages. Discussion, Limitations, Ethics and References all stay on page 6 in both builds. Limitations heading moves up one line (pdftotext line 60 to 59), so the body is one line shorter. |

## Meaning check of reworded sentences

All rewordings keep the original claim. Notes:
- Abstract "The advantage is unstable: it averages +1.2 ..." repeats the Discussion's existing verdict word for word. It is supported, and it is not a new claim.
- §4 "After round 1, the aligned arm's fresh rollouts are the furthest ..." drops "Re-scoring", but it measures the same thing (src comment unchanged; App. F line 540 still says "Re-scoring the fresh rollouts"). OK.
- Intro "Almost all the gain shows up as ..." keeps "almost entirely" as "Almost all". OK.
- Abstract "within" became "at an 8k-token budget", and "using" became "with a third of the steps". Same meaning.

## Issues

| Issue | Location | Fix |
|---|---|---|
| Minor (register): "To be fair," appears twice and reads a little conversational for ACL. It is acceptable, but the second use is the weaker one. | §3 Method l.178; §4 "Against full training" l.288 | Optional. At l.288, write "Full also trailed uniform-k at that round." (same length or shorter). Keep or drop the one at l.178 as you prefer. |
| Minor (register): "so we must choose" is slightly informal. | Intro l.100 | Optional. Write "so selection is needed", or leave it. It fits the author's voice and is not slang. |
| Minor (repetition): the abstract and the Discussion now open with the same clause, "The advantage is unstable: it averages +1.2 points over the ten aligned-versus-uniform checkpoints". | Abstract l.82; Discussion l.376 | Optional. Vary one of them. This is not an integrity issue. |
| Carry-over (not part of this pass): the figure PDFs still contain "grey" (`make_figures.py:392`, `:455`), but the captions now say "Gray". | fig1_main.pdf, fig2_timecourse.pdf | Change the two strings and regenerate. Check the fig2 CI endpoints against the tables (separate bootstrap draw). |

## Humanizer and register scan

AI-vocabulary list (crucial, notably, highlight, underscore, key, landscape, showcase, additionally, moreover, furthermore, delve, pivotal, serves as, not only, essentially, robust, leverage, comprehensive, insights, ultimately, interestingly, indeed, thus/hence): no hits in the prose.

- I grepped for `, X-ing` tack-ons. The one hit (l.142, "including a selector ...") is a normal appositive, not a tack-on.
- I found no new rule-of-three padding. The redundant/divergent/aligned triplet is the method's actual three-way split.
- I found no staccato fragments or signposting. The rhetorical question in the Introduction was removed.
- I found no slang or blog phrasing beyond the minor items above.

## Verdict

PASS. No result, number, unit, citation, `% src:` comment or caveat changed or disappeared. The prose uses American spelling, contains no em or en dashes, and the main body did not grow (one line shorter). The remaining items are minor and optional.
