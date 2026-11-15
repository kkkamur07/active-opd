# Verification review of `main.tex` (verifier lane, 2026-11-12)

Scope: every number in the text, tables, captions and appendix of `papers/apod-acl/main.tex` (636 lines), every `\cite`
key, the integrity conditions and names in `papers/apod-story.md` (v3) §0 and §6, and overclaims. I did not edit `main.tex`.

## How I checked

- **Numbers.** I checked each number against `experiments/apod-data-scan/results_tables.md` (RT), the
  `critic_checks.py` output in the appendix of `notes/apod-critique-results.md` (CC §A–K), `selection_tables.md` (ST),
  `bank_scan.txt`, `datasets.json`, `outputs/runs_results/kl50w/mcn_validate.txt`, `notes/apod-critique-{method,data,verify}.md`,
  `figures/README.md` (the figure script recomputes from the eval shards and asserts agreement with RT) and the
  `resolved_config.yaml` files.
- **Recomputed by hand.**
  - Gap closure (58.9% / 32.9%).
  - The ten aligned-vs-uniform contrasts: mean 1.205, median 0.475, 6 of 10 positive.
  - Seed |Δ| median 4.40.
  - 1.96 × 3.40 = 6.66 and 1.96 × 7.68 = 15.05.
  - CI understatement 4.0× to 9.1×.
  - Teacher finish rates 55% and 77%.
  - That the Eq. 8 decomposition is exact.
  - Batch arithmetic: 544/32 = 17, 512/32 = 16, 1536/32 = 48 and 320/32 = 10.
- **Appendix tables.** I extracted every decimal number from `appendix_tables.tex` and from RT. All 1,562 numeric tokens in
  the appendix occur in RT with matching multiplicity, and none is missing.
- **Citations.** All 64 cited keys exist in `refs.bib`; none is missing. Five bib entries are never cited, which is harmless
  with natbib: micikevicius2018mixed, nazemi2026sr, qi2025fp16, weng2026lostinrounding and zamirai2020bf16. I judged what each
  citation supports against `bibkeys.md`, whose allowed-claim column comes from the abstracts.
- **Compile.** `tectonic main.tex` succeeds with 22 pages, no undefined references or citations, and only underfull boxes.
  The main body ends on page 8, where Discussion is followed by Limitations, which is within the ACL limit.
- **Chronology.** I used `git log` dates for the integrity question on the hypothesis's origin (finding 1 below).

**Overall.** Almost every number in the draft matches its source exactly, including the de-duplicated kl50 overlaps
(2.60/4 and 1.10/4) and the ranges 2.54–2.72 and 0.95–1.11. All CIs are quoted from RT, not from the figure script's
slightly different draws. The issues below are about framing, provenance and wording, plus one chronology problem that
affects integrity condition 5.

---

## Issues, ordered by severity

| # | Severity | Issue | Location | Evidence | Required fix |
|---|---|---|---|---|---|
| 1 | **High (integrity §6.5; needs an author decision)** | "MidAlign-8k is its first test, and a second 16k seed is its first replication" looks chronologically wrong. After the hypothesis formed at 16k round 1 (seed A), the next runs were seed B (16k) and then the 8k pilot, which was **negative** (−2.40 [−4.35, −0.45]). MidAlign-8k came about two weeks later. Calling MidAlign-8k the "first test" hides a failed earlier test. The story's §6.5 wording has the same problem | §1 l.99; Limitations "Origin of the hypothesis" | `git log`: 2026-08-15 oracle16k driver; 2026-08-16 `all` arm added; **2026-08-17 "Bucket experiment: --replicate mode (different-seed rerun)"**; **2026-08-18 "--at8192 truncated-regime rerun"** (oracle8k); kl50 2026-08-31 → 09-01 (`notes/apod-paper-scan-results.md` E9, E10, E12). RT oracle8k kl_mid r1 −2.40 | Confirm the dates with the authors. Then write: "The second 16k seed was its first independent test (+3.70). An 8k pilot that followed was negative at round 1 (−2.40). MidAlign-8k, which reuses 512 of the pilot's rows, came next." Update story §6.5 the same way |
| 2 | **High (integrity §6.2 / §6.3)** | The efficiency claim has no seed band and no "early only" qualifier in the abstract. The aligned − full gap at seed A round 1 is 0.5815 − 0.5135 = **6.80 pts, about equal to the ±6.7 seed band**. Full was run in one seed only. The abstract sentence reads as if the two-seed agreement also covers full | Abstract l.75; Contribution 2 l.117; §5.1 "Efficiency" l.338–339 | RT oracle16k kl_mid r1 0.5815, all r1 0.5135; full exists only in oracle16k (Table 2 "Seed B has no full arm"); CC §B band | Abstract: "…agrees across two seeds (+3.7 points), and in one seed it beat training on all rollouts at round 1 with a third of the optimizer steps (full overtook it later)." Add the seed-band clause to the Efficiency paragraph and to Contribution 2 |
| 3 | **Medium-high (integrity §6.4)** | The abstract's headline accuracy sentence does not say it is within a token budget | Abstract l.74 | Story §6.4: gains are "solved within the token budget"; CC §D: +9.37 of +8.65 comes from finishing | "…raises MATH-500 accuracy within an 8,192-token budget from 27.9% to 47.5%…" (or "strict accuracy at an 8k cap") |
| 4 | **Medium** | The "Reading" that the advantage is "clearest early … and when rollouts can finish" is internally inconsistent. The largest effect (+8.65) is at 8k, where 57–68% of samples hit the cap. At 8k the *earliest* evaluation (step 17, and the pilot at round 1) is negative. The reading fits only 16k round 1 | §1 l.109; §6.2 "Token budget" l.452; "Reading" l.461; Discussion l.479 | RT kl50 step 17 −3.65, step 34 +8.65; cap-hit at step 34 0.566–0.679; oracle8k r1 −2.40 | Narrow it: "clearest at 16k round 1 (both seeds) and at MidAlign-8k step 34; at 8k it was negative after one round." Mark "needs rollouts that can finish" as a conjecture that MidAlign-8k step 34 partly contradicts |
| 5 | **Medium** | "Equal budget" is never defined, and the cost framing is incomplete. The aligned band pays a full teacher scoring pass over all 12 rollouts *plus* the teacher pass during training on its 4. So it uses more teacher forward passes than uniform-k (no scoring) and probably more than full. "Equal budget" means equal trained rollouts / optimizer steps only. The intro motivates the method by teacher cost ("every trained token costs a teacher forward pass") | §1 l.88–89, l.96; §3 l.207; Limitations "Cost" | Method box steps (2) and (4); commit 2026-08-15 "drop scoring passes on the random control arm"; GKDTrainer recomputes teacher logits in training (not verified whether scoring logits are reused) | Define "equal budget = equal number of trained rollouts and optimizer steps". State that the aligned band needs 12 + 4 teacher passes per question against 4 for uniform-k and 12 for full, unless the authors confirm that teacher logits are cached. Remove or soften the teacher-cost motivation in l.88 |
| 6 | **Medium** | Misattributed citation for the prompt template: "following the teacher-aligned template of \citet{li2026rethinking}". The source note does not say this, and Li et al.'s abstract describes *teacher-aligned prompt selection*, not a prompt template | §4 Models l.216 | `notes/apod-paper-scan-method.md:62` gives the prompt from `apod/datasets/load.py:13,72` with no reference to Li; bibkeys.md li2026rethinking: "cold start and teacher-aligned prompt selection" | Drop the clause, or cite the template's real origin (the standard Qwen/DeepSeek math prompt) without attributing it to Li et al. |
| 7 | **Medium** | The noise-pack sentence misstates the comparison set. The 9-arm "pack" (mean 43.64, SD 0.62) **includes** the control itself and **excludes** the three lowest arms (both-wrong, uncertain, certain). Excluding certain_questions was partly post hoc | §5.3 l.384 | CC §K: "pack (9 arms excl. both_wrong/uncertain/certain): mean 43.64" and the r3/random line is in the pack; RES §3.4 "partly post hoc" | "Among the nine 100-step arms not flagged in advance (including the control), the control scores 43.74 against a mean of 43.64 (SD 0.62)." |
| 8 | **Medium** | Overclaim: "Between-question alignment … is the larger lever." This is inferred from the variance decomposition and was never tested by a run | §6.3 l.473; Discussion bullet | ST §1 within-question share 0.14–0.24; story §7 lists the between-question arm as [need] | "…is where most of the KL variance lies (76–86%), so it may be the larger lever; we have not tested it." |
| 9 | **Medium** | "Keeping a third of the rollouts and a third of the steps is enough", from one seed at one round | §5.1 l.339 | RT oracle16k only | "…was enough in this run." |
| 10 | **Medium** | The LR-probe description is wrong. The probe retrained **round 2 from the round-1 checkpoint**, not "the first 16k round". The appendix states it correctly, so main text and appendix disagree | §6.2 "Training regime" l.444 | CC §G "vs r1 checkpoint"; RES E8 "retrain oracle16k kl_mid round 1 → eval at round 2"; Appendix B l.546 | "retraining the second 16k round (from the round-1 checkpoint) at a 4× lower rate…" |
| 11 | **Medium** | "We release … training logs of every run in this paper" is not true for the smoke runs reported in Appendix A. Their data are not on disk; the numbers come from `docs/smoke_report.md` | Ethics l.510; App. A | RES E1–E4: "Run data are not on disk" | "…of every run except the four smoke runs (Appendix A), whose numbers come from the project log." |
| 12 | **Low-medium** | The selection-shift figures (+9.4% / +0.2% / −10.9%) for MidAlign-8k come from MET [S-C]. Nobody checked whether that script de-duplicated the duplicated kl50 round-0 oracle file (3,168 rows, of which 1,293 pairs have *different* rKL values). In addition, "about ±10%" understates oracle16k's −16.2% | §6.3 l.470 | critique-verify M8 and D16; MET §4.3 table | Re-run [S-C] on the de-duplicated pool, or quote the r2 values (+9.3 / −0.0 / −9.8%) and give the range "±10–16%" |
| 13 | **Low-medium** | The "exact" KL ranking is not unique for the MidAlign-8k pool: two scoring passes exist and 1,293 duplicate pairs disagree on rKL. It is unknown which pass selection used. The MC table and the Fig. 4 kl50 point keep the first copy | §3 "KL scoring"; App. C; Fig. 4 caption | critique-verify §0.2, D16, Residual risks | Add one sentence (limitation or App. C) saying that the oracle file contains two scoring passes that differ slightly and that the analysis keeps the first |
| 14 | **Low-medium** | The control is described as "800 random bank questions". It is the first 800 bank rows that have H(q) (a prefix in pool order, not a fresh random draw), and all of them are teacher-labelled (452 both-wrong). Its H, accuracy and truncation match the bank means, so it looks representative, but the wording overstates | §5.3 l.377; App. D l.606 | `notes/apod-paper-scan-method.md:113,225`; bank_scan "first 800 [bank_random?]": buckets both_wrong 452, TRSW 176, mixed 130, both_right 42 | "a shared control of the first 800 bank questions (bank order is a shuffle of the pool; H(q) 0.519, student accuracy 0.087, matching the bank)". Confirm that the bank is shuffled |
| 15 | **Low** | "These are the divergent extreme at question level" asserts a mapping that was never measured: question-level KL was not scored | §5.3 l.379 | r1 has no oracle shards (ST §2 rKL nan) | "These can be read as the divergent extreme at question level…" |
| 16 | **Low** | "so training and evaluation are close to deterministic given the data" rests on one retrain | §6.2 l.445 | CC §G, one cell; critique-verify R21 | "…the one same-data retrain reproduced it to +0.10, suggesting near-determinism given the data." |
| 17 | **Low** | "fades under long low-learning-rate training": in MidAlign-long the effect never appeared (+0.20 at step 10, +0.05 pooled), so nothing faded | Abstract l.76 | RT r2 kl_mid step 10 +0.20; pooled +0.05 | "…and is absent under long low-learning-rate training." |
| 18 | **Low** | The headline in Contribution 2 (+8.65; +3.75/+3.70) has no seed band next to it (integrity §6.2) | §1 Contributions l.117 | CC §B | Append "(single-contrast seed band ±6.7)" |
| 19 | **Low** | Discussion says the aligned band "matched or beat full training with a third of the steps". Early on (round 1) it beat full in one seed. At round 2 full led by 5.65 pts, and round 3 was level. "Matched" has no early-training support | Discussion l.479 | RT oracle16k all r2 0.5575 vs kl_mid 0.5010; r3 0.5315 vs 0.5380 | "…and, in the one seed with a full arm, beat full training at round 1 with a third of the steps (full overtook it at round 2)." |
| 20 | **Low** | Table 1 says the pilot trained "16 (1 round evaluated)". It actually trained 2 rounds (1,024 selected rows per arm, i.e. 32 steps); only round 1 has an eval on disk | Table 1 l.242 | ST §2 oracle8k rounds 2, rows 1024; scan-results E10 "1 → 2 → 3 training rounds" | "16 per round, 2 rounds; only round 1 evaluated" |
| 21 | **Low** | "share 512 of their 544 trained rows". The pilot trains 512 rows per round and MidAlign-8k trains 544 | §4 l.259 | CC §A: oracle8k 512 rows, kl50 544, shared keys 512 | "all 512 of the pilot's round-1 rows are among MidAlign-8k's 544" |
| 22 | **Low** | "We therefore ran entropy-based selection as a full arm" implies the entropy-vs-KL analysis came first. apod (entropy-top-4) ran before KL scoring existed: the smoke runs on 08-14/15 already had entropy_top4, and the oracle scorer was added on 08-15 | §5.2 l.348 | RES E1–E5; git log 2026-08-15 "Add oracle_kl.py" | "We ran entropy-based selection as a full arm twice." |
| 23 | **Low** | Related-work generalisation: "These works train on all rollouts". \citet{zhang2026fullrollouts} train on truncated rollout prefixes and \citet{li2026rethinking} use teacher-aligned prompt selection | §2 l.136 | bibkeys.md A | "These works do not select among a question's rollouts; we ask which to keep." |
| 24 | **Low** | "their reverse-KL-based selector": per bibkeys, Wang et al.'s selector uses Monte Carlo signed log-ratios, not exact KL | §2 l.140 | bibkeys.md B wang2026beyondprompt "Do not claim" | "their selector based on sampled teacher–student log-ratios did not beat random (36.94 vs. 37.65 ± 0.38)" |
| 25 | **Low** | "generating and scoring long reasoning rollouts dominates the cost" is stronger than the cited abstracts. DistiLLM says student generation "significantly escalated computational costs", and Zhang et al. cut the rollout horizon | §1 l.88 | bibkeys.md A ko2024distillm, zhang2026fullrollouts | "…is a major part of the cost" |
| 26 | **Low** | `zhao2026opsd` (on-policy *self*-distillation) is cited for "used at scale for small reasoning models", and `dodge2020finetuning` (BERT fine-tuning seeds) for "Math benchmarks are sensitive to seeds" | §2 l.134, l.153 | bibkeys.md A, G | Move OPSD to the training-data sentence, or describe it as a self-distillation variant. Cite Dodge for fine-tuning seed variance, not for math benchmarks |
| 27 | **Low** | "length collapses" describes what are actually termination failures (cap-hit *rises*, responses get *longer*: kl_low at seed A round 2 has cap-hit 0.857 and mean length 15,396) | §5.1 l.335; §6.2 | RT oracle16k kl_low r2 | "termination collapses (cap-hit spikes)" |
| 28 | **Low** | The noise-table caption cites "critic checks, Appendix of the run notes", which reviewers cannot see. The bf16 sentence says "In MidAlign-long", but the values are for its uniform-k arm only | App. B l.537, l.563 | MET §6.1 "r2 random4 pass means [S-A]" | Point to the released script (`critic_checks.py`) instead. Write "In the uniform-k arm of MidAlign-long…" |
| 29 | **Info** | Section order differs from story §4. Selecting by entropy (§5.2) and selecting questions (§5.3) sit inside Results, before termination (§6.1) and "when" (§6.2). This is consistent with Beat 7 ("reported as a full arm in the main results"), and Beat 6 is in the main text with all its numbers | structure | story §4 outline vs main.tex | No change needed unless the authors want the outline order |
| 30 | **Info** | The Fig. 2 error bars come from a per-contrast RNG and differ from the RT CIs quoted in the text by ≤0.10 pt (e.g. +10.75 vs +10.80) | Fig. 2 | figures/README.md FIG 2 | Optional: add "(CIs may differ from the text by ≤0.1 pt because they use a different bootstrap draw)" |

---

## Integrity conditions (story §6)

| Condition | Status | Note |
|---|---|---|
| 1. Beat 6 in the main text with all its numbers | **Met** | §6.2 has −3.65 / +8.65 / +0.45; −16.65 swing; seed A r3 +6.30, seed B r3 +0.50; MidAlign-long +0.05; saturation by step 10; 97–99.8% bf16; LR probe; 8k vs 16k round 1 (−2.40, −3.65, 512/544); up to 24.40; bootstrap understatement 4–9×; mean +1.21, median +0.48 |
| 2. Seed-noise band next to every headline number | **Partly** | Present in the abstract (8k), the intro, Fig. 1, Table 1, §5.1 (8k and 16k) and Limitations. Missing for Contribution 2 (#18) and the efficiency claim (#2) |
| 3. Efficiency claimed only early | **Partly** | Met in §5.1 and §5.2, including full overtaking at round 2 and apod round 8. The abstract (#2) and the Discussion "matched" (#19) lack the qualifier. Story §6.3 says full "overtakes" at round 3 (0.5315 vs 0.5380), but the numbers show the aligned band slightly ahead; the draft's "level" is the accurate word |
| 4. Gains = "solved within the token budget" | **Mostly** | Stated in §6.1 and Limitations. The abstract headline sentence lacks it (#3) |
| 5. Origin stated; MidAlign-8k is its first test | **Origin met; "first test" doubtful** | See #1 |
| 6. Pilot overlap stated | **Met** | §4 l.259, §6.2, Noise; wording nit #21 |

**Names (§0).** All match: MidAlign-8k, MidAlign-16k seed A/B, MidAlign-8k pilot, MidAlign-long, aligned band / divergent /
redundant, uniform-k, full. "On-policy post-training" is used throughout. "Distillation" appears only for related work
(l.132, l.142) and in the abstract-free method text as "as in GKD", never as the name of our method. Repo names appear
only in parentheses or captions as a mapping, plus "apod" and "r1/r3/r4", which §0 does not rename. Optional: give apod a
paper name, because "apod" is the repo/project name and may weaken anonymity.

**Overclaim words.** The draft contains none of "proves", "reasoning ability", "full hurts" or "does not compound".
"Confirmed" appears twice (App. A loss check, teacher headroom), and both are backed by numbers. The remaining
overreaches are #8, #9, #15, #16, #17 and #4.

## Numbers checked with no issue (summary)

- **Results.** Abstract 27.9 / 47.5 / 38.8 and ρ 0.73. Table 1, every cell. Table 2, every cell. Tables 3–5 (noise, MC,
  regimes). All §5.1–§6.3 deltas and CIs; RT is quoted consistently, including both_right [+0.13, +1.50] and r3/r4
  [−2.02, −0.75] / [−2.34, −0.96]. Decompositions (CC §D). CC §E (+3.90 / +4.75). Base truncation 63% at 16k. Teacher
  finishing 55% / 77%. Step-51 values (CC §A). LR search (E13, M16). bf16 fractions (MET §6.1). Alternative-metric
  values −0.34 / −2.77 / −1.57 (MET §5).
- **Data and bank.** Bank statistics (bank_scan, histogram 12,231/792/561/524/392, which sums to 14,500). datasets.json
  values (rows, sources, answer types, CoT p50/p95, levels, contamination counts). Contamination items (DAT §4: aime2026[0]
  answer 277; math500[237] answer 22; math500[53] false positive; "0.2 pt per problem"). AIME statistics (MET [S-F], DAT §6).
- **Configs.** top-p 0.95, top-k 20, presence penalty 0, effective batch 32, clip 1.0, weight decay 0, 2 GPUs,
  correct_min 3 / wrong_max 1, min_lr_rate 0.1, persisted optimizer.
- **Appendix tables.** Mechanically identical to RT.

## Residual risks

- Chronology (#1) is inferred from commit messages and dates, not from run timestamps (the run directories in
  `outputs/runs_results` carry no original mtimes). The authors should confirm.
- I did not re-run `critic_checks.py`, `method_critic_*` or `datacritic_*`. I relied on their recorded output, which
  `notes/apod-critique-verify.md` §7 reports as re-run with identical output. The figure script independently recomputes
  the Fig. 1–4 numbers from shards.
- I did not verify whether training reuses the teacher logits from the scoring pass, which affects #5.
- I judged citation meaning from `bibkeys.md` abstracts. I did not re-fetch the papers this session; the identities had
  been checked earlier against the arXiv API (bibkeys.md, critique-verify §5).
