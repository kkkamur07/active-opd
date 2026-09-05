# Decisions log

Dated decisions from planning sessions, one line each, newest first. Reasoning
lives in the linked ADR or doc; vocabulary in CONTEXT.md.

## 2026-09-01 train-stage decisions (evening, train.py branch)

- **LR schedule continues across refreshes** ("it has to continue"): one
  warmup + cosine_with_min_lr(0.1) schedule over the run's
  `train.total_training_steps` (100), warmup 5% = 5 steps once at the start;
  every train launch builds the scheduler for the total and advances it by
  `--global-step-offset` (only the scheduler; Adam moments come from
  persist_optimizer). Unset total = the old per-launch schedule.
  tests/test_persist_optimizer.py (e): 10+10 == 20, bit-exact.
- **Per-step bf16 rounding diagnostic** `bf16_rounded_frac` (+ per block):
  fraction of parameter elements whose Adam update is below half a bf16 ulp
  of the weight, from the post-step optimizer state. Pure-bf16 training at
  peak lr 3.16e-6 is expected to lose most updates; the number says how many.
- **Per-step batch diagnostics** (overlap ratio, overlap advantage,
  |H_S - H_T|, response tokens, cap-hit fraction) computed inside the train
  stage from the Liger loss's own hidden states (`train.diag_every`,
  `diag_chunk`); logged to log_history, TensorBoard export and W&B.
- **W&B tracking** (`apod/tracking.py`, `conf/tracking.yaml`): one run per
  arm, offline by default, x-axis = global training step; `wandb sync` later.
- **Atomic checkpoints**: the train stage saves into `checkpoint.tmp` and
  renames, so a crash mid-save can never leave a loadable partial checkpoint.

## 2026-09-01 planning session (post 1 Sept meeting)

- **Diversity: no arm, no monitor.** Not separate from distributional similarity (reverse KL) and entropy; covered by those signals and by pass@k in eval.
- **Padding / collation: leave as is.** Measured pad waste after length grouping
  is 0.3% at cap 8192 and 0.45% at 16384; the rank-interleave patch saves ~0.4 min
  per 100 steps at 8k and changes the shuffle-seed source, so it stays on branch
  `worktree-agent-a0b4a8727f85f374c` unmerged. Padding-free packing is unsafe on
  Qwen3.5 (Gated DeltaNet fallback ignores sequence boundaries). ADR 0007.
- **Run 1 = correctness experiment.** Random OpenThoughts questions, 4 student + 4
  teacher rollouts at 8192, strict 3-of-4 labels, four buckets TC/SW, TC/SC, TW/SW,
  Mixed (= none of the other three). One arm per bucket, 800 questions x 4 rollouts
  all trained, **100 training steps per arm**. No random-question control arm.
  ADR 0005.
- **Run 2 = trajectory-selection experiment, after run 1.** Independent random
  questions, 12 rollouts per question, keep 4: entropy_top4, kl_high, kl_mid,
  kl_low, random. 100 steps per arm. No "all" arm (random-k is the no-selection
  baseline, see CONTEXT.md).
- **Cap 8192 everywhere** (labels, training, eval); cap-hit is wrong (strict).
- **Eval every 10 training steps ("refresh")**: MATH-500 avg@4 + pass@4 on all
  500, AIME 2025+2026 avg@16 + pass@16 on all 60, both at 8192, strict. ADR 0006.
- **Per-step logging on the training batch, all in TensorBoard**: loss, grad_norm,
  learning rate (already exported per step by scripts/tb_export.py), plus new:
  top-16 overlap ratio, overlap-token advantage, mean |H_S - H_T| per token,
  tokens in the step, cap-hit fraction of the batch. Evals only every 10 steps.
  Purpose: see training instabilities (kl50's per-refresh restart spikes came from
  Adam state resetting; fixed by persist_optimizer in kl50w).
- **r3 uses student entropy** (uncertainty sampling queries the learner; teacher
  entropy would measure label reliability, which r1's teacher label already probes).
- **Vocabulary**: "training step" (not round), "question" (not prompt), "refresh"
  for the every-10-steps eval + re-rollout.
- **Question bank**: student sweep ~14,500 questions, teacher only where its label
  decides a bucket; stored with grades, truncation, lengths; reusable.
- **kl50 headline is at 34 training steps** (eval after two refreshes; the post-51-step
  eval never ran). kl_mid's strict gain came with the largest cap-hit drop
  (0.777 -> 0.566): strict gains partly measure learning to finish; every results
  table shows cap-hit next to avg@4.
- **Entropy vs KL offline**: student entropy ranks a question's 12 rollouts like the
  exact reverse KL at Spearman 0.73 (docs/analysis_entropy_vs_kl.md); length does
  not. Motivates the entropy_top4 arm without a teacher forward.
- **Fused kernels kept.** flash-linear-attention 0.5.2 installed in the main venv
  (2B fwd 2.05x, 9B fwd 1.67x, 2B fwd+bwd 5.8x at 8192; docs/kernels.md). All new
  runs start fresh on the fused path.
- **Run 3 = question-level uncertainty sampling.** Score every banked question by
  student question entropy H(q) (mean H(τ) over its 4 rollouts); arm A = the 800
  most uncertain questions, arm B = 800 random questions; 4 rollouts each, all
  trained, 100 steps. Tests whether entropy sampling works at question level.
  Student entropy, not teacher: uncertainty sampling queries the learner.
- **Indicator combination (bucket x selection rule stack): deferred past run 3.**
- **Perf**: vLLM concurrency 256; async grading; MC reverse-KL estimator
  behind a flag with offline validation, not used for selection yet.
- **TensorRT-LLM: no.** Generation is under a third of wall-clock; scoring and
  eval are the levers.
- 2026-09-01: **Peak LR 3.1623e-06 for kl50w and r1/r2/r3** (USER: "use the best lr from the probe"). Source: `outputs/runs/kl50w/lr_probe.json` `chosen_lr`, refined grid over [1e-6, 1e-5] on kl_mid at 20:20 UTC (near-ties within 2% go to the lower LR). Same warmup(1) + cosine_with_min_lr(0.1) per refresh, optimizer state persisted. The 2e-5 in `conf/train/gkd.yaml` stays as the untouched default; the experiment configs override it.
- 2026-09-01: **Fused GDN kernels stay on for every run from here** (USER: "we need to use the kernels to get the speedup"). flash-linear-attention 0.5.2 is installed in `.venv`, the environment every launcher and the peer's chain use; `.venv-kernels` was only the benchmark sandbox and carries nothing extra. Runs that started before the install (kl50) are not numerically comparable at bf16 level to runs after it; kl50w banks nothing trained from kl50: only the base student's pass-0 rollouts, eval and exact-KL scores (generation outputs); every kl50w train pass starts from the base student with the kernels on.
- 2026-09-01: **One LR schedule per run, continued across refreshes** (USER: "the lr schedule ... has to continue"). Warmup 5 steps = 5% of the 100 training steps, once at the start; then cosine_with_min_lr(0.1) to step 100. Each train pass builds the scheduler for the full run and advances it to the global step offset; Adam state persists as before. Replaces the kl50w per-round LR cycles. Verified need: tests/test_persist_optimizer.py showed the per-pass rebuild restarts the LR at [0, peak] every pass.
- 2026-09-01: **Per-step bf16 rounding diagnostic** (USER: "do the diagnosis"): fraction of parameter elements whose Adam update is below half a bf16 ulp and rounds away; logged as train/bf16_rounded_frac with grad_norm and lr. Student trains in pure bf16, no fp32 master weights.
- 2026-09-01: **kl50w cancelled before its first train pass** (USER: "we don't need this again, kill this"). Its LR verdict (peak 3.1623e-06) is kept. Order from here: batch-size preliminary with the fused kernels (largest micro-batch at cap 8192, effective batch 32), then `bank-8k`, then `r1-correctness-8k`.
- 2026-09-01: `.venv` stays the single environment (it has flash-linear-attention 0.5.2); the duplicate `.venv-kernels` benchmark sandbox was deleted.
- 2026-09-01: **Sampled-token reverse-KL estimators validated against the exact full-vocab KL; exact stays the selection statistic** (USER: "do experiment that 1,2,4,8,16,32,64,128 tokens (sampled randomly) and how much on full vocab and see the difference and then we select"). `scripts/oracle_kl.py --estimator mcn` on the kl50 round-0 pool (1632 rollouts, 136 questions), `--validate-mc` table in `outputs/runs/kl50w/mcn_validate.txt`. Generated-token estimate (the only one vLLM gives for free): same-tertile agreement 0.646, kl_mid 0.52, biased -20% (truncated sampling). n i.i.d. draws from the student's full distribution per position: 0.672 (n=1), 0.792 (4), 0.865 (8), 0.918 (32), 0.953 (128); unbiased, but draws need both models' full logit rows, so no cheaper than exact in the HF scorer. The top-16 overlap advantage (0.85 kl_mid agreement, rho -0.97, from top-16 logprobs) is the only cheap proxy that matches; nothing changes in config unless the USER asks for a cheaper scorer.
- 2026-09-01: **W&B entity pinned to `krrish-agar-ludwig-maximilianuniversity-of-munich`** (conf/tracking.yaml). The API key in ~/.netrc defaults to the team entity `asthadu29-ludwig-maximilianuniversity-of-munich`, where `wandb.init(project="apod")` fails with "the provided API key cannot access this resource" (no project-create rights); the personal entity creates runs (checked 20:56 UTC, test run deleted). Runs land at wandb.ai/krrish-agar-ludwig-maximilianuniversity-of-munich/apod. CPU tests compose with tracking.mode=disabled.
- 2026-09-01: **Driver branch merged to master (8e682d0)**: apod.driver step-based loop, conf/experiment/{refresh_8k,r1_correctness_8k,r2_trajsel_8k,r3_qentropy_8k}.yaml, selection rules, refresh plots; all seven CPU test files pass. Dry-run launch plan checked against the contract (one rollout_eval session per refresh with `--eval-dataset math500 aime2526`, torchrun on both GPUs with `--global-step-offset` 0/10/../90, LR 3.1623e-06, warmup 5, total 100, persist_optimizer, keep-last-2 checkpoints).
- 2026-09-01: **Train micro-batch 16 x accum 1 (effective 32) for every run from here** (batch probe with the fused kernels at cap 8192: peak 73,949 MiB, 42.6 s/step; micro 8 x 2: 56,885 MiB, 44.5 s/step; rule = largest micro under 75,000 MiB). Only 4% faster than 8 x 2, so 8 x 2 is the documented fallback on any OOM. **vLLM concurrency stays 256**: 512 matched 256 for the 2B (71.8 req/min both) and slowed the 9B (25.5 vs 34.7 req/min); adoption needed >= 1.8x. Tables in docs/perf_review.md.
- 2026-09-01: **Driver opens each arm's W&B run only around its own `log_refresh` call** (apod/driver.py `record`). A W&B run id can be live in one process at a time; the driver held it across the whole arm while the train stage resumed the same deterministic id in its own process -> "run ID ... is in use" on the first train launch (driver smoke, 21:5x UTC). Rule that follows from the deterministic ids: **never delete a W&B run in project apod** -- a deleted id can never be re-created ("was previously created and deleted; try a new run id"), which would brick the run dir it belongs to; the smoke had to move to run dir smoke-r1b for that reason.
- 2026-09-01: **GPU validation passed; production chain launched** (handoff checklist a+b). Bank smoke (`outputs/runs/bank-smoke`, 40 questions, target 2/bucket): built in ~10 min, all buckets filled, `--report` OK, rerun via `--bank-dir` generated nothing. Driver refresh smoke (r1 shape, 2 arms x 4 steps in 2 passes, run dirs smoke-r1b/smoke-r1c): every per-step key in W&B under `train/*` (loss, grad_norm, learning_rate, response_tokens, cap_hit_frac, overlap_ratio_top16, overlap_adv_top16, abs_entropy_gap, bf16_rounded_frac + per block); learning_rate 0 -> 6.325e-07 | 1.265e-06 -> 1.897e-06 across the two passes (one warmup, no restart); optimizer_state.pt written per pass and "optimizer state restored from" on the next; SIGTERM after step 3 of 4 then relaunch: `--global-step-offset 2`, state restored, step 3 reproduced bit-for-bit (loss 0.06767, LR 1.265e-06), SMOKE DONE; kill during the final eval + relaunch also resumed cleanly (metrics.jsonl upserted, 6 rows, no duplicates); peak train memory 76.76 GiB with diagnostics at micro 16 (< 80, diag_chunk unchanged). Smoke run dirs deleted; the W&B smoke runs stay (see the never-delete rule above). Chain `bank-8k -> r1-correctness-8k -> r2-trajsel-8k -> r3-qentropy-8k` launched 2026-09-01 ~23:00 UTC via scratchpad exp_chain.sh (log outputs/runs/exp_chain.log; 3 attempts per stage, resume-safe).
- 2026-09-01: **OPEN QUESTION for the USER (not changed, recorded per the handoff rule): pure-bf16 weights discard ~97% of Adam updates at the chosen LR.** The new `bf16_rounded_frac` diagnostic (smoke, both arms) reads 0.9875 at LR 6.3e-07, 0.9778 at 1.27e-06, 0.9748 at 1.9e-06 -- i.e. at peak 3.16e-06 roughly 96-97% of elements get an update below half a bf16 ulp and do not move; only elements with |w| < ~1e-3 train. The student is loaded in bf16 with no fp32 master copy (train.py `from_pretrained(..., dtype=bfloat16)`, Adam moments bf16), so this is real, and it was equally true of kl50 and of the LR sweep that produced 3.16e-06 (the sweep optimised the 8-step loss *under* this rounding). Options, none applied: (a) fp32 master weights (load fp32 + bf16 autocast): +~16 GB static -> micro 16 no longer fits (76.8 + 16 GiB), micro 8 x 2 does (~73 GiB); ~4% slower per the probe plus fp32 all-reduce; invalidates the LR verdict (re-run the 5-point kl_mid refine, ~1 h). (b) Kahan-summation / stochastic-rounding optimizer on bf16 params (e.g. `optimi`), memory-neutral, also needs an LR re-check. (c) Proceed as specified (all arms share the handicap, comparisons stay valid; kl50 learned under it: base 0.2785 -> kl_mid 0.4745). The chain launched as specified; bank-8k (~17 h, precision-independent) leaves a decision window before r1's first train pass -- decide before then or r1 runs with (c).
- 2026-09-02: **Train micro-batch back to 8 x accum 2 (effective 32 unchanged) for every run** -- the documented OOM fallback applied. r1-correctness-8k, arm teacher_right_student_wrong, step090 train pass OOMed at step 7/10 on GPU 1 ("Tried to allocate 1.04 GiB ... 131.94 MiB free of 79.25 GiB") at micro 16 x 1; the probe's 73,949 MiB peak / smoke's 76.76 GiB left no headroom for a long-tailed batch. Applied to conf/train/gkd.yaml, conf/experiment/refresh_8k.yaml and, because an existing run dir reads only its own resolved_config.yaml, edited in place in outputs/runs/r1-correctness-8k/resolved_config.yaml (the only two keys changed). Chain stopped 18:5x UTC and relaunched; r1 resumes at that train pass (--global-step-offset 90, optimizer state from round_08). Cost: ~1 h of r1 wall clock plus ~4% per train step from here. Same optimiser trajectory in expectation (same effective batch, same data order per pass).
- 2026-09-05: **r3-qentropy-8k stopped at 01:08 UTC: disk full** ("No space left on device" saving `[random_questions/step010/train]`'s checkpoint; 1.5 GB free of 246). Cause: keep-last-2 + one optimizer state is the right policy *during* an arm but nothing dropped them when an arm finished, so each of the 9 finished r1/r2 arms held 15 GB (round_08 weights 3.8 GB + round_09 weights 3.8 GB + optimizer_state.pt 7.5 GB) = 135 GB, plus kl50's 22 GB. Fix in apod/driver.py: `prune_finished(arm)` after the arm's last refresh keeps only the final weights (test_driver updated). The on-disk cleanup of the already-finished arms could not be done by the babysitting session (its permission classifier blocks `rm`); the USER runs it (command in the session transcript / handoff message), then the chain is relaunched and r3 resumes at that train pass (uncertain_questions arm complete, random_questions from step 10). Chain killed meanwhile so the retries do not burn GPU on a full disk.
