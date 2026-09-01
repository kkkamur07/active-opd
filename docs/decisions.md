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
- **Per-step logging on the training batch**: loss, top-16 overlap ratio,
  overlap-token advantage, mean |H_S - H_T| per token.
- **Vocabulary**: "training step" (not round), "question" (not prompt), "refresh"
  for the every-10-steps eval + re-rollout.
- **Question bank**: student sweep ~14,500 questions, teacher only where its label
  decides a bucket; stored with grades, truncation, lengths; reusable.
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
