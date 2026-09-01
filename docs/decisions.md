# Decisions log

Dated decisions from planning sessions, one line each, newest first. Reasoning
lives in the linked ADR or doc; vocabulary in CONTEXT.md.

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
