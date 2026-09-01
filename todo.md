# Todo

Working file. Done items carry a pointer to where the work lives (file or
commit); the current loop is described in [docs/pipeline.md](docs/pipeline.md)
and driven by `scripts/bucket_experiment.py`. Machine setup, version choices,
and rollout findings are in [docs/guide.md](docs/guide.md).

Sampling in every run: `temperature=1.0, top_p=0.95, top_k=20`,
presence_penalty **0.0** (ADR 0004; the 1.5 once written here was never used).

## Done: pipeline build-out (2026-08-13 .. 08-15)

- [x] Student trajectory collection and length/throughput/grading measurements
      -- `scripts/token_lengths.py`, `scripts/rollout_report.py` (commits
      686a593, c56de70); findings in docs/guide.md "Rollout findings".
- [x] Teacher trajectories over the same prompts, same sampling -- the
      `TEACHER_RUN` block of `scripts/bucket_experiment.py` (128 x 4 at 16384,
      oracle16k; commit eccd79d). Dropped from kl50/kl50w as a diagnostic
      (USER 2026-08-31: no teacher-rollout block); only the teacher-ceiling
      eval remains (`KL50_TEACHER`, eval-only).
- [x] Student entropy scoring `H(tau)` -- `apod/stages/entropy.py` (commit 0003b0d).
- [x] Trajectory-level top-k selection, arms `entropy_top4` / `random_top4` /
      `all` -- `apod/selection.py`, `conf/config.yaml`.
- [x] Iterative loop: TRL GKDTrainer, reverse KL, checkpoint per round --
      `apod/main.py` + `apod/stages/train.py`, ADR 0001; results in
      docs/smoke_report.md and outputs/runs/apod.
- [x] **Oracle-KL selection diagnostic** (exact per-trajectory reverse KL
      between student and teacher; the quantity the beta=1 GKD objective
      minimizes, so it upper-bounds every cheap proxy) -- `scripts/oracle_kl.py`
      (b271920; forward KL + entropies c6a6adf; top-16 overlap ratio /
      advantage e19da4a). Ran once on the apod round-0 pool as planned, then
      became the selection rule itself: the KL-bucket arms (kl_high / kl_mid /
      kl_low vs random) in oracle16k, oracle8k/4k, kl50, kl50w.
- [x] Measure the teacher's own accuracy under the same sampling BEFORE
      committing GPU hours -- `run_rollout_eval(KL50_TEACHER, ..., eval_only=True)`
      in `scripts/bucket_experiment.py`; outputs/runs/kl50_teacher (avg@4 0.6115 at
      8192) and outputs/runs/oracle16k_teacher. Every arm result is
      read against this ceiling.

## Open research items

- [ ] **Cheap KL-divergence proxy for selection, affordable every round.**
      The oracle scoring (`scripts/oracle_kl.py`) IS the per-round selection
      signal today at ~27 min per round-arm (1632 trajectories at cap 8192;
      docs/perf_review.md). Every round's full pool is scored, so the
      calibration data for any proxy already exists unbiased (rejected
      trajectories included) -- regress candidate statistics against the
      stored exact KL, per prompt, and adopt one only if per-prompt Spearman
      and tertile agreement hold. Candidates, none evaluated yet:
      - **MC reverse KL** -- TODO ONLY, explicitly NOT this run (USER
        2026-08-31: "a lot of them just do mc reverse kl -- add this todo but
        we are not doing this"). Sampled-token estimate
        `log pi_S(y_t) - log pi_T(y_t)` averaged over the trajectory: student
        side free from vLLM at generation time, teacher side one vLLM prefill
        pass (`prompt_logprobs`) at ~2-3x the current scoring throughput.
        CAVEAT: not an unbiased estimator of mean KL (sampling is
        top_k/top_p-truncated), so it changes the selection statistic --
        validate offline against the stored exact KL first
        (`oracle_kl.py --analyze` has the machinery).
      - Teacher forward on a sparse subset of positions (every Nth token or
        top-entropy positions only): only the ordering has to be right.
      - Top-k truncated KL (teacher's top-k logits only); teacher forward on a
        prefix only; a smaller/quantized teacher purely as a ranking model.
      - Entropy-profile statistics (p50/p90/max, variance, count above a
        threshold, top1-top2 margin, early-window entropy, length-normalized
        variants): the mean over ~8k tokens has almost no within-prompt range
        (0.0003 nats between rank 4 and 5 at round 0), which is why
        entropy_top4 ~= random_top4 in the apod run.

- [ ] **Token-level entropy alongside trajectory-level.** `apod/stages/entropy.py`
      collapses each trajectory to one mean; `scripts/oracle_kl.py` records
      only mean student/teacher entropy per trajectory. Design space (record,
      don't decide): keep trajectory-level selection but use token-level
      entropy to WEIGHT or MASK the GKD loss so the reverse KL concentrates
      where the student is uncertain (RLVR work: updating only the
      high-entropy minority of tokens captures most of the gain). Storage is
      small (one float per token); decide between full vectors and summary
      percentiles deliberately.

- [ ] **Characterize (and possibly change) dataset difficulty.** Teacher-forced
      ppl on reference solutions (teacher 1.43 vs student 1.53) is a weak
      proxy: rollout accuracy is ~12-20% strict, so the student cannot SOLVE
      these problems, and the teacher solves 61% (avg@4, 8192) --
      arm differences live inside that window and every report must say so.
      - [ ] Break accuracy down by MATH-500 difficulty level (1-5) for student
            and teacher; pick the band where the teacher is strong and the
            student weak. Needs the `level` column carried through
            `apod/datasets/load.py` (`COLUMNS["math500"]` drops it today) into
            `pool/eval_problems.jsonl`.

## Run decisions (USER 2026-08-31 / 09-01) -- implemented

- [x] **kl50**: kl_high / kl_mid / kl_low / random at cap 8192, 136 prompts x
      12 rollouts, 544 selected rows = 17 steps/round x 3 rounds = 51 steps at
      effective batch 32 -- `--kl50` in `scripts/bucket_experiment.py`
      (commit 4290a4e). Supersedes the earlier "kl_mid vs random only" note
      (USER 2026-08-31 late: all three KL buckets plus random).
- [x] **No `all` control arm** (USER: "not going to do the comparison with
      all") -- `ARMS = ("kl_high", "kl_mid", "kl_low", "random")` in
      `bucket_experiment.py:main`.
- [x] **Effective batch 32** was already the setting (2 ranks x per-device x
      accum; asserted by `train.effective_batch` in `apod/stages/train.py`).
      The "16" USER remembered was the bucket arms' optimizer steps per round.
- [x] **Optimizer state persisted across rounds + LR cycles** (USER
      2026-09-01: "keep the optimizer states as well ... consistent run with lr
      cycles") -- `--kl50w` + `train.persist_optimizer` (`apod/stages/train.py`:
      name-keyed Adam state saved per arm per round, remapped on load so a
      restart works at any round boundary; the driver keeps one
      `optimizer_state.pt` per arm). Per-round warmup(1) +
      `cosine_with_min_lr(min_lr_rate=0.1)`; peak LR 5e-6 from the
      3-candidate probe (`outputs/runs/kl50w/lr_probe.json`); the per-arm
      optuna sweep (`scripts/lr_sweep.py`, 10 trials/arm over [1e-6, 1e-2])
      writes the common-LR verdict to the same file. Round 0 banked from
      kl50. Diagnosis that motivated it: a fresh trainer per round reset
      Adam's moments, and the t=1 bias-corrected update produced the
      measured step-2 loss/grad-norm spike each round (kl50: grad_norm 0.56
      -> 6.12 at the r1 restart); warmup-trained kl50 artifacts kept in
      outputs/runs/kl50_warmup_backup/.

## Infrastructure notes (2026-08-14)

- Boot disk resized 74 GB -> 246 GB online (growpart + resize2fs, mid-run, no
  interruption). Checkpoint retention (`keep_checkpoints`) raised 1 -> 5
  accordingly (keep-LAST-5 per arm; see conf/config.yaml comment for why not
  keep-best-5).
- **Two unmounted 375 GB local NVMe SSDs** (`nvme0n1`, `nvme0n2`) sit idle on
  this box — 750 GB of fast local storage. USER DECISION 2026-08-14:
  "configure checkpoint storage and use the nvme ssds effectively — but maybe
  later when everything stabilizes." Explicitly deferred; do not act before
  stabilization. **EPHEMERAL on GCP: contents vanish on instance
  stop/migration/preemption — never park anything there that must survive.**
  Sensible split when the time comes: high-churn ephemeral-safe data on NVMe
  (vLLM compile cache, HF model cache, scratch, in-flight checkpoints
  promoted to the boot disk once a round completes); everything durable stays
  on /dev/sda1. RAID0 across the pair only if I/O MEASURES as a bottleneck —
  no evidence of that yet (measured bottlenecks were compile-cache misses and
  GPU compute, both fixed); do not optimize on speculation.

## Later (explicitly deferred)

- [ ] **Prompt-level selection variant**: score prompts (e.g. mean entropy over their
      rollouts), spend teacher effort only on the top-k most uncertain prompts, train
      on all their rollouts. The classic active-learning claim. Deferred from the
      2026-08-14 planning session to keep the first experiment trajectory-level only.


### Skills : 

To review the changes : meaning that okay a beautiful way to review the essential changes and the assumptions, essentially what was changed and why ?
Grill with docs for ML research with exploration built it, currently these models don't help you build the intuition and search up the spaces more ? 

### Dimensionality : 

- Correctness : Teacher should be correct and student should be wrong. Selection on these 4 combinations. Maybe the most effective one will be teacher is correct is student is wrong. 
Trick : 3000 mathematical questions, ask the teacher and student to generate the answers 4 rollouts for each questions and increase the max token generation as 32000 -> this is just not possible and categorize them in 4 buckets - we can easily find 800 questions. Student is wrong if it is wrong for 3 times or 4 times and then the same applies to teacher as well. What is wrong and what is correct. 

- Distributional Similarity : Distributional similarity is very important
      - We need to test on other evaluations as well 100 training steps per step and evaluations should be done per 10 steps with logging the overlap ratio and training losses per training step as well. 
      - AIME evaluations as well. 

- Entropy : For selection of the trajectories. Entropy incentivizes to select RKL we need to find this and run more experiments on this. 

- Diversity of the student rollouts : For each type I think we need is covered. 

#### Some questions to explore as well. 
- Which question to train on is also a good concern, for the questions - this would be an idea to try. Certainity on the question would be a good dimensionality to look at. 
- A way to combine all 3 indicators to understand what would be a good sample to learn from -> I need to do research on this as well. 

### Literature Review as well. 
- Other dimensionality : Data selection and active learning papers as well - do some literature search please. 

Presentations : Hinrich doesn't have enough background on OPD, you may want to talk about further intuition, spend several minutes on the background. You maybe want to give more intuition and give clearn definition of what it is. Explanation of the results ( consistent terms ) -> A professor is a professor because judgement has to be good -> What is the motivation of doing this. We are designing better training strategy because OPD 