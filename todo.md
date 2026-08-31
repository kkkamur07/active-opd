# Todo

Working file. Phases are in order. See [docs/guide.md](docs/guide.md) for machine setup,
version choices, and rollout findings.

## Current: student trajectory collection

512 prompts, one trajectory per prompt from `Qwen/Qwen3.5-2B` via vLLM.

- [ ] Generate 512 student trajectories (1 rollout each)
- [ ] Measure throughput during the run
- [ ] Grade with math-verify against the dataset's `Answer` column

Being implemented now. Not yet run, so there are no numbers for any of the three.

## Next: teacher trajectory generation

- [ ] Generate `Qwen/Qwen3.5-9B` teacher trajectories over the same prompts, via vLLM,
      with the same sampling settings as the student
      (`temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5`)

Practical facts for this step:

- 9B in bf16 is roughly 18 GB of weights, so it fits on a single 80 GB A100 with plenty
  left for KV cache.
- The student engine must be torn down before the teacher engine is built. Each vLLM
  engine reserves about 90% of the card, so two engines do not coexist in one process.

## After that

Not being built yet.

- [ ] Student entropy scoring, `H(tau) = mean_t Entropy(pi_S(. | x, y_<t))`
- [ ] Trajectory-level top-k selection under the three arms: `entropy_top4`,
      `random_top4`, `all` (standard OPD, no selection)
- [ ] Iterative training loop: TRL GKDTrainer, reverse KL, checkpoint per round
      (see docs/adr/0001)

## Queued (user-requested 2026-08-14, behind smoke2 + vLLM verification)

- [ ] **Oracle-KL selection diagnostic — run ONCE, standalone.** (USER
      2026-08-14: explicitly NOT a real-run launch gate — run later as its own
      experiment; it remains the attribution tool for a null headline result.) Select top-k by the
      ACTUAL per-trajectory reverse KL between student and teacher instead of any
      entropy proxy. That is exactly the quantity the GKD objective minimizes, so it
      is the oracle selection rule and upper-bounds every cheap proxy (mean entropy,
      percentiles, margins). Cost is one teacher forward over all 12 rollouts per
      problem — too expensive per round, cheap as a one-off OFFLINE analysis over
      already-stored rollouts (smoke2 or real-run round 0; we keep the token npz
      files, no new rollout stage needed). Why it matters: it disambiguates a null
      result. If oracle-KL barely beats random_top4 → the selection premise itself is
      weak, learned for one round of compute instead of 8 rounds x 3 arms. If
      oracle-KL beats random substantially while entropy_top4 does not → entropy is
      the wrong proxy and the fix is a better statistic, not a different objective.

- [ ] **Cheap KL-divergence proxy for selection — affordable EVERY round.** Distinct
      from the oracle-KL diagnostic above: the oracle is a one-off ceiling
      measurement; this is a per-round selection signal. Problem: true reverse KL
      needs a teacher forward over all 12 rollouts per problem (too expensive per
      round); mean token entropy is affordable but demonstrably weak (0.0003 nats at
      the selection boundary). Wanted: something between.
      - [ ] **FIRST, nearly free: log per-trajectory reverse KL during training.**
            The train stage already computes teacher logits on the selected
            trajectories, so the true KL on the selected subset costs nothing —
            just log it. That gives labelled data to CALIBRATE proxies: regress
            each candidate statistic (mean/p90/max entropy, entropy variance,
            top1-top2 margin, early-window entropy) against actual KL and pick the
            selection statistic from evidence, not intuition. Caveat: this
            calibration set is biased (selected = high-entropy trajectories only);
            unbiased coverage of rejected trajectories is exactly what the one-off
            oracle run over stored rollouts provides — the two complement each other.
      - Teacher forward on a SPARSE SUBSET of positions (every Nth token, or only
        top-entropy positions) — an unbiased estimator of mean KL; only the
        ORDERING needs to be right, not the value, so few positions may suffice.
      - Top-k truncated KL (teacher's top-k logits only) — cheap to transfer/store;
        the tail contributes little to reverse KL since student mass concentrates.
      - Teacher forward on a PREFIX only — if early divergence predicts total
        divergence, a few hundred tokens may rank trajectories as well as the full
        sequence.
      - A smaller or quantized teacher purely as a RANKING model (real 9B teacher
        still used for the training objective).
      - Reuse cached teacher logits from the previous round where trajectories
        overlap.

- [ ] **Token-level entropy alongside trajectory-level, used together.**
      `apod/stages/entropy.py` currently collapses each trajectory to a single mean.
      That mean has almost no dynamic range within a problem: at round 0, problem 0's
      rank-4 vs rank-5 trajectories differed by 0.0003 nats — the selection boundary
      was pure noise, the best current explanation for entropy_top4 ~= random_top4.
      Token-level entropy has far more spread and is not washed out by averaging over
      1024–2048 tokens. Design space (record, don't decide): keep trajectory-level
      entropy for SELECTION but use token-level entropy to WEIGHT or MASK the GKD
      loss so the reverse KL concentrates where the student is actually uncertain
      (published RLVR work shows updating only the high-entropy minority of tokens
      captures most of the gain). Also store per-trajectory token-entropy percentiles
      (p50/p90/max) as alternative selection statistics. Storage: one float per token
      (2048 tok x 96 traj x 3 arms x rounds is small) — decide deliberately between
      full vectors and summary percentiles.

- [ ] **Selection statistics beyond mean entropy.** The mean over 2048 tokens is one
      summary of the token-entropy distribution and demonstrably a weak one (0.0003
      nats between rank 4 and 5). Candidates to evaluate against stored rollouts:
      p50/p90/p95/max token entropy; variance/spikiness of the profile ("forking
      points" vs uniformly diffuse uncertainty); count or fraction of tokens above an
      entropy threshold; top1-vs-top2 logit margin and count of low-margin positions;
      entropy over an early window only (divergence where the solution path is
      chosen); length-normalized vs unnormalized variants (length and entropy are
      coupled). Reference method for all of these: the oracle-KL diagnostic above.

- [ ] **Characterize (and possibly change) dataset difficulty.** Semantic
      verification measured teacher ppl 1.43 vs student ppl 1.53 on reference
      solutions — and that GAP is the distillation signal; if the teacher barely
      out-predicts the student there is little to transfer (consistent with the small
      measured reverse KL, 0.152). Nuance so this is not misread: low ppl on
      reference solutions does NOT mean the dataset is easy — rollout accuracy was
      4.7% and avg@n 0.0625, so the student cannot SOLVE these problems; teacher-forced
      ppl on formulaic LaTeX is a weak proxy for generation ability. The two numbers
      don't conflict, but neither answers the real question.
      - [ ] **FIRST: measure the teacher's own accuracy** (pass@1 and pass@n) on the
            dataset under the same sampling config, BEFORE committing GPU hours to
            the real run. On-policy distillation assumes the teacher is meaningfully
            better; if the 9B teacher solves only a small fraction, its guidance is
            mostly wrong and no objective tuning helps. Teacher accuracy is a
            CEILING: every arm result must be read against it in the report — if the
            teacher solves few problems, arm differences live inside a very small
            window and the report must say so out loud.
      - [ ] Break accuracy down by MATH-500 difficulty level (1–5) for student and
            teacher; pick the band where the teacher is strong and the student is
            weak — that is where distillation has headroom.

## Next training run: decisions (USER 2026-08-31)

- [ ] **No `all` control arm next run** (USER 2026-08-31: "this time we are not
      going to do the comparison with all"). The all-12-rollouts arm was the
      no-selection baseline in oracle16k (1536 rows, 48 steps/round, 3x the
      bucket arms' training compute); the next run trains selection arms only.
      Remove `all` from the run's ARMS tuple when building the driver.
- **Effective batch: stays 32, no change needed.** USER asked for 32 believing
  the current value was 16; verified 2026-08-31 that conf/train/gkd.yaml is
  already 2 per-device x 8 accum x 2 ranks = 32 (raised from 16 on
  2026-08-15, asserted via train.effective_batch). The "16" is the bucket
  arms' optimizer steps per round (512 trajectories / 32), not the batch.

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