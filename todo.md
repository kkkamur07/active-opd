# Follow-up research

The prototype keeps the verification gate and the distillation objective
separate. Selection is currently a hard filter; the acquisition score is not
built yet, and building it is deliberately blocked on the first item below.

## Next

- Run the budget-matched ablation: `all` vs `random` vs `verified_wrong` at
  equal optimizer steps and equal trained tokens. If matched-budget random
  matches `verified_wrong`, selection does not help and nothing downstream of
  it can be true. This is the kill switch, so run it first.
- Verify the teacher's own answers before using its distributions. Selecting
  student-wrong rollouts conditions on difficulty, which conditions on
  teacher-wrong, so the selected pool is where the teacher is least reliable.
  One offline pass with K=4 per training problem gives a `teacher_pass_rate`
  that both gates the pool and stands as a result on its own.
- Re-label the archived traces in `outputs/` with the current verifier and
  report the old exact-match gate's false-negative rate. Costs no GPU.

## Then

- Add teacher token likelihood as an acquisition signal. It is already computed
  inside the loss and thrown away; cache it onto the rollout instead. Fix the
  sign before running: high teacher likelihood means the teacher agrees with the
  student's path, which means near-zero KL and near-zero gradient.
- Score near misses rather than retaining every verified-wrong rollout equally.
  Gated on whether the rollout's group contains at least one correct answer,
  this separates "wrong but close" from "wrong and far off track", which is the
  distinction correctness alone cannot make.
- Group disagreement across the K rollouts of a problem, used as a
  problem-level weight. At K=8 the per-rollout estimate has a standard
  deviation near 0.18, which is larger than most differences worth ranking on.
- Diversity control over selected answers within a round.
- Compare the exact reverse KL against the sampled and top-k estimators now
  that unknown estimator names raise instead of silently resolving to k3.

## Infrastructure

- Serve generation from vLLM, refreshed once per round. Generation is around 87
  percent of wall clock, so this is the largest single speedup available.
- Fuse the LM head into the loss (Liger or cut-cross-entropy) so full-vocabulary
  logits are never materialized. Chunking bounds the softmax memory but the
  logit tensors themselves remain, at roughly 9 GiB for both models at 16k
  tokens.
- Checkpoint resume. Nothing loads a checkpoint today, so an interrupted run
  restarts from scratch.
- Multiple seeds per arm. Single-seed comparison cannot support a claim about
  learning speed, and the active arm's variance is structurally larger because
  its selection depends on the student it is training.
