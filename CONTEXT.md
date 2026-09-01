# Context: Active On-Policy Distillation (APOD)

Glossary of the project's language. Terms only — implementation lives in code and
`docs/`, decisions live in `docs/adr/`.

## Terms

**Student** — `Qwen/Qwen3.5-2B`. The model being trained. All trajectories used for
training are sampled from the student (on-policy).

**Teacher** — `Qwen/Qwen3.5-9B`. Frozen. Never sampled from; only provides per-token
log-probabilities used as the distillation target.

**Question** — one math problem from the training pool (OpenThoughts) or an eval set.
The word "prompt" is retired (2026-09-01); a question is what gets labelled, selected
and trained on.

**Question bank** — the labelled pool for the correctness experiment: every swept
question with its teacher and student rollouts, strict grades, truncation flags and
bucket label. Built once; arms draw their questions from it.

**Trajectory (trace)** — one sampled student completion for one question: the token
sequence and its metadata (lengths, sampling settings, grade).

**Rollout** — the act of sampling trajectories from a model for a set of questions.
One question yields `num_rollouts` trajectories.

**Trajectory entropy `H(τ)`** — `mean_t Entropy(π_S(· | x, y_<t))` over response
positions: the student's average per-token predictive entropy along its own trace.
The uncertainty score used for active selection.

**Question entropy `H(q)`** — the student's trajectory entropy `H(τ)` averaged over
the student's own rollouts of a question: the question-level uncertainty score used
for question selection (active learning's uncertainty sampling, taken from the model
being trained). Teacher entropy plays no part in selection; the teacher is the
oracle whose effort selection is meant to spend well.

**Selection policy** — the rule that picks what is used for training. Selection acts
at two levels, and the two compose:

- **Question selection** — which questions receive teacher effort and enter a
  training set at all. Not a policy of the first experiments (they used every
  question); introduced with correctness buckets (2026-09-01).
- **Trajectory selection** — for a question that is in, which of its own rollouts are
  trained on (e.g. kl_mid: the middle-KL third of 12 rollouts). Every kl50 result is a
  trajectory-selection result.

**No-selection baseline (random)** — keeping k of n rollouts uniformly at random is,
in distribution, the same as sampling only k rollouts and training on all of them:
both are k i.i.d. draws from the student. So "random-k" *is* standard OPD at the
same training budget, and a separate "all n rollouts" arm is retired (2026-09-01):
it either triples the budget or trains on fewer questions, and answers nothing that
random-k does not.

**Correctness bucket** — a question-level label from teacher and student strict
correctness at the run's cap. A model is *correct* on a question when at least 3 of
its 4 rollouts are strictly correct and *wrong* when at most 1 of 4 is. The teacher
label is fixed (frozen teacher); the student label comes from the base student's
rollouts at labelling time. Four buckets, each an arm of the correctness experiment:

- **TC/SW** — teacher correct, student wrong. The canonical distillation case.
- **TC/SC** — both correct.
- **TW/SW** — both wrong. At cap 8192 most teacher-wrong is teacher cap-hit, so this
  arm tests whether an unfinished teacher trace still transfers.
- **Mixed** — every labelled question in none of the three above: either model in
  between (2 of 4), and the near-empty teacher-wrong/student-correct cell.

**Arm** — one full experimental run of the loop under one selection policy. Arms are
run sequentially and never share rollouts, because their students diverge after the
first training step.

**Training step** — one optimizer update on 32 trajectories (the effective batch).
The unit of training budget and the x-axis of every plot. "Round" is retired
(2026-09-01): nothing is counted in rounds any more.

**Refresh** — what happens every 10 training steps: the current weights are
evaluated, and the next 10 steps' questions are rolled out from those weights so
training stays on-policy. A refresh is a cadence, not a budget unit.

**Strict correctness** — a trajectory is correct only if a `\boxed{}` answer is
present and Math-Verify accepts it. Cap-hit (truncated) trajectories with no boxed
answer are wrong, at every cap, for labels and for eval alike. Truncation is a
property of the regime we train and evaluate in, not an excuse to relabel.

**OPD (on-policy distillation)** — training the student on its own sampled
trajectories, minimising reverse KL to the teacher at each trace position.

**Reverse KL** — `KL(π_S || π_T)` evaluated on student-sampled tokens. The training
objective; mode-seeking, in contrast to forward-KL/SFT distillation.

**Grade** — Math-Verify's verdict of a trajectory's final answer against the
dataset's `Answer` column. Grading has a measured ceiling of ~97–98%, not 100%
(see [docs/guide.md](docs/guide.md#dataset-contract-and-limits)).

**avg@4** — mean accuracy over 4 sampled attempts per eval problem: the unbiased,
low-noise estimator of pass@1 (the expected single-attempt accuracy). The
experiment's primary metric. "pass@1" is not a separate metric here.

**pass@4** — fraction of eval problems where at least one of the same 4 attempts
is correct. Measures coverage/diversity; falling pass@4 under rising avg@4 is the
mode-collapse signature of reverse KL.
