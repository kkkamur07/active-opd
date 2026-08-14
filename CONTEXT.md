# Context: Active On-Policy Distillation (APOD)

Glossary of the project's language. Terms only — implementation lives in code and
`docs/`, decisions live in `docs/adr/`.

## Terms

**Student** — `Qwen/Qwen3.5-2B`. The model being trained. All trajectories used for
training are sampled from the student (on-policy).

**Teacher** — `Qwen/Qwen3.5-9B`. Frozen. Never sampled from; only provides per-token
log-probabilities used as the distillation target.

**Trajectory (trace)** — one sampled student completion for one prompt: the token
sequence and its metadata (lengths, sampling settings, grade).

**Rollout** — the act of sampling trajectories from the student for a prompt pool.
One prompt yields `num_rollouts` trajectories.

**Trajectory entropy `H(τ)`** — `mean_t Entropy(π_S(· | x, y_<t))` over response
positions: the student's average per-token predictive entropy along its own trace.
The uncertainty score used for active selection.

**Selection policy** — the rule that picks which trajectories from a rollout pool are
used for training. Selection is **trajectory-level**: applied per prompt, over that
prompt's own rollouts, so every prompt contributes to training in every arm. The
experiment compares: entropy top-k, random-k, and all (no selection). Prompt-level
selection (choosing which prompts deserve teacher effort at all) is deliberately out
of scope for this pass.

**Arm** — one full experimental run of the loop under one selection policy. Arms are
run sequentially and never share rollouts, because their students diverge after the
first round.

**Round** — one turn of the loop for one arm: eval → rollout → score → select →
train → checkpoint. The checkpoint is the weight hand-off between the trainer and
the next round's rollout engine.

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
