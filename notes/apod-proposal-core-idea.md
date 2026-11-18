# APOD proposal: core idea and hypotheses (user, 2026-11-19; condensed from the user's proposal text)

## Core idea (polished, approved framing)
Active On-Policy Distillation combines on-policy distillation (the student learns from teacher feedback on its own
reasoning) with active learning (teacher supervision goes only to the most informative rollouts).
Central idea: correct only what is worth correcting. The paper's question: what makes a rollout worth correcting?

## Why on-policy distillation (proposal §2.1)
Off-policy distillation trains on teacher solutions (x, y_T): distribution mismatch, because at inference the student follows
its own distribution. On-policy distillation lets the student sample y_S ~ pi_theta(.|x) and the teacher supervises the
states the student actually visits.

## Why select (proposal §2.2)
Not every rollout needs teacher supervision: some are already correct (little to add); some are wrong but close (most useful);
some are far off-track (hard to absorb); some involve teacher errors (should not be distilled).
NOTE (user 2026-11-19): do NOT lead with "on-policy distillation is costly"; lead with the active-learning idea.

## Hypotheses from the proposal (what makes a good sample, §5-7)
A good sample (1) reveals a weakness of the student and (2) is close enough to the teacher that the correction is learnable.
Four cases (§6): (1) student correct -> no teacher needed; (2) student wrong, teacher right, close -> most valuable;
(3) student wrong, far from teacher -> deprioritize; (4) teacher wrong -> filter out.
Signals (§7): correctness (verifier); student uncertainty; teacher-student similarity with an intermediate "teachable" band.
Proposal's uncertainty = final-answer entropy; ours = mean token entropy (state this difference).

## Out of scope for this paper (user 2026-11-19: do not include)
near-miss signal, diversity, prefix-level correction, self-training / answer losses, teacher-query budget sweep, SFT baseline,
combined acquisition score.
