# Do cheap statistics rank trajectories like the exact reverse KL does?

Date: 2026-09-01. Zero-compute analysis over stored oracle scores
(`oracle/oracle_kl.shard*.jsonl` in kl50 and oracle16k: 23,712 trajectories,
1,848 question-pools of 12 rollouts; overlap fields exist on 11,424 rows).

Per question, each candidate statistic ranks that question's 12 rollouts; we
compare the ranking with the exact per-trajectory mean reverse KL (what kl_mid
selects on). Tertile agreement = fraction of rollouts placed in the same
high/mid/low third as the KL puts them (chance 0.33).

| statistic | needs teacher? | median per-question Spearman (IQR) | tertile agreement |
| --- | --- | ---: | ---: |
| student entropy H(τ) | no | +0.73 (+0.55..+0.85) | 0.60 |
| abs entropy gap \|H_S − H_T\| | yes | +0.55 (+0.30..+0.76) | 0.52 |
| forward KL | yes | +0.92 (+0.85..+0.96) | 0.78 |
| −overlap-token advantage (top-16) | yes | +0.97 (+0.96..+0.99) | 0.90 |
| −overlap ratio (top-16) | yes | −0.25 (−0.49..+0.03) | 0.28 |
| response length | no | −0.02 (−0.25..+0.21) | 0.33 |

Pooled (across questions) Spearman: entropy vs KL +0.76, abs gap +0.57,
length −0.24.

Readings:

- Student entropy, which needs no teacher forward, recovers the KL ordering
  within a question at 0.73 and puts 60% of rollouts in the right third. It is
  a usable proxy for a kl_mid-style rule, not the noise the round-0 rank-4/5
  gap (0.0003 nats on one question) suggested — that gap was one boundary, not
  the ordering.
- Length is uncorrelated with KL within a question: the kl_mid result is not
  a length effect.
- Overlap-token advantage tracks KL almost perfectly but is computed from the
  same teacher forward, so it is not a cheaper proxy; the overlap *ratio* is
  not a ranking signal at all.
- Open: whether an entropy-mid selection reproduces the kl_mid gain. That is a
  training experiment, not an offline one.
