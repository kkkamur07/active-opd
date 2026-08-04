# Pilot run, July 2026

Archived because `docs/code-review.md` cites it and `todo.md` has a task that
re-labels it. Qwen3-1.7B student, Qwen3-4B NF4 teacher, one MATH-500 problem,
K=8, `max_new_tokens=1024`, thinking enabled.

What it shows:

- All 8 responses are exactly 1024 tokens. None terminated. Seven contain
  neither `</think>` nor `\boxed`.
- The two rollouts labelled `wrong`, which were the entire Active OPD training
  pool, both answered `(3, π/2)` against a reference of
  `\left( 3, \frac{\pi}{2} \right)`. They were correct. The exact-match
  comparator could not equate the unicode and LaTeX forms.
- The standard arm took 8 optimizer steps and the active arm took 2, on
  byte-identical rollouts, while the run's own fairness block reported PASS.
- `accuracy: 1.0` in `results.json` is pass@8 over a single problem, computed
  from pre-training rollouts.

Re-scoring these files with the current verifier gives the old gate's
false-negative rate, which is the cheapest real measurement available here.
