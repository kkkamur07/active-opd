# Scripts

Run everything from the repository root with `python -m scripts.<path>` so
package imports and the Hydra configuration resolve consistently. Generated
reports, traces, metrics, and checkpoints belong under `outputs/`, never beside
source files.

## main.py

Resolves and prints the Hydra configuration. Loads no model and no dataset, so
it is safe to run anywhere.

```bash
python -m scripts.main
python -m scripts.main filtering=all precision=h100
```

## experiments/

`run_filter_ablation.py` compares selection policies at a matched training
budget. The three arms (`all`, `random`, `verified_wrong`) see byte-identical
rollouts each round, so any difference between them comes from selection rather
than from update count.

```bash
python -m scripts.experiments.run_filter_ablation --dry-run
python -m scripts.experiments.run_filter_ablation \
    --train-prompts 64 --rounds 20 --k 4 --select-budget 16 --eval-problems 200
```

The run exits non-zero if the arms did not consume matched budgets. Each run
writes `results.json`, per-arm `rounds.jsonl`, and `learning_curve.jsonl` to a
timestamped directory.

Start with `--dry-run`, which prints the rollout and optimizer-step budget the
run will consume without loading a model.

## diagnostics/

`smoke_test.py` runs one guarded CUDA training step on real models. Use it
before an experiment to confirm the stack works end to end and that parameters
actually move.

```bash
python -m scripts.diagnostics.smoke_test --run --max-new-tokens 256
```

## data/

`profile_traces.py` measures the token-length distribution of prompts,
references, and complete assistant traces in a corpus. This is how you pick
`max_new_tokens`: a budget below the trace-length p95 truncates most rollouts,
and a truncated rollout carries a length statistic rather than a reasoning
verdict.

```bash
python -m scripts.data.profile_traces --limit 256
```

`generate_traces.py` samples model responses for a bounded dataset slice and
writes them with their verification outcomes, which is useful for inspecting
what the verifier does to real traces.

`records.py` holds the lazy row helpers the two share. It reads raw dataset rows
rather than `MathExample`, because profiling needs to see rows that have no
usable ground-truth answer.
