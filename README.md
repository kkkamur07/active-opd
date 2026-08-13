# Active On-Policy Distillation

Student `Qwen/Qwen3.5-2B`, frozen teacher `Qwen/Qwen3.5-9B`, OpenThoughts math, Math-Verify.

Qwen did not publish MATH-500 for these small models. Official thinking-mode scores:

| Benchmark | 2B student | 9B teacher | Gap |
| --- | ---: | ---: | ---: |
| HMMT Feb 25 | 22.9 | 83.2 | +60.3 |
| HMMT Nov 25 | 19.6 | 82.9 | +63.3 |
| PolyMATH | 26.1 | 57.3 | +31.2 |
| GPQA Diamond | 51.6 | 81.7 | +30.1 |

```bash
uv sync

# 512-example pool, no GPUs
uv run python -m scripts.collect_trajectories --select-only --num-examples 512

# 16 student traces + one teacher generation per prompt
uv run python -m scripts.collect_trajectories --num-examples 512 --num-rollouts 16 --resume
```

Output: `outputs/trajectories/{examples,trajectories}.jsonl`, `student/*.npz`, and `teacher/*.npz` (one teacher generation + logits per prompt).

```
apod/
  datasets/       load.py, io.py
  models/         load.py, student.py, teacher.py
  verification/   verify.py
```

