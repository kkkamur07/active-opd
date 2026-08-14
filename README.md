# Active On-Policy Distillation

Student `Qwen/Qwen3.5-2B`, frozen teacher `Qwen/Qwen3.5-9B`, OpenThoughts math, Math-Verify.

Qwen did not publish MATH-500 for these small models. Official thinking-mode scores:

| Benchmark | 2B student | 9B teacher | Gap |
| --- | ---: | ---: | ---: |
| HMMT Feb 25 | 22.9 | 83.2 | +60.3 |
| HMMT Nov 25 | 19.6 | 82.9 | +63.3 |
| PolyMATH | 26.1 | 57.3 | +31.2 |
| GPQA for 2B, GPQA Diamond for 9B | 51.6 | 81.7 | +30.1 |

The model cards use different labels for the last row, so do not read it as a strict
like-for-like comparison without checking the benchmark definitions.

## This pass

A pre-filter experiment. It stops when the selected trace index sets are on disk. The
default pass does not run the teacher, acquire KL targets, or train the student.

Per prompt: sample 16 student traces, score each, keep the top k=4 under three policies
(`H+correct`, `H+incorrect`, `random`). N=128 prompts, seed 42.

vLLM generates and Hugging Face scores. vLLM has no autograd and cannot train; it is a
rollout engine. Run the two stages separately. The vLLM engine reserves 90% of the card
for its KV cache, so the Hugging Face model will not fit beside it.

## Setup

Machine setup, version choices, dataset limits, and rollout findings are in
[docs/guide.md](docs/guide.md).

```bash
uv sync --extra vllm
```

The environment must use the cu130 torch build selected by vLLM 0.26.0. The
reason and the CUDA troubleshooting steps are in [docs/guide.md](docs/guide.md#cuda-troubleshooting).

## Running

One process per GPU. Every shard rebuilds the same example list from the same seed and
keeps `example_index % num_shards == shard`, writing its own `*.shard{k}.jsonl`. The npz
files are keyed by example index, so shards share one directory and nothing merges by hand.

```bash
# 1. Length sweep: how long do traces run, how many hit the cap
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.token_lengths --shard 0 --num-shards 2 \
  --num-examples 64 --num-rollouts 1 --max-new-tokens 16384 --presence-penalty 1.5 &
CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.token_lengths --shard 1 --num-shards 2 \
  --num-examples 64 --num-rollouts 1 --max-new-tokens 16384 --presence-penalty 1.5 &
wait

# 2. Is a rollout pool worth filtering at all?
uv run python -m scripts.rollout_report

# 3. The pipeline itself (rollouts, entropy scoring, selection, GKD training,
#    eval) is driven by `python -m apod.main`.
```

`H(τ) = mean_t Entropy(π_S(· | x, y_<t))` over response positions. Logits are never
materialised in full: the vocab is ~248k, so a 16k-token float32 logit tensor is ~16 GB.
The body runs once and the lm_head is applied in time-slices.

## Layout

```
apod/
  datasets/       load.py, io.py
  models/         load.py, generate_vllm.py, vllm_qwen35.py, presence_penalty.py
  verification/   verify.py
scripts/          token_lengths.py, rollout_report.py, check_run.py, verify_*.py
docs/             guide.md, adr/
```
