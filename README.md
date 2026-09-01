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

## The experiment

Which student trajectories are worth the teacher's attention? Each **arm** is a
selection policy; each **round** of an arm is one turn of the loop, run from the
arm's current checkpoint (the base student at round 0):

1. **Eval**: MATH-500, 4 samples per problem, strict `avg@4` (a response
   without `\boxed{}` is incorrect) with `pass@4` as the diversity monitor
   (ADR 0003). Reported through `scripts/eval_table.py`, never from the
   stage's console counts.
2. **Rollouts**: 12 student trajectories per prompt for the round's prompt
   slice (128 prompts per round in the Hydra config, 136 in the kl50 runs),
   capped at 8192 new tokens; cap-hit trajectories stay eligible for training
   and grade incorrect (ADR 0002). Same vLLM engine session as the eval.
3. **Scoring**: exact per-trajectory reverse KL `KL(pi_S || pi_T)` over the
   full vocabulary for all 12 rollouts, plus forward KL, both models' mean
   token entropy and the top-16 overlap statistics (`scripts/oracle_kl.py`).
4. **Selection**: per prompt keep 4 of 12 -- the reverse-KL tertile the arm
   names (`kl_high` ranks 1-4, `kl_mid` 5-8, `kl_low` 9-12) or a seeded
   `random` 4; the random arm is not scored after the shared round 0.
5. **Train**: one GKD pass (TRL `GKDTrainer`, `beta=1`, `lmbda=0`: pure
   reverse KL on the provided trajectories) over the 512-544 selected rows at
   effective batch 32, DDP across both GPUs, writing the next checkpoint.

Round 0 is shared across arms (identical base policy: one eval, one rollout
pool, one scoring pass, partitioned four ways). A final eval-only round measures
the last checkpoint. Full contract and file schemas: [docs/pipeline.md](docs/pipeline.md);
the vocabulary is in [CONTEXT.md](CONTEXT.md).

Sampling everywhere is `temperature=1.0, top_p=0.95, top_k=20`, no presence
penalty (ADR 0004): generation, scoring and the training objective share one
distribution.

vLLM generates and Hugging Face scores and trains. vLLM has no autograd and
cannot train; it is a rollout engine. The two run as separate processes: an
engine reserves 90-95% of the card for its KV cache, so the HF models will not
fit beside it.

## Setup

Machine setup, version choices, dataset limits, and rollout findings are in
[docs/guide.md](docs/guide.md).

```bash
uv sync --extra vllm --extra train
```

The environment must use the cu130 torch build selected by vLLM 0.26.0. The
reason and the CUDA troubleshooting steps are in [docs/guide.md](docs/guide.md#cuda-troubleshooting).

## Running

Two drivers, both resumable from on-disk markers:

```bash
# KL-bucket experiment (current): kl_high / kl_mid / kl_low / random.
uv run python scripts/bucket_experiment.py --kl50w     # 8192 cap, 3 rounds, LR cycles + persisted Adam state
uv run python scripts/bucket_experiment.py --report    # per-round table (scripts/eval_table.py)

# Entropy-selection design (entropy_top4 / random_top4 / all), Hydra-configured from conf/.
uv run python -m apod.main +experiment=smoke
uv run python -m apod.main
```

One process per GPU inside every stage; work is sharded by
`example_index % num_shards == shard`, each shard writes its own
`*.shard{k}.jsonl`, and the token npz files are keyed by example index, so
shards share one directory and nothing merges by hand. Stages are plain
scripts that read `<run_dir>/resolved_config.yaml`, so they can be launched by
hand for a single (arm, round, shard).

Logits are never materialised in full: the vocab is ~248k, so a 16k-token
float32 logit tensor is ~16 GB. Entropy and KL scoring run the decoder once and
apply the lm_head in position slices; training uses Liger's chunked fused JSD.

Measured stage costs and the ranked list of remaining throughput levers are in
[docs/perf_review.md](docs/perf_review.md).

## Layout

```
apod/
  main.py         Hydra driver (arms x rounds, resume, metrics.jsonl)
  stages/         rollout_eval.py (vLLM eval + rollouts), entropy.py, train.py (GKD)
  selection.py    entropy_top4 / random_top4 / all
  datasets/       load.py, io.py
  models/         load.py, generate_vllm.py, vllm_qwen35.py, presence_penalty.py (dormant, ADR 0004)
  verification/   verify.py (Math-Verify grading)
scripts/          bucket_experiment.py (KL-bucket driver), oracle_kl.py (scoring), lr_sweep.py,
                  lr_probe.py, eval_table.py, kl_drift.py, tb_export.py, check_run.py,
                  rollout_report.py, token_lengths.py, verify_*.py
conf/             Hydra config tree (model, data, rollout, sampling, engine, eval, selection, train)
docs/             pipeline.md, guide.md, perf_review.md, smoke_report.md, code_critique.md, adr/
```
