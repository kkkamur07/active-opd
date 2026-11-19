# Active On-Policy Distillation

Code, configurations and results for *What Makes a Rollout Worth Correcting? Active On-Policy
Distillation for Mathematical Reasoning*. A Qwen3.5-2B student learns from a frozen Qwen3.5-9B
teacher on its own answers (on-policy distillation, reverse KL); a selection rule decides which
questions and answers receive the teacher's correction.

## Layout

```
apod/                     library
  pipeline.py             training loop: sample -> score -> select -> train, every 15 steps
  selection.py            selection rules (answer level and question level)
  question_bank.py        correctness labels (4 student + 4 teacher answers per question)
  grading.py              answer checking (Math-Verify, boxed answer required)
  data/                   datasets and JSONL I/O
  models/                 vLLM and Hugging Face model loading
  stages/                 GPU stages run as subprocesses:
    generate.py           answers + MATH-500 evaluation (vLLM)
    entropy.py            student token entropy
    teacher_kl.py         exact per-token reverse KL to the teacher
    train.py              on-policy distillation step (TRL GKD, beta=1, lambda=0)
  dry_run.py              CPU stand-ins for the GPU stages (tests, plan checks)
conf/                     Hydra configs; experiment/ holds the four experiments of the paper
experiments/
  run_final_program.sh    reproduces the question bank and all 12 runs
  paper_numbers.py        recomputes every number in the paper from the runs
results/
  runs/*/metrics.jsonl    per-run metrics of the 12 runs (enough for every table)
  paper_numbers.json      all numbers used by the paper
paper/                    LaTeX source, bibliography, figures and make_figures.py
tests/                    unit, end-to-end (dry run) and golden (paper numbers) tests
EXPERIMENTS.md            record of every run
```

## Setup

```bash
uv sync --extra train        # Python 3.12, CUDA 13 wheels (torch 2.11, vLLM 0.26, TRL 1.10)
uv pip install pytest
```

## Reproduce

```bash
bash experiments/run_final_program.sh                      # 2 x 80 GB GPUs; resumable
python experiments/paper_numbers.py --runs-dir outputs/runs --full
python paper/make_figures.py
cd paper && tectonic main.tex
```

Rebuilding the tables only needs the committed metrics:
`python experiments/paper_numbers.py --runs-dir results/runs --out /tmp/numbers.json`.

## One experiment

```bash
python -m apod.pipeline +experiment=question_entropy                       # real run
python -m apod.pipeline +experiment=question_entropy driver.dry_run=true \
    output_dir=/tmp/plan tracking.mode=disabled                            # CPU plan check
```

Experiments: `answer_level` (standard OPD and answer-level rules), `question_kl`,
`question_entropy`, `correctness`. Every run resumes where it stopped.

## Tests

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:anyio tests
```

`tests/test_paper_numbers.py` checks that every mean ± SD and per-seed value in `paper/main.tex`
equals what `experiments/paper_numbers.py` computes from `results/runs/`.

## Earlier experiments

Pilot runs and an abandoned 4B program preceded the final program (see `EXPERIMENTS.md`).
Their code and configurations are preserved at the git tag `v0-pilot`.
