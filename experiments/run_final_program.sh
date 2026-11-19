#!/usr/bin/env bash
# Reproduce the final program reported in the paper: the question bank, then
# 4 experiments x 3 seeds = 12 runs, with exactly the settings that were used.
# Needs 2 GPUs with 80 GB each. Every step resumes where it stopped, so the script
# can be re-run after an interruption. Results land in $RUNS (default outputs/runs).
#
#   bash experiments/run_final_program.sh
#   python experiments/paper_numbers.py --runs-dir outputs/runs --full
#   python paper/make_figures.py
set -euo pipefail
RUNS=${RUNS:-outputs/runs}
PY=${PY:-python}

pipeline() { "$PY" -m apod.pipeline "$@"; }

# Seeds: seed s draws fresh questions via the question offset (no question is reused).
SEED=(42 1042 2042)
OFFSET=(0 1440 2880)

# 1. Question bank: 4 student answers for 14,500 questions; teacher answers until every
#    correctness bucket holds 800 questions. Every rule draws from its teacher-labeled pool.
"$PY" -m apod.question_bank --bank-dir "$RUNS/bank-8k" --gpus 0,1

# 2. Widened bank for the correctness rule in seeds 2 and 3 (>= 1,440 questions per bucket).
[ -d "$RUNS/bank-8k-w" ] || cp -r "$RUNS/bank-8k" "$RUNS/bank-8k-w"
"$PY" -m apod.question_bank --bank-dir "$RUNS/bank-8k-w" --gpus 0,1 bank.student_questions=25000 bank.target_per_bucket=1440

for i in 0 1 2; do
  s=$((i + 1))
  common=(seed="${SEED[$i]}" driver.question_offset="${OFFSET[$i]}" driver.bank_dir="$RUNS/bank-8k")

  # 3. Answer level (standard OPD = the random arm). Makes the base-model evaluation.
  pipeline +experiment=answer_level run_name="answer-level-s$s" output_dir="$RUNS/answer-level-s$s" \
    "${common[@]}" driver.step0_eval_from=null

  # 4. Question level by reverse KL (seed 1 reuses the base evaluation of the answer-level run).
  base_eval=null
  [ "$s" = 1 ] && base_eval="$RUNS/answer-level-s1/arms/random/rounds/round_00/eval"
  pipeline +experiment=question_kl run_name="question-kl-s$s" output_dir="$RUNS/question-kl-s$s" \
    "${common[@]}" driver.step0_eval_from="$base_eval"

  # 5. Question level by student entropy (reuses question-kl's step-0 answers and scores).
  [ "$s" = 1 ] && qe_eval="$RUNS/answer-level-s1/arms/random/rounds/round_00/eval" \
               || qe_eval="$RUNS/question-kl-s$s/arms/qkl_mid/rounds/round_00/eval"
  pipeline +experiment=question_entropy run_name="question-entropy-s$s" output_dir="$RUNS/question-entropy-s$s" \
    "${common[@]}" "driver.step0_rollouts_from=[$RUNS/question-kl-s$s]" driver.step0_eval_from="$qe_eval"

  # 6. Question level by correctness. Seeds 2 and 3 use the widened bank and skip seed 1's questions.
  if [ "$s" = 1 ]; then
    pipeline +experiment=correctness run_name=correctness-s1 output_dir="$RUNS/correctness-s1" \
      "${common[@]}" driver.step0_eval_from="$RUNS/answer-level-s1/arms/random/rounds/round_00/eval"
  else
    pipeline +experiment=correctness run_name="correctness-s$s" output_dir="$RUNS/correctness-s$s" \
      seed="${SEED[$i]}" driver.question_offset=$(( (s - 2) * 480 )) driver.bank_dir="$RUNS/bank-8k-w" \
      "+driver.exclude_questions_from=[$RUNS/correctness-s1]" \
      driver.step0_eval_from="$RUNS/answer-level-s$s/arms/random/rounds/round_00/eval"
  fi
done
