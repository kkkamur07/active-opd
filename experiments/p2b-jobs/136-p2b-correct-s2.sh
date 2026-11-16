set -e; cd ~/Desktop/active-opd
# correctness seed 2: widened bank, every seed-1 question excluded, first 480 of the rest per bucket.
# Step-0 eval reused from the seed-2 random arm (same seed, same base model).
E=outputs/runs/p2b-rollout-s2/arms/random/rounds/round_00/eval; test -f $E/summary.json || { echo "$E missing"; exit 1; }
HF_HUB_OFFLINE=1 .venv/bin/python -m apod.driver +experiment=p2b_correct output_dir=outputs/runs/p2b-correct-s2 run_name=p2b-correct-s2 seed=1042 driver.bank_dir=outputs/runs/bank-8k-w "+driver.exclude_questions_from=[outputs/runs/p2b-correct-s1]" driver.question_offset=0 driver.step0_eval_from=$E
