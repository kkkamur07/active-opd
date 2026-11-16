set -e; cd ~/Desktop/active-opd
# correctness seed 3: seed-1 questions excluded, skip the 480 seed 2 takes, next 480 per bucket.
E=outputs/runs/p2b-rollout-s3/arms/random/rounds/round_00/eval
if [ -f $E/summary.json ]; then S0=$E; else S0=null; fi
HF_HUB_OFFLINE=1 .venv/bin/python -m apod.driver +experiment=p2b_correct output_dir=outputs/runs/p2b-correct-s3 run_name=p2b-correct-s3 seed=2042 driver.bank_dir=outputs/runs/bank-8k-w "+driver.exclude_questions_from=[outputs/runs/p2b-correct-s1]" driver.question_offset=480 driver.step0_eval_from=$S0
