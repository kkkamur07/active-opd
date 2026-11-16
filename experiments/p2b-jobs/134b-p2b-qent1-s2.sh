set -e; cd ~/Desktop/active-opd
# qent1 seed 2: question entropy from rollout 0 only; paired with p2b-qkl-s2 / p2b-qent-s2 (seed 1042, offset 1440); step-0 pool, scores and eval reused from p2b-qkl-s2
Q=outputs/runs/p2b-qkl-s2; test -f $Q/arms/qkl_mid/rounds/round_00/eval/summary.json || { echo "$Q step 0 missing"; exit 1; }
HF_HUB_OFFLINE=1 .venv/bin/python -m apod.driver +experiment=p2b_qent1 output_dir=outputs/runs/p2b-qent1-s2 run_name=p2b-qent1-s2 seed=1042 driver.question_offset=1440 "driver.step0_rollouts_from=[$Q]" driver.step0_eval_from=$Q/arms/qkl_mid/rounds/round_00/eval
