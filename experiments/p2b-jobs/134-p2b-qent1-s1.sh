set -e; cd ~/Desktop/active-opd
# qent1 seed 1: question entropy from rollout 0 only; paired with p2b-qkl-s1 / p2b-qent-s1 (seed 42, offset 0); step-0 pool + scores reused from p2b-qkl-s1, step-0 eval = config default (as p2b-qent-s1)
test -f outputs/runs/p2b-qkl-s1/arms/qkl_mid/rounds/round_00/rollouts/done.shard1 || { echo "p2b-qkl-s1 step-0 pool missing"; exit 1; }
HF_HUB_OFFLINE=1 .venv/bin/python -m apod.driver +experiment=p2b_qent1 output_dir=outputs/runs/p2b-qent1-s1 run_name=p2b-qent1-s1
