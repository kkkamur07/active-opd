set -e; cd ~/Desktop/active-opd
# Widen the 2B bank for correctness seeds 2+3: every bucket >= 1,440 (3 seeds x 480).
# bank-8k-w = full copy of bank-8k (bank-8k itself is never written). Student sweep
# of pool positions < 25,000 (otval200 = positions 29,227+), then teacher sweeps only
# where a bucket below target can still be filled. Raise to 29,227 only if short.
B=outputs/runs/bank-8k-w
check() { .venv/bin/python - "$B" <<'PY'
import sys
from pathlib import Path
from apod import bank
c = bank.bucket_counts(bank.load_bank(Path(sys.argv[1])))
print("buckets", dict(c), flush=True)
sys.exit(0 if all(c[b] >= 1440 for b in bank.BUCKETS[:4]) else 1)
PY
}
HF_HUB_OFFLINE=1 .venv/bin/python -m apod.bank --bank-dir $B --gpus 0,1 bank.student_questions=25000 bank.target_per_bucket=1440
check && exit 0
echo "short after 25,000: widening to 29,227 (never otval200)"
HF_HUB_OFFLINE=1 .venv/bin/python -m apod.bank --bank-dir $B --gpus 0,1 bank.student_questions=29227 bank.target_per_bucket=1440
check
