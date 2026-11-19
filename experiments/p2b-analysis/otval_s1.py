# Seed-1 held-out (otval200: 200 OpenThoughts questions never trained on, 4 samples, 8k, strict) at steps 0/15/30/45/60.
# Pre-registered pick: checkpoint with best otval accuracy; then its MATH-500 accuracy (metrics.jsonl).
import json, glob, os
NAN=float("nan")
M={}
for g in ["rollout","qkl","correct"]:
    for l in open(f"p2b-{g}-s1/metrics.jsonl"):
        x=json.loads(l); e=(x.get("eval") or {}).get("math500")
        if e: M[(x["arm"],x["step"])]=100*e["strict_avg_at_n"]
print("arm | otval avg@4 at step 0/15/30/45/60 | best step | MATH-500 at best step | MATH-500 mean 15-60")
for g in ["rollout","qkl","correct"]:
    for ad in sorted(glob.glob(f"p2b-{g}-s1/terminal_eval/cap8192/arms/*")):
        arm=os.path.basename(ad); ov={}
        for r in range(5):
            f=f"{ad}/rounds/round_{r:02d}/terminal_summary.json"
            if os.path.exists(f): ov[15*r]=100*json.load(open(f))["datasets"]["otval200"]["avg_at_k"]
        if not ov: continue
        tr={k:v for k,v in ov.items() if k>0}; best=max(tr,key=tr.get) if tr else None
        mean=sum(M.get((arm,t),0) for t in (15,30,45,60))/4
        cells=" ".join("%.2f"%ov.get(t,NAN) for t in (0,15,30,45,60))
        print(f"{arm} | {cells} | {best} | {M.get((arm,best),NAN):.2f} | {mean:.2f}")
