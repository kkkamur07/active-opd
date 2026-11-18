"""Seed 1 + seed 2 summary: mean strict avg@4 / pass@4 over steps 15-60, minus same-seed random. Point estimates only.
Also: qent_high vs counterfactual qkl_high overlap per refresh (from qent_high's own logged entropy + KL)."""
import json, glob, statistics as st
from collections import defaultdict
R="outputs/runs_results/p2b/"
runs={1:["p2b-rollout-s1","p2b-qkl-s1","p2b-correct-s1","p2b-qent-s1"],2:["p2b-rollout-s2","p2b-qkl-s2"]}
E={}
for seed,rs in runs.items():
    for r in rs:
        for l in open(R+r+".metrics.jsonl"):
            x=json.loads(l); e=(x.get("eval") or {}).get("math500")
            if e: E[(seed,x["arm"],x["step"])]=(100*e["strict_avg_at_n"],100*e["strict_pass_at_n"])
steps=[15,30,45,60]
def mean(seed,arm,k):
    v=[E.get((seed,arm,s)) for s in steps]
    return None if any(x is None for x in v) else st.mean(x[k] for x in v)
arms=["kl_high","kl_mid","kl_low","ent_top4","qkl_high","qkl_mid","qkl_low","qent_high","qent_mid","qent_low","t_right_s_wrong","both_right","mixed","both_wrong"]
print(f"{'arm':16s} {'s1 avg':>7s} {'s1 d':>6s} {'s2 avg':>7s} {'s2 d':>6s} {'mean d':>7s} | {'s1 pass d':>9s} {'s2 pass d':>9s}")
for seed in (1,2): print(f"random s{seed}: avg {mean(seed,'random',0):.2f} pass {mean(seed,'random',1):.2f}")
out={}
for a in arms:
    row=[]; ds=[]; pd=[]
    for seed in (1,2):
        m=mean(seed,a,0); r=mean(seed,"random",0)
        if m is None or r is None: row+=["   n/a","   n/a"]; pd.append("      n/a"); continue
        d=m-r; ds.append(d); row+=[f"{m:7.2f}",f"{d:+6.2f}"]; pd.append(f"{mean(seed,a,1)-mean(seed,'random',1):+9.2f}")
    md=f"{st.mean(ds):+7.2f}" if len(ds)==2 else "    n/a"
    print(f"{a:16s} {row[0]} {row[1]} {row[2]} {row[3]} {md} | {pd[0]} {pd[1]}")
    out[a]=ds
json.dump(out,open("experiments/p2b-analysis/multiseed.json","w"),indent=1)
# qent_high vs counterfactual qkl_high per refresh
def load(d,fn,key):
    o={}
    for f in glob.glob(f"{d}/{fn}.shard*.jsonl"):
        for l in open(f): x=json.loads(l); o[(x["example_index"],x["rollout_index"])]=x[key]
    return o
print("\nqent_high: overlap with the questions qkl_high would pick from the same pool (of 120)")
for rnd in range(4):
    d=f"{R}p2b-qent-s1/arms/qent_high/rounds/round_{rnd:02d}"
    Ent=load(d+"/entropy","entropy","entropy"); K=load(d+"/oracle","oracle_kl","mean_reverse_kl")
    if not Ent or not K: print(" round",rnd,"missing", len(Ent), len(K)); continue
    q=defaultdict(list)
    for k in Ent:
        if k in K: q[k[0]].append(k)
    ids=list(q); n=len(ids)//3
    te=set(sorted(ids,key=lambda i:-st.mean(Ent[k] for k in q[i]))[:n]); tk=set(sorted(ids,key=lambda i:-st.mean(K[k] for k in q[i]))[:n])
    print(f"  refresh {rnd}: {len(ids)} questions, overlap {len(te&tk)}/{n}")
