"""Paired problem-level bootstrap of the curve-mean difference (steps 15,30,45,60) vs random; also seed-2 partial qkl contrasts."""
import json, glob, random, statistics as st
from collections import defaultdict
R="outputs/runs_results/p2b"
G={"p2b-rollout-s1":["random","kl_high","kl_mid","kl_low","ent_top4"],"p2b-qkl-s1":["qkl_high","qkl_mid","qkl_low"],"p2b-correct-s1":["t_right_s_wrong","both_right","mixed","both_wrong"]}
def pp(run,arm,r):
    d=defaultdict(float)
    for f in glob.glob(f"{R}/{run}/arms/{arm}/rounds/round_{r:02d}/eval/eval.shard*.jsonl"):
        for l in open(f):
            x=json.loads(l); d[x["id"]]+= (x["correct"] and x["has_boxed"])/4
    return d
M={}
for run,arms in G.items():
    for a in arms:
        per=[pp(run,a,r) for r in (1,2,3,4)]
        ids=sorted(per[0]); M[a]={i:st.mean(p[i] for p in per) for i in ids}
ids=sorted(M["random"]); rng=random.Random(1); B=10000
out={}
for a in M:
    if a=="random": continue
    d=[M[a][i]-M["random"][i] for i in ids]; o=st.mean(d); bs=[]
    for _ in range(B): bs.append(sum(d[rng.randrange(500)] for _ in range(500))/500)
    bs.sort(); out[a]=(100*o,100*bs[250],100*bs[9750])
    print(f"{a:16s} {100*o:+5.2f} [{100*bs[250]:+.2f}, {100*bs[9750]:+.2f}]")
# question-lever vs rollout-lever spread
print("spread rollout-level (kl_*,ent):",round(max(out[a][0] for a in ["kl_high","kl_mid","kl_low","ent_top4"]+[])-min(min(out[a][0] for a in ["kl_high","kl_mid","kl_low","ent_top4"]),0),2))
json.dump(out,open("experiments/p2b-analysis/curve_ci.json","w"),indent=1)
# seed 2 partial: qkl_high - qkl_mid at steps 15,30,45 in s1 and s2
for f in ["p2b-qkl-s1","p2b-qkl-s2"]:
    E={}
    for l in open(f"{R}/{f}.metrics.jsonl"):
        x=json.loads(l); e=(x.get("eval") or {}).get("math500")
        if e: E[(x["arm"],x["step"])]=e["strict_avg_at_n"]
    dd=[100*(E[("qkl_high",s)]-E[("qkl_mid",s)]) for s in (15,30,45) if ("qkl_high",s) in E]
    print(f, "qkl_high-qkl_mid steps15/30/45:",[round(v,2) for v in dd],"mean %.2f"%st.mean(dd), " base step0 %.2f"%(100*E[("qkl_mid",0)]))
