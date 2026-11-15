# Independent check: question mean entropy (ENTROPY STAGE output = what a selection rule would use)
# vs question mean reverse KL (ORACLE output), for every (run, arm, round) pool on disk.
import json, glob, os, statistics as st
from collections import defaultdict
def rank(v):
    o=sorted(range(len(v)),key=lambda i:v[i]); r=[0.0]*len(v); i=0
    while i<len(v):
        j=i
        while j+1<len(v) and v[o[j+1]]==v[o[i]]: j+=1
        for k in range(i,j+1): r[o[k]]=(i+j)/2
        i=j+1
    return r
def pear(a,b):
    ma,mb=st.mean(a),st.mean(b); n=sum((x-ma)*(y-mb) for x,y in zip(a,b)); d=(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**.5; return n/d
sp=lambda a,b: pear(rank(a),rank(b))
def load(d,fname,key):
    out={}
    for f in glob.glob(f"{d}/{fname}.shard*.jsonl"):
        for l in open(f):
            x=json.loads(l); out[(x["example_index"],x["rollout_index"])]=x[key]
    return out
print(f"{'pool':42s} {'nq':>4s} {'rho(qEnt,qKL)':>13s} {'top3 overlap':>13s} {'chance':>6s} {'rho(entStage,oracleEnt) rollout':>32s}")
seen=set()
for run in ["p2b-qkl-s1","p2b-qkl-s2","p2b-rollout-s1","p2b-correct-s1"]:
    for d in sorted(glob.glob(f"{run}/arms/*/rounds/round_0[0-3]")):
        if not (glob.glob(f"{d}/entropy/entropy.shard*.jsonl") and glob.glob(f"{d}/oracle/oracle_kl.shard*.jsonl")): continue
        E=load(d+"/entropy","entropy","entropy"); K=load(d+"/oracle","oracle_kl","mean_reverse_kl"); O=load(d+"/oracle","oracle_kl","student_entropy")
        keys=sorted(set(E)&set(K))
        sig=(run,tuple(keys[:3]),len(keys))
        if sig in seen: continue    # shared step-0 pool copied into several arms
        seen.add(sig)
        q=defaultdict(list)
        for k in keys: q[k[0]].append(k)
        ids=sorted(q); qe=[st.mean(E[k] for k in q[i]) for i in ids]; qk=[st.mean(K[k] for k in q[i]) for i in ids]
        n=len(ids)//3; te=set(sorted(ids,key=lambda i:-qe[ids.index(i)])[:n]); tk=set(sorted(ids,key=lambda i:-qk[ids.index(i)])[:n])
        be=set(sorted(ids,key=lambda i:qe[ids.index(i)])[:n]); bk=set(sorted(ids,key=lambda i:qk[ids.index(i)])[:n])
        ro=sp([E[k] for k in keys if k in O],[O[k] for k in keys if k in O])
        print(f"{d:42s} {len(ids):4d} {sp(qe,qk):13.3f} {len(te&tk):6d}/{n:<3d} low {len(be&bk):3d}/{n:<3d} {n*n/len(ids):6.0f} {ro:20.3f}")
