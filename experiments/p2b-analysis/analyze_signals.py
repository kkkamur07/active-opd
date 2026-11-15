"""What do KL / entropy / correctness measure, and which property of the trained set tracks the outcome?
Input: outputs/runs_results/p2b/p2b_rollouts_compact.jsonl (built on the VM by /tmp/compact.py, joins
rollouts + oracle KL + bank labels) and experiments/p2b-analysis/s1_analysis.json.
"""
import json, statistics as st
from collections import defaultdict

def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5
    return num / den if den else float("nan")
spear = lambda a, b: pearson(rank(a), rank(b))

rows = [json.loads(l) for l in open("outputs/runs_results/p2b/p2b_rollouts_compact.jsonl")]
res = {}

# ---------- A. rollout group, round 0 (120 q x 12, base model) ----------
A = [r for r in rows if r["run"] == "p2b-rollout-s1" and r["arm"] == "random" and r["round"] == 0 and "kl" in r]
byq = defaultdict(list)
for r in A: byq[r["ex"]].append(r)
kl = [r["kl"] for r in A]; g = st.mean(kl)
between = sum(len(v) * (st.mean(x["kl"] for x in v) - g) ** 2 for v in byq.values())
total = sum((x - g) ** 2 for x in kl)
res["A_n"] = (len(A), len(byq))
res["A_between_frac"] = between / total
def within_spear(key):
    vals = []
    for v in byq.values():
        a = [x["kl"] for x in v]; b = [float(x[key]) for x in v]
        if len(set(a)) > 1 and len(set(b)) > 1: vals.append(spear(a, b))
    return st.mean(vals), len(vals)
res["A_within_spear_kl_len"] = within_spear("len")
res["A_within_spear_kl_correct"] = within_spear("correct")
res["A_within_spear_kl_sent"] = within_spear("sent")
# rollout-level: correct vs KL rank position
pos = defaultdict(list)
for v in byq.values():
    s = sorted(v, key=lambda x: -x["kl"])
    for i, x in enumerate(s): pos[i // 4].append(x)
res["A_band_props"] = {b: dict(correct=st.mean(x["correct"] for x in pos[i]), trunc=st.mean(x["trunc"] for x in pos[i]),
                               kl=st.mean(x["kl"] for x in pos[i]), sent=st.mean(x["sent"] for x in pos[i]))
                       for i, b in enumerate(["kl_high(1-4)", "kl_mid(5-8)", "kl_low(9-12)"])}
# entropy top4 vs kl top4 overlap
ov = []
for v in byq.values():
    a = {x["ri"] for x in sorted(v, key=lambda x: -x["kl"])[:4]}
    b = {x["ri"] for x in sorted(v, key=lambda x: -x["sent"])[:4]}
    ov.append(len(a & b))
res["A_overlap_enttop4_klhigh"] = st.mean(ov)
# question level: q-KL vs pass (12 rollouts) and teacher label
qk = [st.mean(x["kl"] for x in v) for v in byq.values()]
qp = [st.mean(x["correct"] for x in v) for v in byq.values()]
qt = [v[0].get("b_tg") for v in byq.values()]
qtr = [st.mean(x["trunc"] for x in v) for v in byq.values()]
res["A_q_spear_kl_pass"] = spear(qk, qp)
res["A_q_spear_kl_trunc"] = spear(qk, qtr)
ok = [i for i, t in enumerate(qt) if t is not None]
res["A_q_spear_kl_teacherpass"] = spear([qk[i] for i in ok], [qt[i] for i in ok])

# ---------- B. qkl group, round 0 (360 q x 4) ----------
Bq = defaultdict(list)
for r in rows:
    if r["run"] == "p2b-qkl-s1" and r["arm"] == "qkl_mid" and r["round"] == 0 and "kl" in r: Bq[r["ex"]].append(r)
qs = sorted(Bq.values(), key=lambda v: -st.mean(x["kl"] for x in v))
thirds = {"qkl_high": qs[:len(qs)//3], "qkl_mid": qs[len(qs)//3:2*len(qs)//3], "qkl_low": qs[2*len(qs)//3:]}
res["B_nq"] = len(qs)
res["B_thirds"] = {}
for k, v in thirds.items():
    bk = defaultdict(int)
    for q in v: bk[q[0].get("b_bucket")] += 1
    res["B_thirds"][k] = dict(qkl=st.mean(st.mean(x["kl"] for x in q) for q in v),
                              student_pass_now=st.mean(st.mean(x["correct"] for x in q) for q in v),
                              trunc=st.mean(st.mean(x["trunc"] for x in q) for q in v),
                              teacher_pass_bank=st.mean(q[0]["b_tg"] / 4 for q in v if q[0].get("b_tg") is not None),
                              student_pass_bank=st.mean(q[0]["b_sg"] / 4 for q in v if q[0].get("b_sg") is not None),
                              teacher_trunc_bank=st.mean(q[0]["b_tt"] / 4 for q in v if q[0].get("b_tt") is not None),
                              sent=st.mean(st.mean(x["sent"] for x in q) for q in v),
                              tent=st.mean(st.mean(x["tent"] for x in q) for q in v),
                              buckets=dict(bk))
qk = [st.mean(x["kl"] for x in q) for q in qs]
tp = [q[0].get("b_tg") for q in qs]; sp = [st.mean(x["correct"] for x in q) for q in qs]
ok = [i for i, t in enumerate(tp) if t is not None]
res["B_spear_qkl_teacherpass"] = spear([qk[i] for i in ok], [tp[i] for i in ok])
res["B_spear_qkl_studentpass"] = spear(qk, sp)
res["B_spear_qkl_gap"] = spear([qk[i] for i in ok], [tp[i] / 4 - sp[i] for i in ok])

# ---------- C. trained-set property vs outcome, across 11 arms (descriptive, n=11) ----------
S1 = json.load(open("experiments/p2b-analysis/s1_analysis.json"))
arms = [a for a in S1["vs_random"]] + ["random"]
y = [S1["vs_random"].get(a, {}).get("mean15_60", 0.0) for a in arms]
props = {}
for key in ["kl", "ent", "correct", "trunc", "len"]:
    x = [st.mean(s[key] for s in S1["selected"][a]) for a in arms]
    props[key] = (spear(x, y), pearson(x, y))
res["C_arms"] = arms
res["C_outcome_mean15_60"] = dict(zip(arms, y))
res["C_spear_pearson_with_outcome"] = props

json.dump(res, open("experiments/p2b-analysis/signals.json", "w"), indent=1)
for k, v in res.items(): print(k, "=", json.dumps(v) if not isinstance(v, float) else round(v, 3))
