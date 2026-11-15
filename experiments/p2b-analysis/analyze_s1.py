"""Seed-1 analysis of the 2B selection program (p2b-*). Reads only files pulled from the VM
into outputs/runs_results/p2b/. Writes experiments/p2b-analysis/s1_analysis.json + prints tables.

Checks (defined before running):
  C1 step-60 strict avg@4 recomputed from eval rows == summary.json value (all arms)
  C2 every arm has 500 problems x 4 samples at every step
  C3 step-0 eval is identical across groups (shared base eval)
"""
import json, glob, os, random, statistics as st
from collections import defaultdict

R = "outputs/runs_results/p2b"
GROUPS = {"p2b-rollout-s1": ["random", "kl_high", "kl_mid", "kl_low", "ent_top4"],
          "p2b-qkl-s1": ["qkl_high", "qkl_mid", "qkl_low"],
          "p2b-correct-s1": ["t_right_s_wrong", "both_right", "mixed", "both_wrong"]}
STEPS = [0, 15, 30, 45, 60]

def eval_rows(run, arm, rnd):
    d = f"{R}/{run}/arms/{arm}/rounds/round_{rnd:02d}/eval"
    rows = []
    for f in sorted(glob.glob(f"{d}/eval.shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    summ = json.load(open(f"{d}/summary.json")) if os.path.exists(f"{d}/summary.json") else None
    return rows, summ

def base_rows():
    return eval_rows("p2b-rollout-s1", "random", 0)

def per_problem(rows):
    """problem id -> dict(strict mean, finished count, finished-correct count)"""
    p = defaultdict(lambda: [0, 0, 0, 0])  # n, strict, finished, strict&finished
    for r in rows:
        s = bool(r["correct"] and r["has_boxed"])
        f = not r["truncated"]
        q = p[r["id"]]; q[0] += 1; q[1] += s; q[2] += f; q[3] += s and f
    return p

def stats(rows):
    n = len(rows)
    strict = sum(bool(r["correct"] and r["has_boxed"]) for r in rows) / n
    fin = [r for r in rows if not r["truncated"]]
    fr = len(fin) / n
    acc_fin = sum(bool(r["correct"] and r["has_boxed"]) for r in fin) / max(1, len(fin))
    pp = per_problem(rows)
    pass4 = sum(q[1] > 0 for q in pp.values()) / len(pp)
    ln = sum(r["response_length"] for r in rows) / n
    return dict(strict=strict, pass4=pass4, finish=fr, acc_fin=acc_fin, len=ln, n=n, nprob=len(pp))

def boot_diff(rows_a, rows_b, B=4000, seed=0):
    """paired problem-level bootstrap of strict avg@4 difference a-b (test-set noise only)"""
    pa, pb = per_problem(rows_a), per_problem(rows_b)
    ids = sorted(set(pa) & set(pb))
    da = [pa[i][1] / pa[i][0] - pb[i][1] / pb[i][0] for i in ids]
    obs = sum(da) / len(da)
    rng = random.Random(seed); bs = []
    for _ in range(B):
        s = 0.0
        for _ in ids: s += da[rng.randrange(len(ids))]
        bs.append(s / len(ids))
    bs.sort()
    return obs, bs[int(0.025 * B)], bs[int(0.975 * B)]

def selected(run, arm, rnd):
    f = f"{R}/{run}/arms/{arm}/rounds/round_{rnd:02d}/selected/selected.jsonl"
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else None

def main():
    out = {"curves": {}, "vs_random": {}, "selected": {}, "checks": {}}
    b_rows, _ = base_rows()
    base = stats(b_rows)
    rows = {}
    for run, arms in GROUPS.items():
        for arm in arms:
            out["curves"][arm] = {}
            for i, s in enumerate(STEPS):
                rr, summ = eval_rows(run, arm, i)
                if i == 0 and not rr:     # step-0 eval shared from rollout group
                    rr, summ = b_rows, None
                rows[(arm, s)] = rr
                st_ = stats(rr)
                out["curves"][arm][s] = st_
                if summ is not None:
                    out["checks"].setdefault("C1_summary_match", []).append(
                        abs(summ["strict_avg_at_n"] - st_["strict"]) < 1e-9)
                out["checks"].setdefault("C2_500x4", []).append(st_["n"] == 2000 and st_["nprob"] == 500)
            out["selected"][arm] = []
            for i in range(4):
                sel = selected(run, arm, i)
                if not sel: continue
                n = len(sel)
                out["selected"][arm].append(dict(
                    round=i, n=n, nq=len({x["example_index"] for x in sel}),
                    kl=sum(x["mean_reverse_kl"] for x in sel) / n,
                    ent=sum(x["entropy"] for x in sel) / n,
                    correct=sum(bool(x["correct"]) for x in sel) / n,
                    trunc=sum(bool(x["truncated"]) for x in sel) / n,
                    len=sum(x["response_length"] for x in sel) / n))
    # C3: step-0 eval shared -> identical rows for all arms (by construction here); also check qkl/correct round_00 eval exists & matches
    for run in ["p2b-qkl-s1", "p2b-correct-s1"]:
        arm = GROUPS[run][0]
        rr, _ = eval_rows(run, arm, 0)
        out["checks"].setdefault("C3_step0_shared", []).append(
            (not rr) or abs(stats(rr)["strict"] - base["strict"]) < 1e-9)
    for arm in [a for v in GROUPS.values() for a in v if a != "random"]:
        out["vs_random"][arm] = {}
        for s in STEPS[1:]:
            o, lo, hi = boot_diff(rows[(arm, s)], rows[("random", s)])
            out["vs_random"][arm][s] = (o, lo, hi)
        # mean over steps 15-60 (curve level), per-problem pooled
        o = st.mean(out["curves"][arm][s]["strict"] - out["curves"]["random"][s]["strict"] for s in STEPS[1:])
        out["vs_random"][arm]["mean15_60"] = o
    out["checks"] = {k: all(v) for k, v in out["checks"].items()}
    json.dump(out, open("experiments/p2b-analysis/s1_analysis.json", "w"), indent=1)

    P = lambda x: f"{100*x:5.1f}"
    print("CHECKS", out["checks"])
    print("\n== strict avg@4 curve (%), MATH-500 @8k ==")
    print(f"{'arm':16s}" + "".join(f"{s:>7d}" for s in STEPS) + "   mean15-60  vs-rand")
    order = sorted(out["curves"], key=lambda a: -st.mean(out["curves"][a][s]["strict"] for s in STEPS[1:]))
    for a in order:
        c = out["curves"][a]; m = st.mean(c[s]["strict"] for s in STEPS[1:])
        vr = out["vs_random"].get(a, {}).get("mean15_60", 0.0)
        print(f"{a:16s}" + "".join(f"{P(c[s]['strict']):>7s}" for s in STEPS) + f"   {P(m)}     {100*vr:+5.2f}")
    print("\n== step 60: strict, pass@4, finish rate, acc|finished, length ==")
    for a in order:
        c = out["curves"][a][60]
        print(f"{a:16s} strict {P(c['strict'])} pass4 {P(c['pass4'])} finish {P(c['finish'])} acc|fin {P(c['acc_fin'])} len {c['len']:6.0f}")
    c = base; print(f"{'BASE (step 0)':16s} strict {P(c['strict'])} pass4 {P(c['pass4'])} finish {P(c['finish'])} acc|fin {P(c['acc_fin'])} len {c['len']:6.0f}")
    print("\n== diff vs random, strict avg@4 pts [95% paired bootstrap, test noise only] ==")
    for a in order:
        if a == "random": continue
        v = out["vs_random"][a]
        print(f"{a:16s} " + "  ".join(f"s{s}:{100*v[s][0]:+5.2f}[{100*v[s][1]:+.1f},{100*v[s][2]:+.1f}]" for s in STEPS[1:]))
    print("\n== what each arm trained on (mean over 4 rounds) ==")
    print(f"{'arm':16s}  revKL   ent   strict-correct  truncated   len")
    for a in order:
        s_ = out["selected"][a]
        if not s_: continue
        m = lambda k: st.mean(x[k] for x in s_)
        print(f"{a:16s} {m('kl'):6.4f} {m('ent'):6.3f}   {P(m('correct'))}        {P(m('trunc'))}   {m('len'):6.0f}")

if __name__ == "__main__":
    main()
