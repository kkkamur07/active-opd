"""Emit LaTeX rows for the paper from s1_analysis.json, curve_ci.json, signals.json (no hand-typed numbers)."""
import json, statistics as st
A="experiments/p2b-analysis/"
S=json.load(open(A+"s1_analysis.json")); CI=json.load(open(A+"curve_ci.json")); G=json.load(open(A+"signals.json"))
NAME={"random":"Random","kl_high":"KL high","kl_mid":"KL middle","kl_low":"KL low","ent_top4":"Entropy high",
 "qkl_high":"KL high","qkl_mid":"KL middle","qkl_low":"KL low","t_right_s_wrong":"Teacher right, student wrong",
 "both_right":"Both right","mixed":"Mixed","both_wrong":"Both wrong"}
GR=[("Baseline",["random"]),("Which rollout",["kl_high","kl_mid","kl_low","ent_top4"]),
    ("Which question: KL",["qkl_high","qkl_mid","qkl_low"]),("Which question: correctness",["t_right_s_wrong","both_right","mixed","both_wrong"])]
out=[]
out.append("% main results table rows: steps 15/30/45/60 strict avg@4 (%), curve mean, diff vs random [95% CI], finish@60")
for g,arms in GR:
    out.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{g}}}}} \\\\")
    for a in arms:
        c=S["curves"][a]; m=st.mean(100*c[s]["strict"] for s in ["15","30","45","60"])
        d="---" if a=="random" else f"{CI[a][0]:+.2f} [{CI[a][1]:+.2f}, {CI[a][2]:+.2f}]"
        if a!="random" and CI[a][1]>0: d="\\textbf{"+d+"}"
        if a!="random" and CI[a][2]<0: d="\\underline{"+d+"}"
        out.append(f"\\quad {NAME[a]} & " + " & ".join(f"{100*c[s]['strict']:.2f}" for s in ["15","30","45","60"]) +
                   f" & {m:.2f} & {d} & {100*c['60']['finish']:.1f} \\\\")
out.append("% trained-set table rows: mean over 4 refreshes of the 480 kept rollouts")
for g,arms in GR:
    out.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{g}}}}} \\\\")
    for a in arms:
        s_=S["selected"][a]; f=lambda k: st.mean(x[k] for x in s_)
        out.append(f"\\quad {NAME[a]} & {f('kl'):.3f} & {f('ent'):.3f} & {100*f('correct'):.1f} & {100*f('trunc'):.1f} & {f('len'):,.0f} \\\\")
out.append("% question thirds at round 0")
for k in ["qkl_high","qkl_mid","qkl_low"]:
    t=G["B_thirds"][k]; b=t["buckets"]
    out.append(f"{NAME[k]} & {t['qkl']:.3f} & {100*t['student_pass_now']:.1f} & {100*t['teacher_pass_bank']:.1f} & {100*t['trunc']:.1f} & {t['sent']:.2f} & {t['tent']:.2f} & {b.get('both_wrong',0)} & {b.get('teacher_right_student_wrong',0)} \\\\")
out.append("% KL bands at round 0")
for k,v in G["A_band_props"].items():
    out.append(f"{k} & {v['kl']:.3f} & {v['sent']:.3f} & {100*v['correct']:.1f} & {100*v['trunc']:.1f} \\\\")
open(A+"tex_rows.tex","w").write("\n".join(out)+"\n"); print("\n".join(out))
