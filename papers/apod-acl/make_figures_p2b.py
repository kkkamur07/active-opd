"""Figures for the rewritten paper (2B selection program, seed 1).

Run (from papers/apod-acl):
    python3 make_figures_p2b.py
Reads experiments/p2b-analysis/{s1_analysis.json, curve_ci.json} (built by analyze_s1.py and curve_ci.py from
outputs/runs_results/p2b/). Writes figures/{p2b_forest,p2b_curves,p2b_finish}.{pdf,png}.
Style: figures4papers house style (same constants as make_figures.py).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A = REPO / "experiments" / "p2b-analysis"
FIG = HERE / "figures"
S1 = json.load(open(A / "s1_analysis.json"))
CI = json.load(open(A / "curve_ci.json"))

PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B", "red_2": "#E9A6A1",
           "red_strong": "#B64342", "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E"}
plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
    "figure.constrained_layout.use": True})

NAME = {"kl_high": "KL: high", "kl_mid": "KL: middle", "kl_low": "KL: low",
        "ent_top4": "Entropy: high", "qkl_high": "Question KL: high", "qkl_mid": "Question KL: middle",
        "qkl_low": "Question KL: low", "t_right_s_wrong": "Teacher right, student wrong",
        "both_right": "Both right", "mixed": "Mixed", "both_wrong": "Both wrong", "random": "Random"}
GROUPS = [("Which answer", ["kl_high", "kl_mid", "kl_low", "ent_top4"], PALETTE["teal"]),
          ("Which question: alignment", ["qkl_high", "qkl_mid", "qkl_low"], PALETTE["blue_main"]),
          ("Which question: correctness", ["t_right_s_wrong", "both_right", "mixed", "both_wrong"], PALETTE["violet"])]
STEPS = ["0", "15", "30", "45", "60"]

def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300)
    plt.close(fig)

# ---- Fig 1: curve-mean difference vs random (steps 15-60), point estimates
fig, ax = plt.subplots(figsize=(3.2, 3.5))
y = 0; yt, yl, bold = [], [], []
for gname, arms, col in GROUPS:
    yt.append(y); yl.append(gname); bold.append(True); y -= 1
    for a in arms:
        o = CI[a][0]   # point estimate only (no bootstrap range, USER 2026-11-15)
        ax.plot([0, o], [y, y], color=col, lw=1.6, solid_capstyle="round")
        ax.plot(o, y, "o", color=col, ms=4.5)
        ax.text(3.45, y, f"{o:+.2f}", va="center", ha="right", fontsize=7)
        yt.append(y); yl.append(NAME[a].replace("Rollout ", "").replace("Question ", "")); bold.append(False); y -= 1
    y -= 0.4
ax.axvline(0, color=PALETTE["red_strong"], lw=1.0, ls="--")
ax.set_yticks(yt, yl); ax.set_xlim(-2.4, 3.5); ax.set_xticks([-2, -1, 0, 1, 2])
for t, b in zip(ax.get_yticklabels(), bold):
    if b: t.set_fontweight("bold")
ax.set_xlabel("Minus random (points)", loc="left")
ax.tick_params(axis="y", length=0)
save(fig, "p2b_forest")
print("fig1 values:", {a: CI[a] for _, arms, _ in GROUPS for a in arms})

# ---- Fig 2: learning curves (steps 15-60; base 26.5% at step 0 is in the caption), random in every panel
SHORT = {"kl_high": "KL high", "kl_mid": "KL middle", "kl_low": "KL low", "ent_top4": "Entropy high",
         "qkl_high": "KL high", "qkl_mid": "KL middle", "qkl_low": "KL low", "t_right_s_wrong": "T right, S wrong",
         "both_right": "Both right", "mixed": "Mixed", "both_wrong": "Both wrong"}
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
st_ = STEPS[1:]; steps = [int(s) for s in st_]
for ax, (gname, arms, _) in zip(axs, GROUPS):
    cols = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["violet"], PALETTE["green_3"]]
    r = [100 * S1["curves"]["random"][s]["strict"] for s in st_]
    ax.plot(steps, r, color=PALETTE["red_strong"], lw=1.6, ls="--", marker="o", ms=3, label="Random")
    for a, c in zip(arms, cols):
        v = [100 * S1["curves"][a][s]["strict"] for s in st_]
        ax.plot(steps, v, color=c, lw=1.4, marker="o", ms=3, label=SHORT[a])
    ax.set_title(gname, fontsize=8.5); ax.set_xticks(steps); ax.set_xlabel("Training step")
    ax.set_ylim(38.5, 48.8); ax.set_yticks([40, 42, 44, 46]); ax.legend(loc="upper left", fontsize=6.3, handlelength=1.4, ncol=2, columnspacing=0.8)
axs[0].set_ylabel("Strict avg@4 (%)")
save(fig, "p2b_curves")

# ---- Fig 3: strict accuracy vs finish rate, every arm x step (incl. base)
fig, ax = plt.subplots(figsize=(3.2, 2.3))
xs, ys = [], []
for gname, arms, col in GROUPS + [("Random", ["random"], PALETTE["red_strong"])]:
    for a in arms:
        for s in STEPS[1:]:
            c = S1["curves"][a][s]; xs.append(100 * c["finish"]); ys.append(100 * c["strict"])
            ax.plot(xs[-1], ys[-1], "o", color=col, ms=3.2, alpha=0.85)
b = S1["curves"]["random"]["0"]
ax.plot(100 * b["finish"], 100 * b["strict"], "s", color="black", ms=4)
ax.annotate("base", (100 * b["finish"], 100 * b["strict"]), xytext=(4, -2), textcoords="offset points", fontsize=7)
xa = np.array(xs + [100 * b["finish"]]); ya = np.array(ys + [100 * b["strict"]])
r_all = np.corrcoef(xa, ya)[0, 1]; r_tr = np.corrcoef(xs, ys)[0, 1]
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=g) for g, _, c in GROUPS] +
          [Line2D([], [], marker="o", ls="", color=PALETTE["red_strong"], label="Random")],
          loc="lower right", fontsize=6.3, handletextpad=0.2)
ax.set_xlabel("Finished within 8,192 tokens (%)"); ax.set_ylabel("Strict avg@4 (%)")
ax.text(0.03, 0.95, f"trained checkpoints: r = {r_tr:.2f}", transform=ax.transAxes, fontsize=7, va="top")
save(fig, "p2b_finish")
print(f"fig3: r(trained, n={len(xs)}) = {r_tr:.3f}; r(with base) = {r_all:.3f}")

# ---- Fig 0: method schematic (no data values): answer-level vs question-level selection
from matplotlib.patches import Rectangle
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.1))
rng = np.random.default_rng(3)
def panel(ax, title, nq, na, keep_rows, keep_cols, note, sort_cols):
    ax.set_xlim(0, 10); ax.set_ylim(-0.9, 6.4); ax.axis("off")
    ax.text(0.2, 6.2, title, fontsize=8.5, fontweight="bold", va="top")
    cw, ch = 9.6 / na, 4.6 / nq
    for i in range(nq):
        base = rng.uniform(0.25, 0.85)
        vals = np.clip(base + rng.normal(0, 0.07, na), 0, 1)
        if sort_cols: vals = np.sort(vals)[::-1]
        for j, v in enumerate(vals):
            kept = (i in keep_rows) and (j in keep_cols)
            ax.add_patch(Rectangle((0.2 + j * cw, 0.6 + (nq - 1 - i) * ch), cw * 0.88, ch * 0.8,
                                   facecolor=plt.cm.Blues(0.25 + 0.7 * v),
                                   edgecolor=PALETTE["red_strong"] if kept else "white", lw=1.5 if kept else 0.4))
    ax.text(0.2, -0.2, note, fontsize=7, va="center")
panel(axs[0], "Choosing answers", 6, 12, set(range(6)), {4, 5, 6, 7},
      "12 answers per question, sorted by KL (darker = higher);\nkeep 4 per question, e.g. the middle band", True)
panel(axs[1], "Choosing questions", 9, 4, {0, 3, 6}, set(range(4)),
      "4 answers per question; shade = question KL;\nkeep a third of the questions with all their answers", False)
save(fig, "p2b_method")

# ---- Fig 4: what the signals measure (round-0 pools, seed 1)
import json as _j
from collections import defaultdict as _dd
rows = [_j.loads(l) for l in open(REPO / "outputs/runs_results/p2b/p2b_rollouts_compact.jsonl")]
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
Q = _dd(list)
for r in rows:
    if r["run"] == "p2b-rollout-s1" and r["arm"] == "random" and r["round"] == 0 and "kl" in r:
        Q[r["ex"]].append(r["kl"])
order = sorted(Q, key=lambda q: np.mean(Q[q]))
for x, q in enumerate(order):
    v = np.array(Q[q])
    axs[0].plot([x, x], [v.min(), v.max()], color=PALETTE["neutral"], lw=0.8)
    axs[0].plot(x, v.mean(), "o", color=PALETTE["blue_main"], ms=1.8)
axs[0].set_xlabel("Question (sorted by mean KL)"); axs[0].set_ylabel("Answer reverse KL (nats)")
axs[0].set_xticks([])
axs[0].set_title("Answers of one question sit close together", fontsize=8)
B = _dd(list)
for r in rows:
    if r["run"] == "p2b-qkl-s1" and r["arm"] == "qkl_mid" and r["round"] == 0 and "kl" in r:
        B[r["ex"]].append(r)
qs = sorted(B.values(), key=lambda v: -np.mean([x["kl"] for x in v])); n = len(qs) // 3
thirds = [("high", qs[:n]), ("middle", qs[n:2 * n]), ("low", qs[2 * n:])]
buckets = [("both_wrong", "Both wrong", PALETTE["red_2"]), ("teacher_right_student_wrong", "T right, S wrong", PALETTE["blue_secondary"]),
           ("mixed", "Mixed", PALETTE["neutral"]), ("both_right", "Both right", PALETTE["green_3"])]
left = np.zeros(3)
for key, lab, col in buckets:
    vals = np.array([sum(q[0].get("b_bucket") == key for q in t) for _, t in thirds], float)
    axs[1].barh([0, 1, 2], vals, left=left, color=col, label=lab, height=0.6)
    left += vals
axs[1].set_yticks([0, 1, 2], [f"KL {t}" for t, _ in thirds]); axs[1].invert_yaxis()
axs[1].set_xlabel("Questions in the third"); axs[1].set_title("Question KL thirds by correctness", fontsize=8)
axs[1].legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=2, handlelength=1, columnspacing=0.8)
save(fig, "p2b_signals")
print("fig4 thirds buckets:", [{k: sum(q[0].get('b_bucket') == k for q in t) for k, _, _ in buckets} for _, t in thirds])
