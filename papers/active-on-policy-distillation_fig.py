"""Figures for papers/active-on-policy-distillation.md, in the figures4papers house style (palette + publication rcParams as in
papers/apod-acl/make_figures.py). Run from the repo root:
    uv run --no-project --with matplotlib python papers/active-on-policy-distillation_fig.py
fig1 = copy of papers/apod-acl/figures/p2b_method.png (schematic, no data; made by papers/apod-acl/make_figures_p2b.py).
fig2 data: experiments/p2b-analysis/three_seeds_exact.txt (per-seed values and exact 3-seed means, copied verbatim below).
fig3 data: experiments/p2b-analysis/variance_all.txt (between-question variance share, 12 answer-level pools).
"""
import shutil
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
P = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "teal": "#42949E", "red_strong": "#B64342", "neutral": "#CFCECE"}
plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2, "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})
OUT = "papers/active-on-policy-distillation_fig"
def save(fig, n):
    for ext in ("png", "pdf"): fig.savefig(f"{OUT}{n}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

shutil.copy("papers/apod-acl/figures/p2b_method.png", f"{OUT}1.png")
shutil.copy("papers/apod-acl/figures/p2b_method.pdf", f"{OUT}1.pdf")

# fig2: (level, label, seed1, seed2, seed3, exact 3-seed mean) from three_seeds.txt
rows = [("q", "Teacher right, student wrong", +0.90, +1.24, +1.53, +1.22), ("q", "Teacher-student KL high", +1.69, +1.45, +0.13, +1.09),
        ("q", "Student entropy high (1 answer)", +1.34, +1.06, +0.83, +1.08), ("q", "Both right", +0.80, +1.10, +0.84, +0.91),
        ("q", "Mixed", +0.08, +0.41, +0.34, +0.28), ("q", "Teacher-student KL middle", -0.18, -0.16, -0.40, -0.25),
        ("q", "Teacher-student KL low", -1.93, +0.18, -1.43, -1.06), ("q", "Both wrong", -1.71, -0.23, -1.85, -1.26),
        ("a", "KL high", -0.34, -0.33, -0.15, -0.27), ("a", "KL middle", -0.30, +0.53, -0.26, -0.01),
        ("a", "KL low", +0.25, -0.68, +0.05, -0.13), ("a", "Entropy high", -0.06, +0.04, -0.35, -0.13)]
fig, ax = plt.subplots(figsize=(4.6, 4.3)); y = 0; yt, yl, bold = [], [], []
for lev, title, col in [("q", "Choosing questions", P["blue_main"]), ("a", "Choosing answers", P["teal"])]:
    yt.append(y); yl.append(title); bold.append(True); y -= 1
    for l, lab, s1, s2, s3, m in rows:
        if l != lev: continue
        ax.plot([s1, s2, s3], [y] * 3, "o", color=col, alpha=0.35, ms=4)
        ax.plot(m, y, "D", color=col, ms=5.5)
        ax.text(2.45, y, f"{m:+.2f}", va="center", ha="right", fontsize=7.5)
        yt.append(y); yl.append(lab); bold.append(False); y -= 1
    y -= 0.4
ax.axvline(0, color=P["red_strong"], ls="--", lw=1)
ax.set_yticks(yt, yl); ax.set_xlim(-2.1, 2.5); ax.tick_params(axis="y", length=0)
for t, b in zip(ax.get_yticklabels(), bold):
    if b: t.set_fontweight("bold")
ax.set_xlabel("Minus random (points), MATH-500 strict avg@4")
save(fig, 2)

# fig3: between-question variance share, 12 pools (seed x refresh), from variance_all.txt
kl = [84.6, 85.1, 80.0, 82.1, 85.2, 83.4, 81.6, 81.6, 82.1, 77.2, 83.8, 80.7]
en = [85.8, 82.9, 75.6, 77.0, 87.2, 82.4, 78.4, 79.0, 84.0, 73.4, 82.6, 78.2]
fig, ax = plt.subplots(figsize=(4.6, 2.2)); x = np.arange(12); w = 0.38
ax.bar(x - w / 2, kl, w, color=P["blue_main"], label="Teacher-student KL")
ax.bar(x + w / 2, en, w, color=P["blue_secondary"], alpha=0.6, label="Student entropy")
ax.set_xticks(x, [f"s{s}\nr{r}" for s in (1, 2, 3) for r in range(4)], fontsize=6.5)
ax.set_ylim(0, 100); ax.set_ylabel("Variance between\nquestions (%)")
ax.set_xlabel("Pool (seed, refresh); 120 questions x 12 answers each")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, 3)
print("saved figs 1-3")
