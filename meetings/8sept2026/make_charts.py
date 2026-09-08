"""
Charts for the 8 Sept 2026 results deck. Run from repo root:
    uv run python meetings/8sept2026/make_charts.py
Writes PNGs to meetings/8sept2026/charts/ and prints the pooled table.

Sources: outputs/runs/{r1,r2,r3,r4}/metrics.jsonl (MATH-500 strict avg@4 /
pass@4 / cap-hit at every refresh); kl50 numbers hard-coded from the
1 Sept deck (outputs/runs/kl50, eval at 0/17/34 training steps).
"""
import json
import os
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "charts")
os.makedirs(OUT, exist_ok=True)

BG, GRID, TEXT, DIM = "#ffffff", "#ddd7c5", "#201d17", "#5c584c"
C = {
    # trajectory rules (same as the 1 Sept deck)
    "kl_high": "#c94a35", "kl_mid": "#b3790a", "kl_low": "#3f6690", "random": "#5f6b45",
    "random4": "#5f6b45", "entropy_top4": "#7a4e8c",
    # correctness buckets
    "teacher_right_student_wrong": "#b3790a", "both_right": "#2f7d4f",
    "both_wrong": "#8c2e2e", "mixed": "#3f6690",
    # question entropy
    "uncertain_questions": "#7a4e8c", "random_questions": "#5f6b45", "certain_questions": "#1f8f82",
}
LABEL = {
    "teacher_right_student_wrong": "teacher right / student wrong",
    "both_right": "both right", "both_wrong": "both wrong", "mixed": "mixed",
    "entropy_top4": "entropy top-4", "kl_high": "kl_high", "kl_mid": "kl_mid", "kl_low": "kl_low",
    "random4": "random 4 of 12", "random": "random",
    "uncertain_questions": "uncertain (top-800 H(q))", "random_questions": "random 800",
    "certain_questions": "certain (bottom-800 H(q))",
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor": DIM, "text.color": TEXT, "xtick.color": DIM, "ytick.color": DIM,
    "grid.color": GRID, "font.family": "monospace", "font.size": 12, "axes.grid": True,
    "grid.linewidth": 0.7, "grid.alpha": 0.85, "axes.linewidth": 0.8,
    "legend.frameon": False, "legend.labelcolor": DIM, "savefig.facecolor": BG,
})
DPI = 130
NOISE = 0.02  # +-2 pt round-to-round band, see docs/decisions.md 2026-09-05


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved", name)


def load(run):
    arms = {}
    for line in open(f"outputs/runs/{run}/metrics.jsonl"):
        d = json.loads(line)
        arms.setdefault(d["arm"], {})[d["step"]] = d
    out = {}
    for arm, by_step in arms.items():
        steps = sorted(by_step)
        m = lambda k, f: by_step[k]["eval"]["math500"][f]
        out[arm] = {
            "steps": steps,
            "avg4": [m(k, "strict_avg_at_n") for k in steps],
            "pass4": [m(k, "strict_pass_at_n") for k in steps],
            "cap": [m(k, "cap_hit_rate") for k in steps],
        }
    return out


R1 = load("r1-correctness-8k")
R2 = load("r2-trajsel-8k")
R3 = load("r3-qentropy-8k")
R4 = load("r4-qentropy-low-8k")
R34 = {**R3, **R4}

# kl50 (1 Sept deck): 136 questions x 12 rollouts keep 4, 17 steps per round,
# eval at 0 / 17 / 34 training steps, LR 2e-5 constant, Adam reset per round.
KL50_STEPS = [0, 17, 34]
KL50_AVG4 = {"kl_mid": [0.2785, 0.3935, 0.4745], "kl_high": [0.2785, 0.3630, 0.4320],
             "kl_low": [0.2785, 0.3555, 0.3835], "random": [0.2785, 0.4300, 0.3880]}
KL50_PASS4 = {"kl_mid": [0.412, 0.548, 0.612], "kl_high": [0.412, 0.504, 0.562],
              "kl_low": [0.412, 0.490, 0.516], "random": [0.412, 0.570, 0.520]}
TEACHER_AVG4 = 0.6115


def curves(data, order, name, title_avg="avg@4", title_pass="pass@4",
           steps_key="steps", ylim_avg=(0.24, 0.50), ylim_pass=(0.36, 0.66), teacher=False):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for arm in order:
        d = data[arm]
        a1.plot(d[steps_key], d["avg4"], marker="o", ms=5, lw=2.6, color=C[arm], label=LABEL[arm])
        a2.plot(d[steps_key], d["pass4"], marker="s", ms=5, lw=2.2, ls=(0, (4, 2)), color=C[arm], label=LABEL[arm])
    for ax, t, yl in ((a1, title_avg, ylim_avg), (a2, title_pass, ylim_pass)):
        ax.set_title(t, fontsize=13, color=DIM, loc="left")
        ax.set_xlabel("training step")
        ax.set_ylim(*yl)
        style(ax)
    a1.set_ylabel("MATH-500 accuracy")
    a1.legend(loc="lower right", fontsize=9.5)
    save(fig, name)


# --- kl50: the run where selection looked like it worked ---
kl50 = {a: {"steps": KL50_STEPS, "avg4": KL50_AVG4[a], "pass4": KL50_PASS4[a]} for a in KL50_AVG4}
curves(kl50, ["kl_mid", "kl_high", "kl_low", "random"], "chart_kl50",
       ylim_avg=(0.24, 0.52), ylim_pass=(0.36, 0.66))

# --- r1 / r2 / r3+r4 curves over 100 steps ---
curves(R1, ["teacher_right_student_wrong", "both_right", "mixed", "both_wrong"], "chart_r1")
curves(R2, ["kl_mid", "kl_high", "kl_low", "random4", "entropy_top4"], "chart_r2")
curves(R34, ["random_questions", "uncertain_questions", "certain_questions"], "chart_r3r4")

# --- all 12 arms, saturation at step 10 ---
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
allarms = [(R1, a) for a in R1] + [(R2, a) for a in R2] + [(R34, a) for a in R34]
for data, arm in allarms:
    d = data[arm]
    a1.plot(d["steps"], d["avg4"], lw=1.6, color=C[arm], alpha=0.75)
    a2.plot(d["steps"], d["cap"], lw=1.6, color=C[arm], alpha=0.75)
pooled = [statistics.mean(x) for x in zip(*[data[a]["avg4"] for data, a in allarms])]
a1.plot(R1["mixed"]["steps"], pooled, lw=3.2, color=TEXT, label="mean of 12 arms")
a1.fill_between(R1["mixed"]["steps"], [p - NOISE for p in pooled], [p + NOISE for p in pooled],
                color=TEXT, alpha=0.08, lw=0, label="+-2 pt noise band")
a1.axvline(10, color=DIM, lw=1.2, ls=(0, (4, 3)))
a1.text(12, 0.27, "step 10", color=DIM, fontsize=10)
a1.set_title("avg@4, all 12 arms of r1-r4", fontsize=13, color=DIM, loc="left")
a1.set_ylabel("MATH-500 accuracy"); a1.set_ylim(0.24, 0.50)
a1.legend(loc="lower right", fontsize=9.5)
a2.set_title("cap-hit rate on MATH-500 (8192 tokens)", fontsize=13, color=DIM, loc="left")
a2.set_ylim(0.5, 0.85)
for ax in (a1, a2):
    ax.set_xlabel("training step"); style(ax)
save(fig, "chart_saturation")

# --- pooled steps 50-100, one bar per arm, random arms hatched ---
def pooled_stat(d, key):
    vals = [v for s, v in zip(d["steps"], d[key]) if s >= 50]
    return statistics.mean(vals)

rows = []
for run, data, arms in (("r1", R1, ["teacher_right_student_wrong", "both_right", "mixed", "both_wrong"]),
                        ("r2", R2, ["kl_mid", "kl_high", "kl_low", "random4", "entropy_top4"]),
                        ("r3/r4", R34, ["random_questions", "uncertain_questions", "certain_questions"])):
    for a in arms:
        rows.append((run, a, pooled_stat(data[a], "avg4"), pooled_stat(data[a], "pass4"), pooled_stat(data[a], "cap")))

print("\npooled steps 50-100 (6 refreshes), MATH-500 strict")
print(f"{'run':6s} {'arm':30s} {'avg@4':>7s} {'pass@4':>7s} {'cap':>6s}")
for run, a, m, p, c in rows:
    print(f"{run:6s} {a:30s} {m:7.4f} {p:7.4f} {c:6.3f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
ys = list(range(len(rows)))
for ax, idx, title, lim in ((a1, 2, "avg@4, mean of steps 50-100", (0.38, 0.47)),
                            (a2, 3, "pass@4, mean of steps 50-100", (0.50, 0.62))):
    for y, r in zip(ys, rows):
        is_rand = r[1] in ("random4", "random_questions")
        ax.barh(y, r[idx], color=C[r[1]], height=0.62, hatch="///" if is_rand else None,
                edgecolor=BG if not is_rand else TEXT, lw=0.6, zorder=3)
        ax.text(r[idx] + 0.002, y, f"{r[idx]:.3f}", va="center", fontsize=10.5, color=TEXT)
    ax.set_xlim(*lim); ax.set_title(title, fontsize=13, color=DIM, loc="left")
    ax.grid(axis="y", visible=False); style(ax)
a1.set_yticks(ys); a1.set_yticklabels([f"{r[0]}  {LABEL[r[1]]}" for r in rows], fontsize=10.5)
a1.invert_yaxis()
for y in (3.5, 8.5):
    a1.axhline(y, color=GRID, lw=1); a2.axhline(y, color=GRID, lw=1)
save(fig, "chart_pooled")

# --- r3/r4 symmetry: three bars ---
fig, ax = plt.subplots(figsize=(6.2, 4.4))
order = ["uncertain_questions", "random_questions", "certain_questions"]
vals = [pooled_stat(R34[a], "avg4") for a in order]
bars = ax.bar(range(3), vals, color=[C[a] for a in order], width=0.6, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.3f}", ha="center", fontsize=12, color=TEXT)
ax.set_xticks(range(3)); ax.set_xticklabels(["uncertain\n(high H(q))", "random", "certain\n(low H(q))"])
ax.set_ylim(0.40, 0.45); ax.set_ylabel("avg@4, mean of steps 50-100")
ax.grid(axis="x", visible=False); style(ax)
save(fig, "chart_symmetry")
print("done")
