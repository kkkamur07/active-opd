"""
Generate all deck charts as PNGs for base64 embedding into the HTML slides.
Run from repo root with: uv run python <this file>

Deck covers steps 0/17/34 (selection rounds 0/1/2) only. Step 51 (round 3,
trained under a warmup LR regime) is dropped from the deck entirely.
"""
import json
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

OUT_DIR = "/tmp/claude-1001/-home-krishuagarwal-Desktop-active-opd/18e72832-7bae-4766-bd6f-ca5a9c4f1f46/scratchpad"

# ---- palette (matches deck CSS tokens, light theme) ----
BG = "#ffffff"       # --surface (white card, matches panel bg)
GRID = "#ddd7c5"
TEXT = "#201d17"
TEXT_DIM = "#5c584c"
KL_HIGH = "#c94a35"
KL_MID = "#b3790a"
KL_LOW = "#3f6690"
RANDOM = "#5f6b45"
TEACHER = "#1f8f82"

ARM_COLOR = {"kl_high": KL_HIGH, "kl_mid": KL_MID, "kl_low": KL_LOW, "random": RANDOM}
ARMS_ORDER = ["kl_mid", "kl_high", "kl_low", "random"]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_DIM,
    "text.color": TEXT,
    "xtick.color": TEXT_DIM,
    "ytick.color": TEXT_DIM,
    "grid.color": GRID,
    "font.family": "monospace",
    "font.size": 12,
    "axes.grid": True,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.85,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "legend.labelcolor": TEXT_DIM,
    "savefig.facecolor": BG,
})

DPI = 120


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved", name)


# ===========================================================
# Per-arm data, selected rollouts, rounds 0/1/2 (steps 0/17/34)
# random's oracle scoring stops after round 0 by design (its
# selection needs no scoring in later rounds), so random has
# only a round-0 point for overlap ratio / advantage / entropy
# ===========================================================
AVG4 = {
    "kl_mid":  [0.2785, 0.3935, 0.4745],
    "kl_high": [0.2785, 0.3630, 0.4320],
    "kl_low":  [0.2785, 0.3555, 0.3835],
    "random":  [0.2785, 0.4300, 0.3880],
}
PASS4 = {
    "kl_mid":  [0.412, 0.548, 0.612],
    "kl_high": [0.412, 0.504, 0.562],
    "kl_low":  [0.412, 0.490, 0.516],
    "random":  [0.412, 0.570, 0.520],
}
OVERLAP = {
    "kl_mid":  [0.7259, 0.7038, 0.6955],
    "kl_high": [0.7260, 0.7031, 0.7010],
    "kl_low":  [0.7255, 0.7020, 0.6950],
    "random":  [0.7258],
}
ADVANTAGE = {
    "kl_mid":  [-0.0089, -0.0096, -0.0095],
    "kl_high": [-0.0096, -0.0103, -0.0104],
    "kl_low":  [-0.0081, -0.0083, -0.0087],
    "random":  [-0.0088],
}
ENTROPY = {
    "kl_mid":  [0.0707, -0.0575, -0.0544],
    "kl_high": [0.0765, -0.0639, -0.0566],
    "kl_low":  [0.0639, -0.0493, -0.0552],
    "random":  [0.0697],
}

# consistent y-limits across all four arm slides, per statistic
ACC_YLIM = (0.22, 0.66)
OVERLAP_YLIM = (0.688, 0.735)
ADV_YLIM = (-0.0115, -0.0058)
ENTROPY_YLIM = (-0.075, 0.09)

PANEL_SIZE = (4.5, 4.3)


def plot_arm_accuracy(arm):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    color = ARM_COLOR[arm]
    steps = [0, 17, 34]
    ax.plot(steps, AVG4[arm], marker="o", markersize=6.5, linewidth=2.8,
            color=color, label="avg@4", zorder=4)
    ax.plot(steps, PASS4[arm], marker="s", markersize=6, linewidth=2.2,
            linestyle=(0, (4, 2)), color=color, alpha=0.62, label="pass@4", zorder=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy (strict)")
    ax.set_xticks(steps)
    ax.set_ylim(*ACC_YLIM)
    style_axes(ax)
    ax.legend(loc="lower right", fontsize=9.5)
    save(fig, f"chart_arm_{arm}_accuracy")


def plot_arm_round_stat(arm, data, ylabel, ylim, suffix, hline=None):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    color = ARM_COLOR[arm]
    vals = data[arm]
    if hline is not None:
        ax.axhline(hline, color=TEXT_DIM, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    if len(vals) == 1:
        ax.scatter([0], vals, s=95, color=color, zorder=4)
    else:
        ax.plot([0, 1, 2], vals, marker="o", markersize=6.5, linewidth=2.8, color=color, zorder=4)
    ax.set_xlabel("selection round")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 1, 2])
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(*ylim)
    style_axes(ax)
    save(fig, f"chart_arm_{arm}_{suffix}")


for arm in ARMS_ORDER:
    plot_arm_accuracy(arm)
    plot_arm_round_stat(arm, OVERLAP, "top-16 overlap ratio", OVERLAP_YLIM, "overlap")
    plot_arm_round_stat(arm, ADVANTAGE, "overlap-token advantage", ADV_YLIM, "advantage")
    plot_arm_round_stat(arm, ENTROPY, "student minus teacher entropy", ENTROPY_YLIM, "entropy", hline=0)

# ===========================================================
# Indicator-comparison slide: agreement bars + scatter
# ===========================================================
indicators = ["overlap ratio", "entropy difference", "overlap advantage"]
agreement = [32.2, 50.0, 84.7]
bar_colors = [KL_HIGH, TEACHER, KL_MID]
CHANCE = 33.3

fig, ax = plt.subplots(figsize=(6.4, 4.6))
y_pos = range(len(indicators))
bars = ax.barh(list(y_pos), agreement, color=bar_colors, height=0.52, zorder=3)
ax.axvline(CHANCE, color=TEXT_DIM, linewidth=1.6, linestyle=(0, (5, 4)), zorder=2)
blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.text(CHANCE, 1.03, "chance", color=TEXT_DIM, fontsize=10, ha="center", va="bottom", transform=blended)
for rect, val in zip(bars, agreement):
    ax.text(rect.get_width() + 1.8, rect.get_y() + rect.get_height() / 2, f"{val:.1f}%",
            color=TEXT, fontsize=11, va="center")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(indicators)
ax.set_xlabel("agreement with kl_mid's actual picks (%)")
ax.set_xlim(0, 100)
ax.invert_yaxis()
style_axes(ax)
ax.grid(axis="y", visible=False)
save(fig, "chart_agreement")

# scatter: mean reverse KL vs overlap advantage, round-0 shared
# pool, 1632 scored rollouts (136 prompts x 12), oracle_kl JSONL
rows = {}
for fp in sorted(glob.glob("outputs/runs/kl50/arms/kl_high/rounds/round_00/oracle/*.jsonl")):
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (d["example_index"], d["rollout_index"])
            rows[key] = d  # last row per key wins

pts = [d for d in rows.values() if "overlap_ratio_top16" in d]
print("scatter points:", len(pts))
xs = [d["mean_reverse_kl"] for d in pts]
ys = [d["overlap_adv_top16"] for d in pts]

fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.scatter(xs, ys, s=7, alpha=0.4, color=KL_MID, edgecolors="none")
ax.set_xlabel("mean reverse KL")
ax.set_ylabel("overlap-token advantage")
style_axes(ax)
save(fig, "chart_scatter")

print("done")
