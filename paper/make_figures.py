"""Paper figures, in the figures4papers house style, from results/paper_numbers.json.

    python paper/make_figures.py

Writes paper/figures/{method,rule_differences,variance_between_questions,gain_over_training}.pdf.
The method figure is a schematic (no data). Every other value comes from
results/paper_numbers.json, made by experiments/paper_numbers.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "results" / "paper_numbers.json"
FIGURES = ROOT / "paper" / "figures"
PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "teal": "#42949E", "red_strong": "#B64342",
           "green_3": "#8BCF8B", "violet": "#9A4D8E"}
QUESTION_RULES = (("t_right_s_wrong", "T right, S wrong"), ("qkl_high", "Rev. KL high"), ("qent1_high", "Entropy high (1 answer)"),
                  ("both_right", "Both right"), ("mixed", "Mixed"), ("qkl_mid", "Rev. KL middle"), ("qkl_low", "Rev. KL low"),
                  ("both_wrong", "Both wrong"))
ANSWER_RULES = (("kl_high", "Rev. KL high"), ("kl_mid", "Rev. KL middle"), ("kl_low", "Rev. KL low"), ("ent_top4", "Entropy high"))


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans"], "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 1.2,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def save(fig: Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def schematic_panel(ax: Axes, title: str, shape: tuple[int, int], keep: tuple[set[int], set[int]], note: str, sort: bool) -> None:
    """A grid of questions (rows) x answers (columns); kept cells outlined in red."""
    rng = np.random.default_rng(3)
    rows, cols = shape
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.9, 6.4)
    ax.axis("off")
    ax.text(0.2, 6.2, title, fontsize=8.5, fontweight="bold", va="top")
    cw, ch = 9.6 / cols, 4.6 / rows
    for i in range(rows):
        values = np.clip(rng.uniform(0.25, 0.85) + rng.normal(0, 0.07, cols), 0, 1)
        if sort:
            values = np.sort(values)[::-1]
        for j, v in enumerate(values):
            kept = i in keep[0] and j in keep[1]
            ax.add_patch(Rectangle((0.2 + j * cw, 0.6 + (rows - 1 - i) * ch), cw * 0.88, ch * 0.8,
                                   facecolor=plt.get_cmap("Blues")(0.25 + 0.7 * v),
                                   edgecolor=PALETTE["red_strong"] if kept else "white", lw=1.5 if kept else 0.4))
    ax.text(0.2, -0.2, note, fontsize=7, va="center")


def method_figure() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.1))
    schematic_panel(axs[0], "Choosing answers", (6, 12), (set(range(6)), {4, 5, 6, 7}),
                    "12 answers per question, sorted by KL (darker = higher);\nkeep 4 per question, e.g. the middle band", True)
    schematic_panel(axs[1], "Choosing questions", (9, 4), ({0, 3, 6}, set(range(4))),
                    "4 answers per question; shade = question KL;\nkeep a third of the questions with all their answers", False)
    save(fig, "method")


def rule_differences_figure(table: dict[str, dict[str, object]]) -> None:
    """Difference to standard OPD per rule: three seeds (dots) and their mean (diamond)."""
    fig, ax = plt.subplots(figsize=(2.2, 3.9))
    y = 0.0
    ticks: list[float] = []
    labels: list[str] = []
    bold: list[bool] = []
    for title, rules, color in (("Choosing questions", QUESTION_RULES, PALETTE["blue_main"]),
                                ("Choosing answers", ANSWER_RULES, PALETTE["teal"])):
        ticks.append(y)
        labels.append(title)
        bold.append(True)
        y -= 1
        for arm, label in rules:
            seeds = table[arm]["avg4_delta_per_seed"]
            mean = table[arm]["avg4_delta"]
            assert isinstance(seeds, list) and isinstance(mean, dict)
            ax.plot(seeds, [y] * len(seeds), "o", color=color, alpha=0.35, ms=4)
            ax.plot(mean["mean"], y, "D", color=color, ms=5.5)
            ax.text(2.45, y, f"{mean['mean']:+.2f}", va="center", ha="right", fontsize=7)
            ticks.append(y)
            labels.append(label)
            bold.append(False)
            y -= 1
        y -= 0.4
    ax.axvline(0, color=PALETTE["red_strong"], ls="--", lw=1)
    ax.set_yticks(ticks, labels)
    ax.set_xlim(-2.1, 2.5)
    ax.tick_params(axis="y", length=0)
    for tick, is_bold in zip(ax.get_yticklabels(), bold):
        tick.set_fontweight("bold" if is_bold else "normal")
    ax.set_xlabel("Difference to standard OPD (points)")
    save(fig, "rule_differences")


def variance_figure(shares: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(3.0, 2.1))
    x = np.arange(len(shares["reverse_kl"]))
    width = 0.38
    ax.bar(x - width / 2, shares["reverse_kl"], width, color=PALETTE["blue_main"], label="Reverse KL")
    ax.bar(x + width / 2, shares["entropy"], width, color=PALETTE["blue_secondary"], alpha=0.6, label="Student entropy")
    ax.set_xticks(x, [f"s{s}\nr{r}" for s in (1, 2, 3) for r in range(4)], fontsize=6.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Variance between\nquestions (%)")
    ax.set_xlabel("Pool (seed s, refresh r)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "variance_between_questions")


def ramp_line(ax: Axes, x: np.ndarray, y: list[float], color: str) -> None:
    """Line whose opacity rises from left to right (figures4papers post-training style)."""
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    rgb = to_rgba(color)
    alphas = np.linspace(0.35, 0.95, len(segments))
    ax.add_collection(LineCollection(segments, colors=[(*rgb[:3], a) for a in alphas], linewidths=1.8, capstyle="round"))
    ax.plot(x, y, lw=0, marker="o", ms=4.2, color=color)


def gain_figure(gains: dict[str, list[float]], steps: list[int]) -> None:
    panels = (("Choosing questions", (("t_right_s_wrong", "T right, S wrong", PALETTE["blue_main"]),
                                      ("qent1_high", "Entropy high (1 answer)", PALETTE["green_3"]),
                                      ("qkl_high", "Reverse KL high", PALETTE["teal"]),
                                      ("both_wrong", "Both wrong", PALETTE["red_strong"]))),
              ("Choosing answers", (("kl_high", "Reverse KL high", PALETTE["blue_main"]), ("kl_mid", "Reverse KL middle", PALETTE["teal"]),
                                    ("kl_low", "Reverse KL low", PALETTE["violet"]), ("ent_top4", "Entropy high", PALETTE["green_3"]))))
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.75), sharey=True)
    x = np.arange(len(steps))
    for ax, (title, rules) in zip(axs, panels):
        ax.axhline(0, color="black", alpha=0.35, lw=1.6, ls="--")
        handles = [Line2D([0], [0], color="black", alpha=0.35, lw=1.6, ls="--", label="Standard OPD")]
        for arm, label, color in rules:
            ramp_line(ax, x, gains[arm], color)
            handles.append(Line2D([0], [0], color=color, lw=1.8, marker="o", ms=4.2, label=label))
        ax.set_xticks(x, [str(t) for t in steps])
        ax.set_xlim(-0.25, len(steps) - 0.75)
        ax.set_xlabel("Training step")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylim(-2.4, 2.4)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7, handlelength=2.2)
    axs[0].set_ylabel("Gain over standard OPD\n(MATH-500 avg@4, points)")
    save(fig, "gain_over_training")


def main() -> None:
    numbers = json.loads(NUMBERS.read_text())
    apply_style()
    method_figure()
    rule_differences_figure(numbers["mean_over_steps_15_60"])
    variance_figure(numbers["variance_between_questions_pct"])
    gain = numbers["gain_per_step"]
    gain_figure({k: v for k, v in gain.items() if k != "steps"}, gain["steps"])
    print(f"wrote 4 figures to {FIGURES}")


if __name__ == "__main__":
    main()
