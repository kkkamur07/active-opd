"""Plot MATH-500 accuracy vs training step (contract: docs/pipeline.md).

Reads metrics.jsonl and renders plots/accuracy_vs_steps.png: one curve
per arm, avg@n on top, pass@n below (dashed) as two stacked panels sharing the
x-axis -- never a dual-axis chart.

X-axis semantics: one training step = one optimizer step at the effective
batch (cfg.train.effective_batch, asserted 32 by the train stage), so
step = trajectories / effective_batch. Row r's eval measured the model BEFORE
round r's training, while trajectories_cumulative includes round r's
selection. So every point sits at x = (trajectories_cumulative -
trajectories_round) / effective_batch: the round-0 row lands on x=0
(untrained anchor) and the final eval-only row (trajectories_round=0) lands
at the full cumulative step count -- one uniform rule, no special-casing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the driver imports and calls plot_results
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import OmegaConf

from apod.datasets.io import read_jsonl

# Fixed arm -> hue mapping (validated palette, slots 1-3 are all-pairs safe).
# Color follows the entity: an arm keeps its hue regardless of which arms are
# present in this run, so plots stay comparable across runs.
ARM_COLORS = {
    "entropy_top4": "#2a78d6",  # blue
    "random_top4": "#eb6834",  # orange
    "all": "#1baf7a",  # aqua
}
EXTRA_SLOTS = ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def _arm_color(arm: str, overflow: list[str]) -> str:
    if arm in ARM_COLORS:
        return ARM_COLORS[arm]
    # Unknown arms take the remaining validated slots in fixed order.
    if arm not in overflow:
        overflow.append(arm)
    return EXTRA_SLOTS[overflow.index(arm) % len(EXTRA_SLOTS)]


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)


def plot_results(run_dir: Path) -> Path:
    rows = read_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        raise FileNotFoundError(f"no metrics rows found at {run_dir / 'metrics.jsonl'}")

    run_name = run_dir.name
    num_samples = None
    eff_batch = 32
    cfg_path = run_dir / "resolved_config.yaml"
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        run_name = str(cfg.get("run_name", run_name))
        num_samples = cfg.eval.num_samples if "eval" in cfg else None
        eff_batch = int(cfg.train.get("effective_batch", eff_batch)) if "train" in cfg else eff_batch
    n_label = str(num_samples) if num_samples is not None else "n"

    # Preserve first-seen arm order (driver writes arms in cfg.arms order).
    arms: list[str] = []
    for row in rows:
        if row["arm"] not in arms:
            arms.append(row["arm"])

    fig, (ax_avg, ax_pass) = plt.subplots(
        2, 1, sharex=True, figsize=(8.5, 7.5), facecolor=SURFACE
    )
    overflow: list[str] = []
    for arm in arms:
        arm_rows = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["round"])
        # Row r evaluated the pre-training model, so subtract the trajectories
        # this round added; see module docstring.
        xs = [
            (r["trajectories_cumulative"] - (r["trajectories_round"] or 0)) / eff_batch
            for r in arm_rows
        ]
        color = _arm_color(arm, overflow)
        ax_avg.plot(
            xs,
            [r["avg_at_n"] for r in arm_rows],
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            label=arm,
        )
        ax_pass.plot(
            xs,
            [r["pass_at_n"] for r in arm_rows],
            color=color,
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=6,
            label=arm,
        )
        # Direct labels at the line end, in neutral ink (identity is carried by
        # the adjacent colored mark, not by colored text).
        ax_avg.annotate(
            arm,
            (xs[-1], arm_rows[-1]["avg_at_n"]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    _style_axis(ax_avg)
    _style_axis(ax_pass)
    ax_avg.set_ylabel(f"avg@{n_label}", color=INK_SECONDARY)
    ax_pass.set_ylabel(f"pass@{n_label}", color=INK_SECONDARY)
    ax_pass.set_xlabel(f"training step (batch={eff_batch})", color=INK_SECONDARY)
    ax_avg.set_title(
        f"{run_name}: MATH-500 accuracy vs training step (n={n_label})",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
    )
    ax_pass.set_title("pass@n (monitor)", color=INK_SECONDARY, fontsize=10, loc="left")
    ax_avg.legend(
        loc="best", frameon=False, fontsize=9, labelcolor=INK_SECONDARY
    )
    ax_pass.legend(
        loc="best", frameon=False, fontsize=9, labelcolor=INK_SECONDARY
    )
    ax_avg.margins(x=0.12)  # room for the direct labels past the last point

    out_path = run_dir / "plots" / "accuracy_vs_steps.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote {}", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_results(args.run_dir.resolve())


if __name__ == "__main__":
    main()
