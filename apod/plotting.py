"""Plot accuracy vs training step (contract: docs/pipeline.md).

Reads metrics.jsonl and renders plots/accuracy_vs_steps.png. Two row
schemas: ``plot_refresh_curves`` for apod.driver runs (one row per (arm,
step), strict avg@n per eval set with a noise band, cap-hit panel) and
``plot_results`` for apod.main runs (one row per (arm, round), avg@n on
top, pass@n below). Stacked panels sharing the x-axis -- never a dual-axis
chart. The rest of this docstring describes the apod.main schema.

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


def plot_refresh_curves(run_dir: Path, *, band_points: float = 5.0) -> Path:
    """Curves of a step-based run (apod.driver): strict accuracy vs training step.

    metrics.jsonl rows are one per (arm, step) with ``eval[<set>]`` summaries.
    Panels, sharing the step axis: strict avg@n on the primary set
    (MATH-500), strict avg@n on each monitor set (AIME 2025+2026), and the
    cap-hit rate of both. Every accuracy curve carries a shaded band of
    ``band_points`` (full width, in accuracy points): the seed-replicate
    noise floor (docs/eval_benchmarks.md section 6, 4-7 point gaps at a
    fixed step), so two curves whose bands overlap are not distinguishable
    on one seed.
    """

    rows = read_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        raise FileNotFoundError(f"no metrics rows found at {run_dir / 'metrics.jsonl'}")
    run_name = run_dir.name
    eff_batch = 32
    cfg_path = run_dir / "resolved_config.yaml"
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        run_name = str(cfg.get("run_name", run_name))
        eff_batch = int(cfg.train.get("effective_batch", eff_batch))

    arms: list[str] = []
    sets: list[str] = []
    for row in rows:
        if row["arm"] not in arms:
            arms.append(row["arm"])
        for name in row["eval"]:
            if name not in sets:
                sets.append(name)
    half_band = band_points / 200.0  # points -> fraction, half width each side

    fig, axes = plt.subplots(
        len(sets) + 1, 1, sharex=True, figsize=(8.5, 3.2 * (len(sets) + 1)), facecolor=SURFACE
    )
    overflow: list[str] = []
    for arm in arms:
        arm_rows = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["step"])
        xs = [r["step"] for r in arm_rows]
        color = _arm_color(arm, overflow)
        for ax, name in zip(axes, sets):
            ys = [r["eval"][name]["strict_avg_at_n"] for r in arm_rows]
            ax.fill_between(xs, [y - half_band for y in ys], [y + half_band for y in ys], color=color, alpha=0.10, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=arm)
            ax.annotate(arm, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_SECONDARY)
        for name, style in zip(sets, ("-", "--", ":", "-.")):
            axes[-1].plot(
                xs, [r["eval"][name]["cap_hit_rate"] for r in arm_rows],
                color=color, linewidth=1.5, linestyle=style, marker="o", markersize=4,
                label=f"{arm} ({name})",
            )

    for ax, name in zip(axes, sets):
        _style_axis(ax)
        n = next(r["eval"][name]["num_samples"] for r in rows if name in r["eval"])
        ax.set_ylabel(f"{name} strict avg@{n}", color=INK_SECONDARY)
        ax.set_title(
            f"{name}: strict avg@{n} vs training step (band = {band_points:g}-point noise floor)",
            color=INK_SECONDARY, fontsize=10, loc="left",
        )
        ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
        ax.margins(x=0.12)
    _style_axis(axes[-1])
    axes[-1].set_ylabel("cap-hit rate", color=INK_SECONDARY)
    axes[-1].set_title("cap-hit rate (solid: primary set; dashed: monitor sets)", color=INK_SECONDARY, fontsize=10, loc="left")
    axes[-1].set_xlabel(f"training step (batch={eff_batch})", color=INK_SECONDARY)
    axes[-1].legend(loc="best", frameon=False, fontsize=7, labelcolor=INK_SECONDARY, ncol=2)
    fig.suptitle(f"{run_name}: accuracy vs training step", color=INK_PRIMARY, fontsize=12, x=0.02, ha="left")

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
    parser.add_argument("--band-points", type=float, default=None,
                        help="noise band width in accuracy points for step-based runs "
                             "(default: driver.noise_band_points of the run, else 5)")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rows = read_jsonl(run_dir / "metrics.jsonl")
    if rows and "step" in rows[0]:  # apod.driver run: one row per (arm, step)
        band = args.band_points
        if band is None:
            cfg_path = run_dir / "resolved_config.yaml"
            cfg = OmegaConf.load(cfg_path) if cfg_path.exists() else {}
            band = float(cfg.get("driver", {}).get("noise_band_points", 5.0))
        plot_refresh_curves(run_dir, band_points=band)
    else:  # apod.main run: one row per (arm, round)
        plot_results(run_dir)


if __name__ == "__main__":
    main()
