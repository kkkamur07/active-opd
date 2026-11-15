"""Figures for the ACL paper (on-policy post-training with aligned-band selection).

Run (from papers/apod-acl):
    uv run --no-project --with matplotlib --with numpy python make_figures.py

Reads only outputs/runs_results/ (raw eval shards, oracle-KL shards) and
experiments/apod-data-scan/ (results.json, used only to cross-check CIs).
Writes figures/{fig0_method,fig1_main,fig2_timecourse,fig3_termination,fig4_entropy}.{pdf,png} (png at 300 dpi)
and figures/README.md (data sources + every number printed below). Visual style: figures4papers house style.
Units: accuracies are shown as percentages (exact x100 of the 4-decimal strict rate, 2 decimals: 0.4745 -> 47.45%);
differences are percentage points (pts). fig0_method is a schematic and contains no data values.

Repo -> paper names (papers/apod-story.md section 0):
    kl50 = MidAlign-8k; oracle16k / oracle16k_seed2 = MidAlign-16k seed A / seed B;
    oracle8k = MidAlign-8k pilot; r2-trajsel-8k = MidAlign-long;
    kl_mid / kl_high / kl_low = aligned band / divergent / redundant; random, random4 = uniform-k; all = full.
kl50 eval rounds: round_01 = step 17, round_02 = step 34; step 51 exists only as the
warmup-variant retrain in kl50_warmup_backup/<arm>/round03.
"""
from __future__ import annotations

import glob
import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "outputs" / "runs_results"
SCAN = REPO / "experiments" / "apod-data-scan"
FIG = HERE / "figures"

N_BOOT = 10_000
COL_W = 3.2   # single-column width (in)
PAGE_W = 6.5  # double-column width (in)

# Seed-to-seed spread of an arm-minus-uniform contrast: per-run SD 3.40 pts (oracle16k vs oracle16k_seed2,
# kl_mid and kl_high contrasts at rounds 1-3, kl_low excluded) -- notes/apod-critique-results.md section 1
# ("Seed replicate: arm-random contrast ... 3.40 excluding kl_low"; critic_checks.py section B).
SEED_SD = 3.40
SEED_BAND = 1.96 * SEED_SD  # 6.664 -> shown as +/-6.7 pts

# ---------------------------------------------------------------- house style
# figures4papers house style (Chen Liu, CC BY-NC 4.0; papers/figures4papers/scientific-figure-making):
# semantic PALETTE, FigureStyle + apply_publication_style, annotate_bars, finalize_figure (pdf + png, 300 dpi).
# Sizes are scaled down from the repo's 16-24 pt presets so text stays >= 7 pt after LaTeX shrinks a 3.2 in figure
# to the ~3.03 in ACL column (and a 6.5 in figure to the ~6.3 in text width).
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
# Role -> colour. Blues = our method (aligned band; both MidAlign-16k seeds are aligned-band runs);
# red = the baseline (uniform-k); teal / violet = the other KL thirds; neutral = background (seed band).
C_ALIGNED = PALETTE["blue_main"]
C_ALIGNED_SEED_B = PALETTE["blue_secondary"]
C_UNIFORM = PALETTE["red_strong"]
C_DIVERGENT = PALETTE["teal"]
C_REDUNDANT = PALETTE["violet"]
C_BAND = PALETTE["neutral"]
# Fig 3 components (not methods): finishing = the main channel, the two accuracy channels in green / red.
C_FINISH = PALETTE["blue_secondary"]
C_ACCFIN = PALETTE["green_3"]
C_ACCTRUNC = PALETTE["red_2"]
# Fig 4 pools (not methods): kept off the method colours (blue_main / red_strong / teal / violet) so those keep
# their aligned / uniform-k / divergent / redundant meaning. 8k pools = blue_secondary, 16k seeds = green_3
# (second member hatched), MidAlign-long = neutral.
C_POOLS = [PALETTE["blue_secondary"], PALETTE["blue_secondary"], PALETTE["green_3"], PALETTE["green_3"], PALETTE["neutral"]]
H_POOLS = ["", "////", "", "////", ""]

ANNOT_FS = 7.5  # in-place value annotations


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 8.5
    axes_linewidth: float = 1.2
    use_tex: bool = False
    font_family: tuple[str, ...] = ("Helvetica", "Arial", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    style = style or FigureStyle()
    plt.rcParams.update({
        "font.family": list(style.font_family),
        "font.size": style.font_size,
        "axes.labelsize": style.font_size,
        "xtick.labelsize": style.font_size - 0.5,
        "ytick.labelsize": style.font_size - 0.5,
        "legend.fontsize": style.font_size - 1,
        "mathtext.fontset": "dejavusans",
        "text.usetex": style.use_tex,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": style.axes_linewidth,
        "xtick.major.width": style.axes_linewidth,
        "ytick.major.width": style.axes_linewidth,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        # constrained layout keeps the page exactly at figsize (3.2 in / 6.5 in) instead of bbox="tight" resizing
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
    })


def annotate_bars(ax, bars, fmt: str = "{:.2f}", fontsize: float = ANNOT_FS, padding: float = 2) -> None:
    for bar in bars:
        ax.annotate(fmt.format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, padding), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize)


WRITTEN: list[Path] = []


def finalize_figure(fig, out_stem: Path, formats: tuple[str, ...] = ("pdf", "png"), dpi: int = 300) -> None:
    for fmt in formats:
        p = out_stem.with_suffix(f".{fmt}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi)
        WRITTEN.append(p)
    plt.close(fig)


apply_publication_style()


# ---------------------------------------------------------------- data access
def load_jsonl(pattern: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(pattern))):
        rows += [json.loads(line) for line in open(f) if line.strip()]
    if not rows:
        raise FileNotFoundError(pattern)
    return rows


def eval_rows(run: str, arm: str, rnd: int) -> list[dict]:
    return load_jsonl(RUNS / run / "arms" / arm / "rounds" / f"round_{rnd:02d}" / "eval" / "eval.shard*.jsonl")


def warmup_step51_rows(arm: str) -> list[dict]:
    return load_jsonl(RUNS / "kl50_warmup_backup" / arm / "round03" / "eval" / "eval.shard*.jsonl")


def is_strict(r: dict) -> bool:
    return bool(r["correct"] and r["has_boxed"])


def strict_rate(rows: list[dict]) -> float:
    return sum(map(is_strict, rows)) / len(rows)


def per_problem_strict(rows: list[dict]) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["id"]].append(1.0 if is_strict(r) else 0.0)
    return {k: float(np.mean(v)) for k, v in by.items()}


def paired_bootstrap(arm_rows: list[dict], ctrl_rows: list[dict]) -> tuple[float, float, float, int]:
    """Paired bootstrap over problems of the per-problem strict mean; fresh RNG (seed 0) per contrast. Points."""
    a, b = per_problem_strict(arm_rows), per_problem_strict(ctrl_rows)
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return 100 * d.mean(), 100 * lo, 100 * hi, len(d)


def termination_decomposition(x: list[dict], y: list[dict]) -> tuple[float, float, float, float]:
    """strict = f*a + (1-f)*t (f finish rate, a acc|finished, t acc|truncated). Exact split of strict(x)-strict(y):
    finish part = df*(abar-tbar); acc|finished part = fbar*da; acc|truncated part = (1-fbar)*dt  (bars = midpoints).
    Definition from notes/apod-critique-results.md section 3.6 (critic_checks.py decomp). Returns points."""
    def parts(rows):
        fin = [r for r in rows if not r["truncated"]]
        tr = [r for r in rows if r["truncated"]]
        return (len(fin) / len(rows), sum(map(is_strict, fin)) / max(1, len(fin)),
                sum(map(is_strict, tr)) / max(1, len(tr)), strict_rate(rows))
    fx, ax_, tx, sx = parts(x)
    fy, ay, ty, sy = parts(y)
    fbar, abar, tbar = (fx + fy) / 2, (ax_ + ay) / 2, (tx + ty) / 2
    return (100 * (sx - sy), 100 * (fx - fy) * (abar - tbar), 100 * fbar * (ax_ - ay), 100 * (1 - fbar) * (tx - ty))


def oracle_pool(run: str, dedup: bool) -> tuple[list[dict], int]:
    rows = load_jsonl(RUNS / run / "arms" / "kl_high" / "rounds" / "round_00" / "oracle" / "oracle_kl.shard*.jsonl")
    if not dedup:
        return rows, 0
    seen, kept = set(), []
    for r in rows:  # keep the first row per (example_index, rollout_index)
        k = (r["example_index"], r["rollout_index"])
        if k not in seen:
            seen.add(k)
            kept.append(r)
    return kept, len(rows) - len(kept)


def rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def spearman(x, y) -> float:
    rx, ry = rank(np.asarray(x)), rank(np.asarray(y))
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def question_groups(rows: list[dict]) -> list[list[dict]]:
    by: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["example_index"]].append(r)
    return [g for g in by.values() if len(g) >= 8]  # same filter as scan_selection.py


def check(label: str, got: float, want: float, tol: float) -> None:
    ok = abs(got - want) <= tol
    print(f"    check {label}: got {got:+.2f}, expected {want:+.2f} (tol {tol}) -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"mismatch: {label}")


PAIRED_TABLE = json.load(open(SCAN / "results.json"))["paired"]


def table_ci(key: str) -> tuple[float, float, float]:
    v = PAIRED_TABLE[key]
    return 100 * v["diff"], 100 * v["ci95"][0], 100 * v["ci95"][1]


# ---------------------------------------------------------------- Figure 0 (method schematic, no data)
def fig0() -> None:
    """Aligned-band selection for one question: 12 student rollouts -> teacher mean reverse KL s(y) -> sorted thirds
    (divergent / aligned band / redundant) -> train on the aligned band -> re-sample. Drawn like the figures4papers
    concept plots (figure_VIGIL/plot_concept.py): axis off, white rounded label boxes, black '->' arrows.
    The coordinate system is inches (axes fill the 6.5 x 1.6 in figure). Bar lengths and the score order are
    illustrative only."""
    print("FIG 0  method schematic (illustrative; no data values)")
    # A free-form canvas (axes = whole figure, coordinates in inches) needs no layout engine. The house style turns
    # constrained layout on globally, and savefig re-reads that rcParam after the first (pdf) save, so the whole
    # figure, including finalize_figure, runs with constrained layout switched off.
    with plt.rc_context({"figure.constrained_layout.use": False}):
        _draw_fig0()


def _draw_fig0() -> None:
    W, H = PAGE_W, 1.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fs, fs_head = 7.5, 8.0
    c_unscored = "#8C8C8C"
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=9, lw=1.3, color="black")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), shrinkA=0, shrinkB=0, **arrow_kw))

    def box(x0, y0, w, h, text, edge="black", face="white", lw=1.2, **text_kw):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=face, ec=edge,
                                    lw=lw, zorder=1))
        if text:
            ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, **text_kw)

    # relative rollout lengths (1.0 = token cap); three rollouts hit the cap
    lengths = np.array([0.62, 1.00, 0.45, 0.78, 0.55, 1.00, 0.70, 0.38, 0.86, 0.50, 1.00, 0.66])
    # illustrative score order: rollout indices from highest s(y) (rank 1) to lowest (rank 12)
    order = np.array([5, 0, 8, 3, 10, 6, 2, 9, 1, 11, 7, 4])
    bar_h = 0.052

    # 1) question
    y_mid = 0.80
    box(0.03, y_mid - 0.22, 0.56, 0.44, "question\n$x$")
    arrow(0.63, y_mid, 1.18, y_mid)
    ax.text(0.905, y_mid + 0.06, "student\nsamples 12", ha="center", va="bottom", fontsize=fs)

    # 2) 12 unscored rollouts with a dashed token cap
    x0_roll, span = 1.24, 0.85
    roll_y = 0.34 + np.arange(12)[::-1] * 0.083  # rollout 0 on top
    for i, (yv, ln) in enumerate(zip(roll_y, lengths)):
        ax.add_patch(plt.Rectangle((x0_roll, yv), ln * span, bar_h, fc=c_unscored, ec="black", lw=0.5, zorder=2))
    cap_x = x0_roll + span
    ax.plot([cap_x, cap_x], [roll_y.min() - 0.04, roll_y.max() + bar_h + 0.04], ls=(0, (3, 2)), lw=1.0,
            color="black", zorder=3)
    ax.text(cap_x, roll_y.max() + bar_h + 0.05, "token cap", ha="center", va="bottom", fontsize=fs)
    ax.text(x0_roll, roll_y.max() + bar_h + 0.05, "rollouts $y$", ha="left", va="bottom", fontsize=fs)

    # 3) teacher scoring
    arrow(2.20, y_mid, 2.74, y_mid)
    ax.text(2.47, y_mid + 0.06, "teacher\nscores $s(y)$", ha="center", va="bottom", fontsize=fs)
    ax.text(2.47, y_mid - 0.06, "mean\nreverse KL", ha="center", va="top", fontsize=fs)

    # 4) sorted into three groups of four
    x0_sort = 2.86
    gap = 0.07
    groups = [("divergent", "ranks 1\u20134", C_DIVERGENT), ("aligned band", "ranks 5\u20138", C_ALIGNED),
              ("redundant", "ranks 9\u201312", C_REDUNDANT)]
    pitch = 0.072
    top = 1.30
    group_y = []
    for g, (_, _, col) in enumerate(groups):
        ys = top - (np.arange(4) + 4 * g) * pitch - g * gap - bar_h
        for k, yv in enumerate(ys):
            ln = lengths[order[4 * g + k]]
            ax.add_patch(plt.Rectangle((x0_sort, yv), ln * span, bar_h, fc=col, ec="black", lw=0.5, zorder=2))
        group_y.append((ys.min(), ys.max() + bar_h))
    ax.plot([x0_sort + span] * 2, [group_y[-1][0] - 0.04, group_y[0][1] + 0.04], ls=(0, (3, 2)), lw=1.0,
            color="black", zorder=3)
    ax.text(x0_sort, group_y[0][1] + 0.05, "sorted by $s(y)$, high to low", ha="left", va="bottom", fontsize=fs)
    x_lbl = x0_sort + span + 0.10
    for (name, ranks, col), (lo, hi) in zip(groups, group_y):
        ax.text(x_lbl, (lo + hi) / 2, f"{name}\n{ranks}", ha="left", va="center", fontsize=fs, color=col,
                fontweight="bold" if col == C_ALIGNED else "normal", linespacing=1.1)
    lo, hi = group_y[1]
    keep_x0, keep_x1 = x0_sort - 0.06, x_lbl + 1.22
    box(keep_x0, lo - 0.035, keep_x1 - keep_x0, hi - lo + 0.07, "", edge=C_ALIGNED, face="#E7EEF6", lw=1.5)
    ax.text(keep_x1 - 0.04, (lo + hi) / 2, "keep", ha="right", va="center", fontsize=fs_head, color=C_ALIGNED,
            fontweight="bold")

    # 5) train on the kept band
    train_x0, train_w = 5.48, 0.97
    arrow(keep_x1 + 0.02, (lo + hi) / 2, train_x0 - 0.02, (lo + hi) / 2)
    box(train_x0, (lo + hi) / 2 - 0.25, train_w, 0.50, "train\n(reverse KL\nto teacher)", linespacing=1.1)

    # inset: the two baselines
    ax.text(train_x0 + train_w, 1.50, "uniform-k keeps 4 at random;\nfull keeps all 12", ha="right", va="top",
            fontsize=fs, color="#4D4D4D", linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.25", fc="#F2F2F2", ec="none"))

    # 6) loop back: re-sample from the updated student
    loop_y = 0.13
    x_tr = train_x0 + train_w / 2
    x_back = x0_roll + span / 2
    loop = MplPath([(x_tr, (lo + hi) / 2 - 0.25), (x_tr, loop_y), (x_back, loop_y), (x_back, roll_y.min() - 0.03)],
                   [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=loop, shrinkA=0, shrinkB=0, **arrow_kw))
    ax.text((x_tr + x_back) / 2, loop_y, "re-sample next round", ha="center", va="center", fontsize=fs,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=4)
    finalize_figure(fig, FIG / "fig0_method")


# ---------------------------------------------------------------- Figure 1
def fig1() -> None:
    print("FIG 1  MidAlign-8k (kl50) step 34, strict avg@4 (correct AND has_boxed) in %, recomputed from eval shards")
    arms = [("kl_mid", "aligned\nband", C_ALIGNED), ("kl_high", "divergent", C_DIVERGENT),
            ("random", "uniform-k", C_UNIFORM), ("kl_low", "redundant", C_REDUNDANT)]
    expected = {"kl_mid": 0.4745, "kl_high": 0.4320, "random": 0.3880, "kl_low": 0.3835}  # results_tables.md kl50
    vals = {}
    for arm, _, _ in arms:
        rows = eval_rows("kl50", arm, 2)
        vals[arm] = 100 * strict_rate(rows)  # percent
        print(f"  {arm:8s} n={len(rows)} strict {vals[arm]:.2f}%")
        check(f"fig1 {arm} strict %", vals[arm], 100 * expected[arm], 0.005)
    base = 100 * strict_rate(eval_rows("kl50", "kl_mid", 0))
    teacher = 100 * strict_rate(eval_rows("kl50_teacher", "all", 0))
    print(f"  base (kl50 round 0) strict {base:.2f}%; teacher (kl50_teacher, cap 8192) strict {teacher:.2f}%")
    check("fig1 base %", base, 27.85, 0.005)
    check("fig1 teacher %", teacher, 61.15, 0.005)
    u = vals["random"]
    print(f"  seed band around uniform-k: {u:.2f}% +/- {SEED_BAND:.2f} pts -> [{u - SEED_BAND:.2f}%, {u + SEED_BAND:.2f}%]")
    gap = teacher - base
    for arm in ("kl_mid", "random"):
        print(f"  {arm} closes {(vals[arm] - base) / gap:.1%} of the base->teacher gap")

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    x = np.arange(len(arms))
    ax.axhspan(u - SEED_BAND, u + SEED_BAND, color=C_BAND, alpha=0.45, zorder=0, lw=0)
    ax.text(len(arms) - 0.55, u + SEED_BAND + 0.4, "gray: seed spread\n(uniform-k \u00b1 6.7 pts)",
            ha="right", va="bottom", fontsize=7, color="#4D4D4D")
    bars = ax.bar(x, [vals[a] for a, _, _ in arms], width=0.62, color=[c for _, _, c in arms],
                  edgecolor="black", linewidth=1.0, zorder=2)
    annotate_bars(ax, bars, fmt="{:.2f}%")
    ax.axhline(base, ls="--", lw=1.4, color="black", alpha=0.45, zorder=1)
    ax.axhline(teacher, ls="--", lw=1.4, color="black", alpha=0.45, zorder=1)
    ax.text(len(arms) - 0.45, base, f"base\n{base:.2f}%", ha="left", va="center", fontsize=ANNOT_FS, clip_on=False)
    ax.text(len(arms) - 0.45, teacher, f"teacher\n{teacher:.2f}%", ha="left", va="center", fontsize=ANNOT_FS,
            clip_on=False)
    ax.set_xticks(x, [lbl for _, lbl, _ in arms])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.set_ylim(20, 65)
    ax.set_yticks([20, 30, 40, 50, 60])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("MATH-500 strict avg@4")
    finalize_figure(fig, FIG / "fig1_main")


# ---------------------------------------------------------------- Figure 2
def fig2() -> None:
    print("FIG 2  arm minus uniform-k, paired bootstrap 95% CI over problems (10k resamples, fresh seed-0 RNG per contrast), pts")
    print(f"  gray band: +/-{SEED_BAND:.3f} pts = 1.96 x {SEED_SD:.2f} (per-run SD of an arm-minus-uniform contrast, excl. kl_low)")
    # expected values: results_tables.md (steps 17/34, MidAlign-16k) and critic_checks section A (step 51)
    step51_expected = {"kl_mid": (0.45, -1.40, 2.35), "kl_high": (-2.50, -4.45, -0.60), "kl_low": (-1.00, -2.90, 0.95)}
    left = {}
    for arm in ("kl_mid", "kl_high", "kl_low"):
        pts = []
        for step, rnd in ((17, 1), (34, 2)):
            t = paired_bootstrap(eval_rows("kl50", arm, rnd), eval_rows("kl50", "random", rnd))
            ref = table_ci(f"kl50/{arm}-vs-random/{rnd}")
            print(f"  MidAlign-8k {arm:7s} step {step}: {t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}] n={t[3]}   "
                  f"(results_tables: {ref[0]:+.2f} [{ref[1]:+.2f}, {ref[2]:+.2f}])")
            check(f"fig2 kl50 {arm} s{step} delta", t[0], ref[0], 0.005)
            pts.append((step, t))
        t = paired_bootstrap(warmup_step51_rows(arm), warmup_step51_rows("random"))
        ref = step51_expected[arm]
        print(f"  MidAlign-8k {arm:7s} step 51 (warmup-variant retrain): {t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}] n={t[3]}   "
              f"(critic_checks A: {ref[0]:+.2f} [{ref[1]:+.2f}, {ref[2]:+.2f}])")
        check(f"fig2 kl50 {arm} s51 delta", t[0], ref[0], 0.005)
        pts.append((51, t))
        left[arm] = pts
    right = {}
    for run in ("oracle16k", "oracle16k_seed2"):
        pts = []
        for rnd in (1, 2, 3):
            t = paired_bootstrap(eval_rows(run, "kl_mid", rnd), eval_rows(run, "random", rnd))
            ref = table_ci(f"{run}/kl_mid-vs-random/{rnd}")
            print(f"  MidAlign-16k {'seed A' if run == 'oracle16k' else 'seed B'} ({run}) aligned round {rnd}: "
                  f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}] n={t[3]}   (results_tables: {ref[0]:+.2f} [{ref[1]:+.2f}, {ref[2]:+.2f}])")
            check(f"fig2 {run} r{rnd} delta", t[0], ref[0], 0.005)
            pts.append((rnd, t))
        right[run] = pts

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(PAGE_W, 2.3), gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in (axl, axr):
        ax.axhspan(-SEED_BAND, SEED_BAND, color=C_BAND, alpha=0.45, zorder=0, lw=0)
        ax.axhline(0, color="black", lw=1.0, alpha=0.6, zorder=1)
        ax.set_ylim(-11, 13.5)
        ax.set_yticks([-10, -5, 0, 5, 10])
    # band label placed where no data point sits: upper right (left panel), upper centre (right panel)
    for ax, xa, ha in ((axl, 0.99, "right"), (axr, 0.5, "center")):
        ax.text(xa, 0.99, "gray band: seed-to-seed spread (\u00b11.96 \u00d7 3.4)", transform=ax.transAxes,
                ha=ha, va="top", fontsize=7, color="#4D4D4D")

    line_kw = dict(lw=1.8, alpha=0.9, zorder=2)
    err_kw = dict(ms=5.5, mew=1.4, elinewidth=1.3, capsize=2.5, zorder=3)
    series = [("kl_mid", "aligned band", C_ALIGNED, "o", -2.0),
              ("kl_high", "divergent", C_DIVERGENT, "s", 0.0),
              ("kl_low", "redundant", C_REDUNDANT, "^", 2.0)]
    for arm, label, col, mk, dx in series:
        pts = left[arm]
        xs = [s + dx for s, _ in pts]
        axl.plot(xs, [t[0] for _, t in pts], color=col, **line_kw)
        for (s, t), xv in zip(pts, xs):
            hollow = s == 51
            axl.errorbar(xv, t[0], yerr=[[t[0] - t[1]], [t[2] - t[0]]], fmt=mk, color=col,
                         mfc="white" if hollow else col, mec=col, **err_kw)
    axl.set_xticks([17, 34, 51], ["17", "34", "51\u2020"])
    axl.set_xlim(10, 58)
    axl.set_xlabel("MidAlign-8k training step  (\u2020 hollow: warmup-variant retrain)")
    axl.set_ylabel("arm \u2212 uniform-k (pts)")
    left_handles = [Line2D([], [], color=c, marker=m, lw=1.8, ms=5.5, label=lbl) for _, lbl, c, m, _ in series]

    for run, col, mk, dx in (("oracle16k", C_ALIGNED, "o", -0.06), ("oracle16k_seed2", C_ALIGNED_SEED_B, "D", 0.06)):
        pts = right[run]
        xs = [r + dx for r, _ in pts]
        axr.plot(xs, [t[0] for _, t in pts], color=col, **line_kw)
        for (_, t), xv in zip(pts, xs):
            axr.errorbar(xv, t[0], yerr=[[t[0] - t[1]], [t[2] - t[0]]], fmt=mk, color=col, **err_kw)
    axr.set_xticks([1, 2, 3])
    axr.set_xlim(0.6, 3.4)
    axr.set_xlabel("MidAlign-16k round")
    axr.set_ylabel("aligned \u2212 uniform-k (pts)")
    right_handles = [Line2D([], [], color=C_ALIGNED, marker="o", lw=1.8, ms=5.5, label="seed A"),
                     Line2D([], [], color=C_ALIGNED_SEED_B, marker="D", lw=1.8, ms=5.5, label="seed B")]
    for ax, handles in ((axl, left_handles), (axr, right_handles)):
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(handles),
                  handlelength=2.0, columnspacing=1.4, borderaxespad=0.2)
    finalize_figure(fig, FIG / "fig2_timecourse")


# ---------------------------------------------------------------- Figure 3
def fig3() -> None:
    print("FIG 3  termination decomposition of delta strict (pts): total = finish + acc|finished + acc|truncated")
    r3_random_pooled = sum((eval_rows("r3-qentropy-8k", "random_questions", k) for k in range(5, 11)), [])
    r1_both_wrong_pooled = sum((eval_rows("r1-correctness-8k", "both_wrong", k) for k in range(5, 11)), [])
    rows = [
        ("aligned vs uniform-k\n(MidAlign-8k, step 34)",
         eval_rows("kl50", "kl_mid", 2), eval_rows("kl50", "random", 2), (8.65, 9.37, 0.01, -0.72)),
        ("aligned vs base\n(MidAlign-8k, step 34)",
         eval_rows("kl50", "kl_mid", 2), eval_rows("kl50", "kl_mid", 0), (19.60, 19.49, -0.34, 0.45)),
        ("full vs uniform-k\n(round 8)",
         eval_rows("apod", "all", 8), eval_rows("apod", "random_top4", 8), (3.45, 4.69, -0.09, -1.14)),
        ("both fail vs random\nquestions (steps 50–100)",
         r1_both_wrong_pooled, r3_random_pooled, (-2.59, -1.42, 0.01, -1.18)),
    ]  # expected = notes/apod-critique-results.md section 3.6 / critic_checks.py section D
    results = []
    for label, x, y, want in rows:
        t = termination_decomposition(x, y)
        name = label.replace("\n", " ")
        print(f"  {name:55s} total {t[0]:+.2f} finish {t[1]:+.2f} acc|fin {t[2]:+.2f} acc|trunc {t[3]:+.2f}"
              f"  (n {len(x)} vs {len(y)}; sum of parts {t[1] + t[2] + t[3]:+.2f})")
        for part, g, w in zip(("total", "finish", "acc|fin", "acc|trunc"), t, want):
            check(f"fig3 {name} {part}", g, w, 0.05)
        results.append((label, t))

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    # hatches keep the three parts distinct in grayscale print (house style "print-safe bars")
    comps = [(1, "finish rate", C_FINISH, ""), (2, "accuracy given finished", C_ACCFIN, "///"),
             (3, "accuracy given truncated", C_ACCTRUNC, "...")]
    ypos = np.arange(len(results))[::-1]
    for yv, (_, t) in zip(ypos, results):
        pos, neg = 0.0, 0.0
        for idx, _, col, hatch in comps:
            v = t[idx]
            left = pos if v >= 0 else neg + v
            ax.barh(yv, abs(v), left=left, height=0.58, color=col, hatch=hatch, edgecolor="black", linewidth=0.8,
                    zorder=2)
            if v >= 0:
                pos += v
            else:
                neg += v
        ax.plot(t[0], yv, marker="D", ms=5, color="black", mec="white", mew=0.6, zorder=3)
        ax.text(max(pos, t[0]) + 0.6, yv, f"{t[0]:+.2f}".replace("-", "\u2212"), va="center", ha="left",
                fontsize=ANNOT_FS)
    ax.axvline(0, color="black", lw=1.0, zorder=1)
    ax.set_yticks(ypos, [lbl for lbl, _ in results])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("\u0394 strict avg@4 (pts)")
    ax.set_xlim(-6, 26)
    handles = [Patch(facecolor=c, hatch=h, edgecolor="black", linewidth=0.8, label=l) for _, l, c, h in comps]
    handles.append(Line2D([], [], marker="D", color="black", lw=0, ms=5, label="net \u0394"))
    fig.legend(handles=handles, loc="outside upper center", ncol=2, handlelength=1.4, columnspacing=1.0,
               labelspacing=0.3)
    finalize_figure(fig, FIG / "fig3_termination")


# ---------------------------------------------------------------- Figure 4
def fig4() -> None:
    print("FIG 4  student entropy vs reverse KL within a question (round-0 oracle pools; kl_high copy of the shared pool)")
    pools = [("kl50", "MidAlign-8k", (2.60, 1.10)), ("oracle8k", "earlier 8k run", (2.61, 1.10)),
             ("oracle16k", "MidAlign-16k A", (2.72, 1.08)), ("oracle16k_seed2", "MidAlign-16k B", (2.66, 0.95)),
             ("r2-trajsel-8k", "MidAlign-long", (2.54, 1.11))]
    # expected: notes/apod-critique-verify.md (kl50 de-duplicated) and selection_tables.md section 1 (other pools)
    overlaps = {}
    rho16 = None
    for run, name, (want_top, want_mid) in pools:
        rows, n_dup = oracle_pool(run, dedup=True)
        groups = question_groups(rows)
        ov_top, ov_mid, ov_low, rhos = [], [], [], []
        for g in groups:
            k = np.array([r["mean_reverse_kl"] for r in g])
            h = np.array([r["student_entropy"] for r in g])
            rhos.append(spearman(h, k))
            n = len(g)
            h_top = set(np.argsort(-h)[:4])
            k_order = np.argsort(-k)
            ov_top.append(len(h_top & set(k_order[:4])))
            ov_mid.append(len(h_top & set(k_order[(n - 4) // 2:(n - 4) // 2 + 4])))
            ov_low.append(len(h_top & set(k_order[n - 4:])))
        rhos = np.array([x for x in rhos if not np.isnan(x)])
        overlaps[name] = (np.mean(ov_top), np.mean(ov_mid), np.mean(ov_low))
        print(f"  {name:18s} ({run}): rows {len(rows)} (duplicates dropped {n_dup}), questions {len(groups)}; "
              f"Spearman median {np.median(rhos):+.2f} [IQR {np.quantile(rhos, .25):+.2f}, {np.quantile(rhos, .75):+.2f}]; "
              f"entropy-top4 overlap: KL-top {overlaps[name][0]:.2f}/4, KL-mid {overlaps[name][1]:.2f}/4, KL-low {overlaps[name][2]:.2f}/4")
        check(f"fig4 {name} KL-top overlap", overlaps[name][0], want_top, 0.005)
        check(f"fig4 {name} KL-mid overlap", overlaps[name][1], want_mid, 0.005)
        if run == "oracle16k":
            rho16 = rhos
    print(f"  oracle16k round-0 per-question Spearman: n={len(rho16)}, median {np.median(rho16):+.2f}, "
          f"mean {np.mean(rho16):+.2f}, min {rho16.min():+.2f}, max {rho16.max():+.2f}, "
          f"fraction > 0: {np.mean(rho16 > 0):.3f}")
    print("  chance overlap for 4 of 12: 4*4/12 = 1.33/4")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(COL_W, 2.2), gridspec_kw={"width_ratios": [1, 1.25]})
    axl.hist(rho16, bins=np.linspace(-1, 1, 21), color=PALETTE["blue_secondary"], edgecolor="black", linewidth=0.7)
    med = float(np.median(rho16))
    axl.axvline(med, color="black", ls="--", lw=1.2, alpha=0.7)
    axl.text(med - 0.06, axl.get_ylim()[1] * 0.97, f"median\n{med:+.2f}", ha="right", va="top", fontsize=ANNOT_FS)
    axl.set_xlabel("Spearman \u03c1 (entropy, rev. KL)")
    axl.set_ylabel("questions")
    axl.set_xlim(-1, 1)
    axl.set_xticks([-1, 0, 1])

    names = [name for _, name, _ in pools]
    w = 0.16
    for i, (name, col, hatch) in enumerate(zip(names, C_POOLS, H_POOLS)):
        axr.bar(np.arange(3) + (i - 2) * w, overlaps[name], width=w, color=col, edgecolor="black", linewidth=0.6,
                hatch=hatch, label=name)
    axr.axhline(4 * 4 / 12, color="black", ls="--", lw=1.2, alpha=0.7)
    axr.text(2.5, 4 / 3 + 0.06, "chance", ha="right", va="bottom", fontsize=ANNOT_FS)
    axr.set_xticks(np.arange(3), ["KL-top", "KL-mid", "KL-low"])
    axr.tick_params(axis="x", length=0)
    axr.set_ylabel("overlap with entropy-top-4")
    axr.set_ylim(0, 4)
    fig.legend(*axr.get_legend_handles_labels(), loc="outside upper center", ncol=3, handlelength=1.0,
               columnspacing=0.9, labelspacing=0.25)
    finalize_figure(fig, FIG / "fig4_entropy")


README_HEAD = """# Figures

Generated by `../make_figures.py` (run from `papers/apod-acl`:
`uv run --no-project --with matplotlib --with numpy python make_figures.py`). This file is regenerated on every run;
the "Printed numbers" section is the script's stdout verbatim. The script reads only `outputs/runs_results/` and
`experiments/apod-data-scan/results.json`, and exits with an error if any recomputed number differs from the
reference tables (`experiments/apod-data-scan/results_tables.md`, `selection_tables.md`,
`notes/apod-critique-verify.md`, and the `critic_checks.py` output in `notes/apod-critique-results.md`).

Each figure is written as a vector `.pdf` (for LaTeX) and a 300-dpi `.png` (for slides and previews).

Units: accuracies (strict avg@4, base and teacher lines) are percentages, the exact x100 conversion of the
4-decimal strict rate printed with 2 decimals (0.4745 -> 47.45%; base 27.85%, teacher 61.15%). Differences
(arm minus uniform-k, termination decomposition, seed band) are percentage points (pts).

Style credit: the figures follow the figures4papers house style by Chen Liu
(https://github.com/ChenLiu-1996/figures4papers, local clone `papers/figures4papers`), licensed CC BY-NC 4.0. This
covers the semantic PALETTE (blues for our method, red for the baseline), FigureStyle / apply_publication_style
(no top or right spines, frameless legends, sans-serif Helvetica/DejaVu text, thick axes), annotate_bars and
finalize_figure. Font sizes are scaled down from the repo presets so text stays at 7 pt or larger at ACL column width.

Strict = `correct AND has_boxed`. Bootstrap CIs: paired over the 500 problems, 10k resamples, a fresh
`numpy.random.default_rng(0)` per contrast. The point estimates match the reference tables exactly; the CI endpoints
differ from `results_tables.md` / `critic_checks.py` by at most 0.10 pt (see the parenthesised reference values below),
because those scripts drew every contrast from one shared RNG stream. Quote the table CIs in the text.

| Figure | Content | Data sources (under `outputs/runs_results/`) |
|---|---|---|
| `fig0_method.pdf` (6.5 × 1.6 in) | Method schematic, no data: question x → student samples 12 rollouts (three hit the dashed token cap) → teacher scores each by mean reverse KL s(y) → sorted into ranks 1–4 divergent (teal), 5–8 aligned band (blue, boxed, "keep"), 9–12 redundant (violet) → train (reverse KL to teacher) → re-sample next round. Inset: uniform-k keeps 4 at random; full keeps all 12. Bar lengths and score order are illustrative | none |
| `fig1_main.pdf` (3.2 in) | MidAlign-8k step 34 strict avg@4 (%): aligned band (blue), divergent (teal), uniform-k (red), redundant (violet); dashed base and teacher; gray band = uniform-k ± 6.7 pts seed spread (story §4 "Fig 1 with seed band", §6.2) | `kl50/arms/{kl_mid,kl_high,random,kl_low}/rounds/round_02/eval/eval.shard*.jsonl`; base `kl50/arms/kl_mid/rounds/round_00/eval`; teacher `kl50_teacher/arms/all/rounds/round_00/eval` |
| `fig2_timecourse.pdf` (6.5 in) | Left: MidAlign-8k aligned/divergent/redundant − uniform-k at steps 17, 34, 51 (51 hollow = warmup-variant retrain). Right: MidAlign-16k seed A/B aligned − uniform-k, rounds 1–3. Error bars: paired bootstrap 95% CI. Gray band ±6.7 pts = 1.96 × 3.4 (per-run SD of an arm-minus-uniform contrast across seeds, kl_low excluded; `notes/apod-critique-results.md` §1) | `kl50/arms/*/rounds/round_0{1,2}/eval`; `kl50_warmup_backup/*/round03/eval`; `oracle16k/arms/{kl_mid,random}/rounds/round_0{1,2,3}/eval`; `oracle16k_seed2/arms/{kl_mid,random}/rounds/round_0{1,2,3}/eval` |
| `fig3_termination.pdf` (3.2 in) | Stacked bars: Δstrict = finish-rate part + accuracy\\|finished part + accuracy\\|truncated part (definition: `notes/apod-critique-results.md` §3.6); diamond = net Δ | `kl50/arms/{kl_mid,random}/rounds/round_02/eval`, `kl50/arms/kl_mid/rounds/round_00/eval`; `apod/arms/{all,random_top4}/rounds/round_08/eval`; `r1-correctness-8k/arms/both_wrong/rounds/round_05..10/eval` vs `r3-qentropy-8k/arms/random_questions/rounds/round_05..10/eval` (steps 50–100 pooled) |
| `fig4_entropy.pdf` (3.2 in) | Left: per-question Spearman(student entropy, mean reverse KL), MidAlign-16k seed A round-0 pool. Right: mean overlap of entropy-top-4 with KL-top-4 / KL-mid-4 (ranks 5–8) / KL-low-4 for the five round-0 pools; dashed chance 1.33 | `{kl50,oracle8k,oracle16k,oracle16k_seed2,r2-trajsel-8k}/arms/kl_high/rounds/round_00/oracle/oracle_kl.shard*.jsonl` (kl50 de-duplicated by `(example_index, rollout_index)`, first row kept) |

Paper names: kl50 = MidAlign-8k; oracle8k = MidAlign-8k pilot; oracle16k / oracle16k_seed2 = MidAlign-16k seed A / B;
r2-trajsel-8k = MidAlign-long; kl_mid / kl_high / kl_low = aligned band / divergent / redundant; random, random4,
random_top4 = uniform-k; all = full. The fig3 "full vs uniform-k (round 8)" row is the `apod` run (cap 8192);
"both fail vs random questions" is r1 `both_wrong` vs the shared r3 `random_questions` control.

## Printed numbers

```
"""


def main() -> None:
    FIG.mkdir(exist_ok=True)
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            buf.write(s)
            return __import__("sys").__stdout__.write(s)

    with redirect_stdout(Tee()):
        for f in (fig0, fig1, fig2, fig3, fig4):
            f()
        for p in WRITTEN:
            print(f"wrote {p.relative_to(HERE)} ({p.stat().st_size} bytes)")
    (FIG / "README.md").write_text(README_HEAD + buf.getvalue() + "```\n")


if __name__ == "__main__":
    main()
