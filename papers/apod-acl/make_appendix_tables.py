"""Generate appendix_tables.tex (Appendix F: full per-step tables for every run).

Run from papers/apod-acl:  python3 make_appendix_tables.py
Source: experiments/apod-data-scan/results_tables.md (copied mechanically; no number is recomputed or rounded).
Units: rates (strict, loose, pass@n, cap-hit, acc|finished) are stored there as fractions with 3-4 decimals and are
written here as percentages with 2 decimals by an exact x100 decimal shift (0.4745 -> 47.45, 0.406 -> 40.60);
differences and CIs are already percentage points and are copied verbatim.
Repo run/arm names are mapped to paper names (papers/apod-story.md section 0); the repo name is kept in brackets.
"""
from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "experiments" / "apod-data-scan" / "results_tables.md"
OUT = HERE / "appendix_tables.tex"

RUN_NAMES = {
    "apod": "Entropy run (repo: apod; entropy-top-4 / uniform-k / full; 8 rounds, cap 8192)",
    "kl50": "MidAlign-8k (repo: kl50; cap 8192)",
    "kl50_teacher": "Teacher at cap 8192 (repo: kl50\\_teacher)",
    "kl50w": "Learning-rate search run, round-0 base evaluation only (repo: kl50w)",
    "oracle16k": "MidAlign-16k seed A (repo: oracle16k; cap 16384)",
    "oracle16k_lr2e5": "LR probe: MidAlign-16k aligned band, round 2 retrained at 2e-5 (repo: oracle16k\\_lr2e5)",
    "oracle16k_lr5e6": "LR probe: MidAlign-16k aligned band, round 2 retrained at 5e-6 (repo: oracle16k\\_lr5e6)",
    "oracle16k_seed2": "MidAlign-16k seed B (repo: oracle16k\\_seed2; cap 16384)",
    "oracle16k_teacher": "Teacher at cap 16384 (repo: oracle16k\\_teacher)",
    "oracle8k": "Earlier 8k run, one evaluated round (repo: oracle8k; cap 8192)",
    "r1-correctness-8k": "Questions selected by teacher and student correctness (repo: r1-correctness-8k; cap 8192, 100 steps)",
    "r2-trajsel-8k": "MidAlign-long (repo: r2-trajsel-8k; cap 8192, 100 steps)",
    "r3-qentropy-8k": "Most uncertain 800 questions and the random-question control (repo: r3-qentropy-8k; cap 8192, 100 steps)",
    "r4-qentropy-low-8k": "Least uncertain 800 questions (repo: r4-qentropy-low-8k; cap 8192, 100 steps)",
    "teacher_eval": "Teacher gate eval (100 problems x 1 sample, cap 8192)",
}
ARM_NAMES = {
    "kl_mid": "aligned band",
    "kl_high": "divergent",
    "kl_low": "redundant",
    "random": "uniform-k",
    "random4": "uniform-k",
    "random_top4": "uniform-k",
    "all": "full",
    "entropy_top4": "entropy-top-4",
    "teacher_right_student_wrong": "only teacher solves",
    "both_right": "both solve",
    "both_wrong": "both fail",
    "mixed": "mixed",
    "random_questions": "random questions",
    "uncertain_questions": "uncertain questions",
    "certain_questions": "certain questions",
}


def arm(a: str) -> str:
    a = a.strip()
    if "/" in a:  # e.g. r1-correctness-8k/both_wrong
        run, rest = a.split("/", 1)
        return f"{run.split('-')[0]} {ARM_NAMES.get(rest, rest)}"
    return ARM_NAMES.get(a, a.replace("_", "\\_"))


def cell(s: str) -> str:
    s = s.strip().replace("\\|", "|")
    s = s.replace("−", "-")
    s = re.sub(r"(?<![\w.])-(?=\d)", "$-$", s)
    s = s.replace("×", "$\\times$")
    return s


def pct(s: str) -> str:
    """Exact fraction -> percent, keeping the source precision (0.4745 -> 47.45, 0.778 -> 77.8, 0.6000 -> 60.00).

    Refuses any value that would need rounding."""
    s = s.strip()
    m = re.fullmatch(r"[01]\.(\d{1,4})", s)
    if not m:
        raise SystemExit(f"unexpected rate cell: {s!r}")
    places = max(len(m.group(1)) - 2, 0)
    v = Decimal(s) * 100
    out = v.quantize(Decimal(1).scaleb(-places))
    if out != v:
        raise SystemExit(f"rate {s} is not exactly representable with {places} decimals in percent")
    return f"{out:.{places}f}"


def rows_of(block: list[str]) -> list[list[str]]:
    out = []
    for ln in block:
        if not ln.startswith("|") or set(ln.replace("|", "").strip()) <= set("-: "):
            continue
        parts = [p for p in re.split(r"(?<!\\)\|", ln)[1:-1]]
        out.append(parts)
    return out


def emit_table(caption: str, header: list[str], body: list[list[str]], fmt: str, arm_cols=(0,), pct_cols=(), label: str | None = None) -> str:
    lab = f"\\label{{{label}}}" if label else ""
    lines = [f"\\begin{{longtable}}{{{fmt}}}", f"\\caption{{{caption}}}{lab}\\\\", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule", "\\endfirsthead", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
    prev = None
    for r in body:
        r = list(r)
        for c in arm_cols:
            r[c] = arm(r[c])
        for c in pct_cols:
            r[c] = pct(r[c])
        if arm_cols and prev is not None and r[arm_cols[0]] != prev:
            lines.append("\\midrule")
        prev = r[arm_cols[0]] if arm_cols else None
        lines.append(" & ".join(cell(x) if i not in arm_cols else x for i, x in enumerate(r)) + " \\\\")
    lines.append("\\end{longtable}")
    return "\n".join(lines)


def main() -> None:
    text = SRC.read_text().splitlines()
    sections: list[tuple[str, list[str]]] = []
    cur = None
    for ln in text:
        if ln.startswith("## "):
            cur = (ln[3:].strip(), [])
            sections.append(cur)
        elif cur is not None:
            cur[1].append(ln)

    out = ["% AUTO-GENERATED by make_appendix_tables.py from experiments/apod-data-scan/results_tables.md. Do not edit by hand.",
           "% src: results_tables.md (rates = source fractions x100, exact decimal shift, source precision kept; differences and CIs in points copied verbatim)"]
    for name, block in sections:
        # split block into sub-tables at blank-line separated groups that start with '|'
        groups, g = [], []
        for ln in block:
            if ln.startswith("|"):
                g.append(ln)
            else:
                if g:
                    groups.append(g)
                    g = []
        if g:
            groups.append(g)
        if name in RUN_NAMES:
            title = RUN_NAMES[name]
            for gi, grp in enumerate(groups):
                rows = rows_of(grp)
                if len(rows) <= 1:
                    continue
                hdr, body = rows[0], rows[1:]
                if hdr[0].strip() == "arm" and "strict" in [h.strip() for h in hdr]:
                    header = ["arm", "when", "prob.$\\times$samp.", "strict", "loose", "pass@n", "cap-hit", "acc$|$fin.", "mean len"]
                    out.append(emit_table(f"{title}: per-evaluation metrics. Strict = correct and \\texttt{{\\textbackslash boxed\\{{\\}}}}. "
                                          "Strict, loose, pass@n, cap-hit and acc$|$fin. in \\%; mean len in tokens.",
                                          header, body, "llrrrrrrr", pct_cols=(3, 4, 5, 6, 7),
                                          label=f"tab:app-metrics-{name}"))
                else:
                    header = ["arm", "when", "$n$", "$\\Delta$ (pts)", "95\\% CI (pts)", "$p$ (boot)"]
                    ctl = "uniform-k"
                    out.append(emit_table(f"{title}: paired difference vs {ctl} (strict; bootstrap over problems, 10k resamples).",
                                          header, body, "llrrrr", label=f"tab:app-diff-{name}"))
        elif name.startswith("r1/r3/r4"):
            rows = rows_of(groups[0])
            out.append(emit_table("Question-level arms vs the shared random-question control (run with the most-uncertain arm; repo r1/r3/r4 prefixes), strict.",
                                  ["run / arm", "$\\Delta$ step 100 [95\\% CI]", "$\\Delta$ pooled steps 50--100 [95\\% CI]"],
                                  rows[1:], "lrr", label="tab:app-qlevel"))
        elif name.startswith("r2 pooled"):
            rows = rows_of(groups[0])
            out.append(emit_table("MidAlign-long: pooled steps 50--100 vs uniform-k, strict.",
                                  ["arm", "$\\Delta$ pooled (pts) [95\\% CI]"], rows[1:], "lr", label="tab:app-long-pooled"))
        elif name.startswith("Noise floor"):
            rows = rows_of(groups[0])
            out.append(emit_table("Seed noise: MidAlign-16k seed A minus seed B, same arm and round (identical configuration and questions, different sampling and training seed).",
                                  ["arm", "round", "$\\Delta$ seed A $-$ seed B (pts) [95\\% CI]"], rows[1:], "llr", label="tab:app-seednoise"))
        elif name.startswith("Base-model"):
            rows = rows_of(groups[0])
            out.append(emit_table("Base-model round-0 evaluations across runs (same model, same 500 problems, different processes and seeds).",
                                  ["run", "strict (\\%)", "cap-hit (\\%)"], rows[1:], "lrr", arm_cols=(), pct_cols=(1, 2)))
    OUT.write_text("\n\n".join(out) + "\n")
    print(f"wrote {OUT} ({len(out) - 2} tables)")


if __name__ == "__main__":
    main()
