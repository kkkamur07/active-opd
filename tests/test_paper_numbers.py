"""Golden test: the numbers printed in the paper are exactly what the analysis
computes from the committed per-run metrics (results/runs/)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("paper_numbers", ROOT / "experiments" / "paper_numbers.py")
assert _spec is not None and _spec.loader is not None
paper_numbers = importlib.util.module_from_spec(_spec)
sys.modules["paper_numbers"] = paper_numbers
_spec.loader.exec_module(paper_numbers)

PAPER = (ROOT / "paper" / "main.tex").read_text()
COMPUTED = paper_numbers.compute(ROOT / "results" / "runs", full=False)


def _mean_sd_pairs() -> set[tuple[float, float]]:
    pairs: set[tuple[float, float]] = set()
    for table in ("mean_over_steps_15_60", "final_checkpoint_step_60"):
        for entry in COMPUTED[table].values():
            for value in entry.values():
                if isinstance(value, dict):
                    pairs.add((abs(value["mean"]), value["sd"]))
    return pairs


def test_every_mean_sd_in_the_paper_is_computed() -> None:
    printed = {(abs(float(m)), float(s)) for m, s in re.findall(r"([+-]?\d+\.\d\d) \\pm (\d+\.\d\d)", PAPER)}
    assert len(printed) > 40
    assert printed <= _mean_sd_pairs(), sorted(printed - _mean_sd_pairs())


def test_per_seed_table_matches() -> None:
    table = COMPUTED["mean_over_steps_15_60"]
    rows = re.findall(r"^(.+?) & \$([+-]\d\.\d\d)\$ & \$([+-]\d\.\d\d)\$ & \$([+-]\d\.\d\d)\$ \\\\$", PAPER, re.M)
    assert len(rows) == 12
    computed = {tuple(e["avg4_delta_per_seed"]) for arm, e in table.items() if arm != paper_numbers.BASELINE}
    assert {tuple(float(v) for v in row[1:]) for row in rows} == computed


def test_committed_numbers_are_current() -> None:
    committed = json.loads((ROOT / "results" / "paper_numbers.json").read_text())
    for key, value in COMPUTED.items():
        assert committed[key] == value, key
