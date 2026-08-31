"""Run-directory layout: the single source of the on-disk path convention.

Every stage, driver, and analysis script builds round paths through these
helpers, so the layout (``<run>/arms/<arm>/rounds/round_<XX>``) has exactly
one definition. Model-resolution logic (round-0 base-model fallback, weight
presence checks) deliberately lives with its consumers, not here.
"""

from __future__ import annotations

from pathlib import Path


def round_dir(run_dir: Path, arm: str, round_index: int) -> Path:
    """``<run_dir>/arms/<arm>/rounds/round_<round_index:02d>``."""

    return Path(run_dir) / "arms" / arm / "rounds" / f"round_{round_index:02d}"


def checkpoint_dir(run_dir: Path, arm: str, round_index: int) -> Path:
    """The checkpoint written by ``round_index``'s train stage."""

    return round_dir(run_dir, arm, round_index) / "checkpoint"


if __name__ == "__main__":
    # Self-test: byte-identical to the literals these helpers replaced.
    assert str(round_dir(Path("/r"), "kl_mid", 1)) == "/r/arms/kl_mid/rounds/round_01"
    assert str(round_dir(Path("/r"), "entropy_top4", 0)) == "/r/arms/entropy_top4/rounds/round_00"
    assert str(checkpoint_dir(Path("/r"), "all", 12)) == "/r/arms/all/rounds/round_12/checkpoint"
    print("apod/paths.py self-test passed")
