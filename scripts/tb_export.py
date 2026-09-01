"""Export per-step training logs to TensorBoard, retroactively.

The train stage already records everything per optimizer step
(logging_steps=1 -> train/log_history.jsonl with loss, grad_norm,
learning_rate, epoch, step), so no trainer changes and no retraining are
needed: this walks arms/*/rounds/round_*/train/log_history.jsonl and writes
event files under <run-dir>/tb/<arm>.

The x-axis is the CUMULATIVE optimizer step across rounds (each round is a
fresh GKDTrainer whose global_step restarts at 1, so raw steps from
different rounds would overwrite each other). One TB run per arm, so arms
overlay in the UI. Round boundaries are visible as train/round step
functions, and each round's eval/summary.json numerics are logged at that
round's starting step.

Requires the tensorboard package (uv add tensorboard), then:
  uv run python scripts/tb_export.py --run-dir outputs/runs/oracle16k
  tensorboard --logdir outputs/runs/oracle16k/tb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCALARS = {
    "loss": "train/loss",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/lr",
    # Per-step batch diagnostics written by apod.stages.train.DiagGKDTrainer
    # (absent in log histories that predate them; exported when present).
    "overlap_ratio_top16": "train/overlap_ratio_top16",
    "overlap_adv_top16": "train/overlap_adv_top16",
    "abs_entropy_gap": "train/abs_entropy_gap",
    "response_tokens": "train/response_tokens",
    "cap_hit_frac": "train/cap_hit_frac",
    "bf16_rounded_frac": "train/bf16_rounded_frac",
    "bf16_rounded_frac_embeddings": "train/bf16_rounded_frac_embeddings",
    "bf16_rounded_frac_attention": "train/bf16_rounded_frac_attention",
    "bf16_rounded_frac_mlp": "train/bf16_rounded_frac_mlp",
    "bf16_rounded_frac_lm_head": "train/bf16_rounded_frac_lm_head",
    "bf16_rounded_frac_other": "train/bf16_rounded_frac_other",
}
# Offline oracle scoring is per ROUND, not per step: one TB point per round,
# at the round's starting cumulative step (same convention as eval/*).
ORACLE_KEYS = ("mean_reverse_kl", "mean_forward_kl", "overlap_ratio_top16", "overlap_adv_top16")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="event-file root (default <run-dir>/tb)")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def oracle_means(rdir: Path) -> dict[str, float]:
    rows: dict[tuple[int, int], dict] = {}
    for shard_file in sorted((rdir / "oracle").glob("oracle_kl.shard*.jsonl")):
        for r in read_jsonl(shard_file):
            rows[(r["example_index"], r["rollout_index"])] = r  # last row wins (resume re-scores)
    means = {}
    for key in ORACLE_KEYS:
        vals = [r[key] for r in rows.values() if key in r]
        if vals:
            means[key] = sum(vals) / len(vals)
    return means


def export_arm(writer, arm_dir: Path) -> int:
    offset = 0  # cumulative optimizer steps across this arm's rounds
    exported = 0
    for rdir in sorted((arm_dir / "rounds").glob("round_*")):
        round_index = int(rdir.name.split("_")[1])

        # Eval of the round's STARTING model, pinned to the step count so far.
        eval_summary = rdir / "eval" / "summary.json"
        if eval_summary.exists():
            for key, value in json.loads(eval_summary.read_text()).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    writer.add_scalar(f"eval/{key}", value, offset)

        for key, value in oracle_means(rdir).items():
            writer.add_scalar(f"oracle/{key}", value, offset)

        history = rdir / "train" / "log_history.jsonl"
        if not history.exists():
            continue
        last_step = 0
        for row in read_jsonl(history):
            step = int(row.get("step", 0))
            if step <= 0:
                continue  # summary rows (train_runtime etc.) carry no step
            last_step = max(last_step, step)
            for key, tag in SCALARS.items():
                if key in row:
                    writer.add_scalar(tag, float(row[key]), offset + step)
            writer.add_scalar("train/round", round_index, offset + step)
        offset += last_step
        exported += 1
    return exported


def main() -> None:
    args = parse_args()
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as err:
        raise SystemExit(f"tensorboard is not installed ({err}); run: uv add tensorboard")

    out_root = args.out if args.out is not None else args.run_dir / "tb"
    arms_dir = args.run_dir / "arms"
    if not arms_dir.is_dir():
        raise SystemExit(f"no arms/ under {args.run_dir}")

    for arm_dir in sorted(p for p in arms_dir.iterdir() if p.is_dir()):
        writer = SummaryWriter(log_dir=str(out_root / arm_dir.name))
        rounds = export_arm(writer, arm_dir)
        writer.close()
        print(f"{arm_dir.name}: {rounds} round(s) exported -> {out_root / arm_dir.name}")


if __name__ == "__main__":
    main()
