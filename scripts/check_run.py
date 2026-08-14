"""Sanity-check a run directory's stage outputs (smoke or real).

Walks every completed stage (identified by its done markers) under
outputs/runs/<name> and asserts the artifacts are real, well-formed, and
mutually consistent — rollouts non-degenerate with agreeing tokens/text,
entropies finite and varying, selection traceable, training loss finite and
moving, eval metrics plausible. Prints one line per check and exits non-zero
with a clear message on the first violation.

    python scripts/check_run.py --run-dir outputs/runs/smoke
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from apod.datasets.io import read_jsonl, read_shards


def fail(msg: str) -> None:
    raise SystemExit(f"CHECK FAILED: {msg}")


def ok(msg: str) -> None:
    print(f"  ok: {msg}")


def check_rollouts(rdir: Path, cfg, tokenizer) -> None:
    rows = read_shards(rdir / "rollouts", "trajectories.shard*.jsonl")
    num_prompts = int(cfg.rollout.num_prompts)
    num_rollouts = int(cfg.rollout.num_rollouts)
    expected = num_prompts * num_rollouts
    if len(rows) != expected:
        fail(f"{rdir}: {len(rows)} trajectory rows, expected {expected}")

    by_example: dict[int, list[dict]] = {}
    for r in rows:
        by_example.setdefault(r["example_index"], []).append(r)
    for example_index, group in by_example.items():
        if len(group) != num_rollouts:
            fail(f"{rdir}: example {example_index} has {len(group)} rollouts, expected {num_rollouts}")
        if len({r["response"] for r in group}) < 2:
            fail(f"{rdir}: example {example_index} rollouts are all identical (seeds not varying?)")

    empty = [r for r in rows if r["response_length"] <= 0 or not r["response"].strip()]
    if empty:
        fail(f"{rdir}: {len(empty)} empty responses, e.g. example {empty[0]['example_index']}")
    truncated = sum(r["truncated"] for r in rows)
    if truncated == len(rows):
        # Expected at smoke scale (1024 cap on a base 2B model); alarming at 8192.
        print(f"  WARNING: {rdir}: every trajectory hit the {cfg.sampling.max_new_tokens}-token cap")

    # Same terminator derivation as the pipeline (one implementation). The
    # generating model's config/generation_config survive retention pruning,
    # so this works on pruned rounds too.
    from apod.stages.rollout_eval import collect_eos_ids

    meta = rdir / "rollouts" / "model_path.json"
    generating_model = (
        json.loads(meta.read_text())["model_path"] if meta.exists() else str(cfg.model.student_id)
    )
    eos_ids = collect_eos_ids(tokenizer, generating_model)

    npz_paths = sorted((rdir / "rollouts" / "tokens").glob("example_*.npz"))
    if len(npz_paths) != num_prompts:
        fail(f"{rdir}: {len(npz_paths)} npz token files, expected {num_prompts}")
    for npz_path in npz_paths:
        with np.load(npz_path, allow_pickle=True) as data:
            ids = data["input_ids"]
            prompt_length = int(data["prompt_length"])
            lengths = data["response_lengths"]
            trunc = data["truncated"]
            responses = data["responses"]
        if ids.shape[0] != num_rollouts:
            fail(f"{npz_path}: {ids.shape[0]} rollouts in npz, expected {num_rollouts}")
        for j in range(num_rollouts):
            resp_ids = [int(t) for t in ids[j, prompt_length : prompt_length + int(lengths[j])]]
            if not bool(trunc[j]) and resp_ids[-1] not in eos_ids:
                fail(f"{npz_path} rollout {j}: finished 'stop' but last token {resp_ids[-1]} is not EOS")
        # tokens <-> text agreement on the first rollout of each example
        resp_ids = [int(t) for t in ids[0, prompt_length : prompt_length + int(lengths[0])]]
        if not bool(trunc[0]) and resp_ids and resp_ids[-1] in eos_ids:
            resp_ids = resp_ids[:-1]  # vLLM's .text excludes the terminating EOS
        decoded = tokenizer.decode(resp_ids)
        if decoded != str(responses[0]):
            fail(f"{npz_path} rollout 0: decoded tokens do not match stored response text")

    n_correct = sum(r["correct"] for r in rows)
    ok(
        f"rollouts: {len(rows)} trajectories, {truncated}/{len(rows)} cap-hit, "
        f"{n_correct}/{len(rows)} correct, tokens<->text agree, EOS present"
    )


def check_entropy(rdir: Path, cfg) -> None:
    rows = read_shards(rdir / "entropy", "entropy.shard*.jsonl")
    trajectories = read_shards(rdir / "rollouts", "trajectories.shard*.jsonl")
    scorable = {(r["example_index"], r["rollout_index"]) for r in trajectories if r["response_length"] > 0}
    scored = {(r["example_index"], r["rollout_index"]) for r in rows}
    if len(rows) != len(scored):
        fail(f"{rdir}: duplicate entropy rows ({len(rows)} rows, {len(scored)} unique)")
    if scored != scorable:
        fail(f"{rdir}: entropy covers {len(scored)} trajectories, expected {len(scorable)}")

    values = [r["entropy"] for r in rows]
    if not all(math.isfinite(v) for v in values):
        fail(f"{rdir}: non-finite entropy values")
    if min(values) < 0 or max(values) > 15:  # ln(248k vocab) ~= 12.4
        fail(f"{rdir}: entropy out of sane range [{min(values):.3f}, {max(values):.3f}]")
    if max(values) - min(values) < 1e-6:
        fail(f"{rdir}: entropy identical across all trajectories — scoring degenerate")
    ok(
        f"entropy: {len(rows)} scores, finite, range [{min(values):.3f}, {max(values):.3f}], "
        f"mean {sum(values) / len(values):.3f}"
    )


def check_selection(rdir: Path, cfg, arm: str) -> None:
    rows = read_jsonl(rdir / "selected" / "selected.jsonl")
    num_prompts = int(cfg.rollout.num_prompts)
    per_prompt = int(cfg.rollout.num_rollouts) if arm == "all" else int(cfg.selection.k)
    if len(rows) != num_prompts * per_prompt:
        fail(f"{rdir}: {len(rows)} selected rows, expected {num_prompts * per_prompt} for arm {arm}")
    keys = {"example_index", "rollout_index", "entropy", "correct", "truncated", "response_length"}
    for r in rows:
        if set(r) != keys:
            fail(f"{rdir}: selected row keys {sorted(r)} != contract {sorted(keys)}")
    real = {(t["example_index"], t["rollout_index"]) for t in read_shards(rdir / "rollouts", "trajectories.shard*.jsonl")}
    ghosts = [(r["example_index"], r["rollout_index"]) for r in rows if (r["example_index"], r["rollout_index"]) not in real]
    if ghosts:
        fail(f"{rdir}: selected rows not traceable to rollouts: {ghosts[:5]}")
    if arm == "entropy_top4" and any(r["entropy"] is None for r in rows):
        fail(f"{rdir}: entropy_top4 selection has rows without entropy")
    ok(f"selection: {len(rows)} rows ({per_prompt}/prompt), schema matches, all trace to real rollouts")


def check_train(rdir: Path, *, weights_expected: bool = True) -> None:
    summary = json.loads((rdir / "train" / "summary.json").read_text())
    for key in ("train_loss_mean", "train_loss_final"):
        if summary[key] is None or not math.isfinite(summary[key]):
            fail(f"{rdir}: {key} is {summary[key]}")
    history = read_jsonl(rdir / "train" / "log_history.jsonl")
    losses = [h["loss"] for h in history if "loss" in h]
    if len(losses) < 2:
        fail(f"{rdir}: only {len(losses)} logged optimizer steps — trainer barely ran")
    if len(set(losses)) == 1:
        fail(f"{rdir}: loss frozen at {losses[0]} across {len(losses)} steps")
    if not any("grad_norm" in h for h in history):
        fail(f"{rdir}: no grad_norm in log history — optimizer not stepping?")
    checkpoint = rdir / "checkpoint"
    json.loads((checkpoint / "config.json").read_text())
    has_weights = bool(list(checkpoint.glob("*.safetensors")))
    # The driver prunes a round's weights once its successor checkpoint
    # exists (disk cannot hold every round), so only the latest round of an
    # arm must still carry them.
    if weights_expected and not has_weights:
        fail(f"{checkpoint}: no safetensors weight files")
    ok(
        f"train: {len(losses)} steps, loss {losses[0]:.4f} -> {losses[-1]:.4f} "
        f"(mean {summary['train_loss_mean']:.4f}), {summary['tokens_trained']} tokens, "
        f"checkpoint {'reloadable' if has_weights else 'weights pruned (superseded)'}"
    )


def check_eval(rdir: Path, cfg) -> None:
    rows = read_shards(rdir / "eval", "eval.shard*.jsonl")
    summary = json.loads((rdir / "eval" / "summary.json").read_text())
    # Round 0 and the terminal eval-only round run the full set; intermediate
    # rounds run the prefix subset (runs predating the subset lack the config
    # key and default to full). Mirrors _eval_num_problems in apod.main.
    rnd = int(rdir.name.split("_")[1])
    full = int(cfg.eval.num_problems)
    if rnd == 0 or rnd >= int(cfg.rounds):
        n_problems = full
    else:
        n_problems = min(int(cfg.eval.get("intermediate_num_problems", full)), full)
    expected = n_problems * int(cfg.eval.num_samples)
    if len(rows) != expected:
        fail(f"{rdir}: {len(rows)} eval rows, expected {expected}")
    for key in ("avg_at_n", "pass_at_n", "cap_hit_rate"):
        if not 0.0 <= summary[key] <= 1.0:
            fail(f"{rdir}: eval {key}={summary[key]} outside [0, 1]")
    if summary["pass_at_n"] < summary["avg_at_n"]:
        fail(f"{rdir}: pass@n {summary['pass_at_n']} < avg@n {summary['avg_at_n']} — impossible")
    ok(
        f"eval: {len(rows)} rows, avg@{summary['num_samples']}={summary['avg_at_n']:.3f}, "
        f"pass@{summary['num_samples']}={summary['pass_at_n']:.3f}, "
        f"cap_hit={summary['cap_hit_rate']:.2f}"
    )


def check_metrics(run_dir: Path, cfg) -> None:
    rows = read_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        fail(f"{run_dir}/metrics.jsonl is empty")
    keys = {
        "arm", "round", "trajectories_round", "trajectories_cumulative", "tokens_trained",
        "avg_at_n", "pass_at_n", "strict_avg_at_n", "strict_pass_at_n", "eval_num_problems",
        "eval_cap_hit_rate", "rollout_cap_hit_rate", "rollout_accuracy",
        "mean_entropy_selected", "train_loss_mean", "train_loss_final", "wall_clock",
        "rollout_throughput_tok_s",
    }
    seen = set()
    for r in rows:
        missing = keys - set(r)
        if missing:
            fail(f"metrics.jsonl row {r.get('arm')}/{r.get('round')} missing keys {sorted(missing)}")
        if (r["arm"], r["round"]) in seen:
            fail(f"metrics.jsonl duplicate row for ({r['arm']}, {r['round']})")
        seen.add((r["arm"], r["round"]))
    ok(f"metrics.jsonl: {len(rows)} rows, all keys present, no duplicates")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.student_id))

    checked_any = False
    for arm_dir in sorted((run_dir / "arms").glob("*")):
        arm = arm_dir.name
        round_dirs = sorted(arm_dir.glob("rounds/round_*"))
        last_trained = max(
            (r for r in round_dirs if (r / "train" / "done.shard0").exists()),
            default=None,
        )
        for rdir in round_dirs:
            print(f"{arm}/{rdir.name}:")
            if list((rdir / "eval").glob("done.shard*")):
                check_eval(rdir, cfg)
                checked_any = True
            if list((rdir / "rollouts").glob("done.shard*")):
                check_rollouts(rdir, cfg, tokenizer)
                checked_any = True
            if list((rdir / "entropy").glob("done.shard*")):
                check_entropy(rdir, cfg)
            if (rdir / "selected" / "selected.jsonl").exists():
                check_selection(rdir, cfg, arm)
            if (rdir / "train" / "done.shard0").exists():
                check_train(rdir, weights_expected=rdir == last_trained)
    if (run_dir / "metrics.jsonl").exists():
        check_metrics(run_dir, cfg)
    if not checked_any:
        fail(f"no completed stages found under {run_dir}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
