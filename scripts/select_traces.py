"""Score collected traces by student entropy, then keep the top-k per prompt.

Two stages, because the forward pass wants both GPUs and the ranking wants
neither:

  --stage entropy   HF student forward over this shard's traces -> entropy.shard{k}.jsonl
  --stage rank      read every entropy shard, apply the policies -> selected/*.jsonl
  --stage all       both, single process

    H(tau) = mean_t  Entropy( pi_S( . | x, y_<t) )   over response positions t

The vLLM engine must be gone before this runs: it holds 90% of the card for KV
cache and the HF model will not fit next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from apod.datasets import append_jsonl, read_jsonl, read_shards, write_jsonl
from apod.models import STUDENT_ID, load_student

POLICIES = ("h_correct", "h_incorrect", "random")


def _decoder_and_head(model):
    """Split the LM into body and lm_head so logits never materialise in full.

    The vocab is ~248k: a 16k-token float32 logit tensor is ~16 GB. Running the
    body once and applying the head in time-slices keeps the peak at one slice.
    """

    body = getattr(model, "model", None)
    if body is None:
        body = getattr(model, "base_model", None)
    head = model.get_output_embeddings()
    # ``base_model`` returns the model itself when no inner body exists, which
    # would recurse straight back into the head.
    if body is None or body is model or head is None:
        raise RuntimeError(
            f"{type(model).__name__} does not expose .model/.lm_head; cannot compute "
            "entropy without materialising the full logit tensor."
        )
    return body, head


@torch.no_grad()
def trace_scores(model, body, head, ids: np.ndarray, prompt_length: int, response_length: int, chunk: int) -> dict:
    """Mean entropy and mean taken-token logprob over the response positions.

    Logits at position ``t-1`` predict token ``t``, so response token ``i``
    (absolute index ``prompt_length + i``) is scored at position
    ``prompt_length + i - 1``.
    """

    total = prompt_length + response_length
    device = next(model.parameters()).device
    tokens = torch.as_tensor(np.asarray(ids[:total], dtype=np.int64), device=device)[None]

    hidden = body(input_ids=tokens, use_cache=False)
    hidden = hidden[0] if isinstance(hidden, tuple) else hidden.last_hidden_state
    hidden = hidden[0]

    start, stop = prompt_length - 1, total - 1
    entropy_sum = 0.0
    logprob_sum = 0.0
    for a in range(start, stop, chunk):
        b = min(a + chunk, stop)
        log_probs = torch.log_softmax(head(hidden[a:b]).float(), dim=-1)
        entropy_sum += float(-(log_probs.exp() * log_probs).sum(-1).sum())
        targets = tokens[0, a + 1 : b + 1, None]
        logprob_sum += float(log_probs.gather(-1, targets).sum())

    count = stop - start
    return {
        "entropy": entropy_sum / count,
        "mean_logprob": logprob_sum / count,
        "scored_tokens": count,
    }


def stage_entropy(args, out: Path) -> None:
    rows = read_shards(out, args.pattern)
    if not rows:
        raise SystemExit(f"No trajectories matched {out / args.pattern}")

    path = out / f"entropy.shard{args.shard}.jsonl"
    done = {(r["example_index"], r["rollout_index"]) for r in read_jsonl(path)} if args.resume else set()
    if not args.resume:
        path.unlink(missing_ok=True)

    pending = [
        row
        for row in rows
        if row["example_index"] % args.num_shards == args.shard
        and (row["example_index"], row["rollout_index"]) not in done
        and row["response_length"] > 0
    ]
    print(f"shard {args.shard}/{args.num_shards}: scoring {len(pending)} traces ({len(done)} already done)")
    if not pending:
        return

    _, model = load_student(args.student_id)
    model.eval()
    body, head = _decoder_and_head(model)

    cache: tuple[int, dict] | None = None
    for row in tqdm(sorted(pending, key=lambda r: (r["example_index"], r["rollout_index"])), desc="traces"):
        example_index = row["example_index"]
        if cache is None or cache[0] != example_index:
            npz_path = out / row.get("npz", f"student/{example_index:04d}.npz")
            cache = (example_index, dict(np.load(npz_path, allow_pickle=True)))
        batch = cache[1]

        scores = trace_scores(
            model, body, head,
            batch["input_ids"][row["rollout_index"]],
            int(batch["prompt_length"]),
            int(batch["response_lengths"][row["rollout_index"]]),
            args.logit_chunk,
        )
        append_jsonl(path, {"example_index": example_index, "rollout_index": row["rollout_index"], **scores})

    print(f"wrote {path}")


def rank_example(traces: list[dict], k: int, rng: np.random.Generator) -> dict[str, list[dict]]:
    """Three policies over one prompt's traces. Ties break on the lower index."""

    by_entropy = sorted(traces, key=lambda t: (-t["entropy"], t["rollout_index"]))
    picks = {
        "h_correct": [t for t in by_entropy if t["correct"]][:k],
        "h_incorrect": [t for t in by_entropy if not t["correct"]][:k],
    }
    order = rng.permutation(len(traces))[:k]
    picks["random"] = [traces[i] for i in sorted(order)]
    return picks


def stage_rank(args, out: Path) -> None:
    trajectories = read_shards(out, args.pattern)
    scores = {(r["example_index"], r["rollout_index"]): r for r in read_shards(out, "entropy.shard*.jsonl")}
    if not scores:
        raise SystemExit(f"No entropy shards in {out}; run --stage entropy first")

    dropped_truncated = 0
    dropped_unscored = 0
    by_example: dict[int, list[dict]] = defaultdict(list)
    for row in trajectories:
        if row["truncated"] and not args.include_truncated:
            dropped_truncated += 1
            continue
        score = scores.get((row["example_index"], row["rollout_index"]))
        if score is None:
            dropped_unscored += 1
            continue
        by_example[row["example_index"]].append({**row, **score})

    print(
        f"{len(trajectories)} traces -> {sum(len(v) for v in by_example.values())} eligible "
        f"({dropped_truncated} truncated, {dropped_unscored} unscored)"
    )

    selected: dict[str, list[dict]] = {policy: [] for policy in POLICIES}
    for example_index in sorted(by_example):
        traces = sorted(by_example[example_index], key=lambda t: t["rollout_index"])
        picks = rank_example(traces, args.k, np.random.default_rng(args.seed + example_index))
        for policy, chosen in picks.items():
            for rank, trace in enumerate(chosen):
                selected[policy].append(
                    {
                        "policy": policy,
                        "rank": rank,
                        "example_index": trace["example_index"],
                        "rollout_index": trace["rollout_index"],
                        "id": trace["id"],
                        "prompt": trace["prompt"],
                        "answer": trace["answer"],
                        "correct": trace["correct"],
                        "truncated": trace["truncated"],
                        "response_length": trace["response_length"],
                        "entropy": trace["entropy"],
                        "mean_logprob": trace["mean_logprob"],
                        "npz": trace.get("npz", f"student/{trace['example_index']:04d}.npz"),
                    }
                )

    directory = out / "selected"
    summary = {"k": args.k, "eligible_traces": sum(len(v) for v in by_example.values()), "policies": {}}
    for policy, chosen in selected.items():
        write_jsonl(directory / f"{policy}.jsonl", chosen)
        entropies = np.asarray([t["entropy"] for t in chosen], dtype=np.float64)
        summary["policies"][policy] = {
            "traces": len(chosen),
            "prompts": len({t["example_index"] for t in chosen}),
            "accuracy": float(np.mean([t["correct"] for t in chosen])) if chosen else None,
            "mean_entropy": float(entropies.mean()) if entropies.size else None,
            "mean_response_length": float(np.mean([t["response_length"] for t in chosen])) if chosen else None,
        }
        print(f"{policy:14} {len(chosen):5} traces over {summary['policies'][policy]['prompts']} prompts")

    # Does entropy actually separate correct from incorrect? If these two means
    # coincide, an entropy prefilter is noise and the experiment says so here.
    everything = [t for traces in by_example.values() for t in traces]
    for label, subset in (("correct", [t for t in everything if t["correct"]]),
                          ("incorrect", [t for t in everything if not t["correct"]])):
        values = np.asarray([t["entropy"] for t in subset], dtype=np.float64)
        summary[f"entropy_{label}"] = {
            "n": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
        }
        print(f"entropy | {label:9} n={values.size:5} mean={values.mean() if values.size else float('nan'):.4f}")

    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {directory}")


def parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["entropy", "rank", "all"], default="entropy")
    parser.add_argument("--trajectories-dir", default="outputs/trajectories")
    parser.add_argument("--pattern", default="trajectories.shard*.jsonl")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--student-id", default=STUDENT_ID)
    parser.add_argument("--logit-chunk", type=int, default=512, help="response positions per lm_head slice")
    parser.add_argument("--include-truncated", action="store_true",
                        help="keep traces that hit the token cap (they have no final answer)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.num_shards:
        parser.error(f"--shard must be in [0, {args.num_shards}); got {args.shard}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.trajectories_dir)
    if args.stage in ("entropy", "all"):
        stage_entropy(args, out)
    if args.stage in ("rank", "all"):
        stage_rank(args, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
