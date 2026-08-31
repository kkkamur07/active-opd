"""Entropy-score one round's trajectories under the round's starting model.

Pipeline stage (docs/pipeline.md): the driver launches one process per GPU
with CUDA_VISIBLE_DEVICES set; work is sharded by
``example_index % num_shards == shard``. For each trajectory this computes

    H(tau) = mean_t  Entropy( pi_S( . | x, y_<t) )   over response positions t

with a full-vocab softmax under the HF student, plus the mean taken-token
logprob. Every trajectory is scored, truncated ones included (ADR 0002).
The vLLM engine must be torn down before this runs: it holds 90% of the card
for KV cache and the HF model will not fit next to it.

Reads  <round>/rollouts/tokens/example_XXXXX.npz
Writes <round>/entropy/entropy.shard{K}.jsonl  (+ done.shard{K} marker)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from apod import paths
from apod.datasets import append_jsonl, read_jsonl, write_jsonl
from apod.models import load_lm
from apod.stages.common import parse_stage_args, stage_parser


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

    # Anything narrower than longTensor gets cast away. 
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


def shard_examples(tokens_dir: Path, shard: int, num_shards: int) -> list[tuple[int, Path]]:
    examples: list[tuple[int, Path]] = []
    for path in sorted(tokens_dir.glob("example_*.npz")):
        example_index = int(path.stem.split("_")[1])
        if example_index % num_shards == shard:
            examples.append((example_index, path))
    return examples


def parse_args(argv: list[str] | None):
    return parse_stage_args(stage_parser(description=__doc__), argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")

    round_dir = paths.round_dir(run_dir, args.arm, args.round_index)
    tokens_dir = round_dir / "rollouts" / "tokens"
    entropy_dir = round_dir / "entropy"
    out_path = entropy_dir / f"entropy.shard{args.shard}.jsonl"
    marker = entropy_dir / f"done.shard{args.shard}"

    # resume=false must redo THIS stage too: the rollout stage regenerates its
    # trajectories under the same flag, and stale entropy rows would join the
    # fresh trajectories silently (keys match, values do not).
    resume = bool(cfg.resume)
    if not resume:
        marker.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
    if resume and marker.exists():
        logger.info("{} exists; shard already complete", marker)
        return 0

    examples = shard_examples(tokens_dir, args.shard, args.num_shards)
    if not examples:
        raise SystemExit(f"No example_*.npz for shard {args.shard}/{args.num_shards} in {tokens_dir}")

    scored_rows = read_jsonl(out_path, drop_torn_tail=True)
    if scored_rows:
        # Atomic rewrite so a torn final line never precedes new appends.
        write_jsonl(out_path, scored_rows)
    done = {(r["example_index"], r["rollout_index"]) for r in scored_rows}
    logger.info(
        "arm={} round={} shard={}/{}: {} examples, {} trajectories already scored",
        args.arm, args.round_index, args.shard, args.num_shards, len(examples), len(done),
    )

    # Same resolution (and weights-present check) as rollout_eval and train.
    from apod.stages.rollout_eval import resolve_model_path

    model_path = resolve_model_path(run_dir, args.arm, args.round_index, cfg)
    # Entropy must be the GENERATING policy's uncertainty: scoring under any
    # other weights silently corrupts the selection signal. Assert we resolve
    # to the exact model the rollout stage recorded, and leave our own record.
    rollout_meta_path = round_dir / "rollouts" / "model_path.json"
    if rollout_meta_path.exists():
        rollout_model = json.loads(rollout_meta_path.read_text())["model_path"]
        if rollout_model != model_path:
            raise RuntimeError(
                f"entropy would score under {model_path} but the rollouts were "
                f"generated by {rollout_model} (rollouts/model_path.json)"
            )
    entropy_dir.mkdir(parents=True, exist_ok=True)
    with open(entropy_dir / "meta.json", "w") as f:
        json.dump({"model_path": model_path}, f)
    logger.info("loading starting model {} (bf16, frozen)", model_path)
    _, model = load_lm(model_path, frozen=True)
    body, head = _decoder_and_head(model)
    chunk = int(cfg.selection.logit_chunk)

    started = time.perf_counter()
    scored_tokens = 0
    scored_trajectories = 0
    for example_index, npz_path in tqdm(examples, desc=f"shard{args.shard}"):
        # Only the arrays this stage scores: the object-dtype response text
        # stays unread on disk.
        with np.load(npz_path, allow_pickle=True) as batch:
            input_ids = batch["input_ids"]
            prompt_length = int(batch["prompt_length"])
            response_lengths = batch["response_lengths"]
        for rollout_index in range(input_ids.shape[0]):
            if (example_index, rollout_index) in done:
                continue
            response_length = int(response_lengths[rollout_index])
            if response_length <= 0:
                logger.warning(
                    "example {} rollout {} has empty response; skipping",
                    example_index, rollout_index,
                )
                continue
            scores = trace_scores(
                model, body, head,
                input_ids[rollout_index],
                prompt_length,
                response_length,
                chunk,
            )
            append_jsonl(out_path, {
                "example_index": example_index,
                "rollout_index": rollout_index,
                **scores,
            })
            scored_tokens += scores["scored_tokens"]
            scored_trajectories += 1

    elapsed = time.perf_counter() - started
    marker.touch()
    logger.info(
        "scored {} trajectories / {} tokens in {:.1f}s ({:.0f} tok/s) -> {}",
        scored_trajectories, scored_tokens, elapsed,
        scored_tokens / elapsed if elapsed > 0 else 0.0, out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
