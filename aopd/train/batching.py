"""Turn selected rollouts into padded training batches.

This is the link between selection and the trainer that previously existed
only as private, batch-size-1 copies inside individual scripts.

Batches are **right-padded**: for a forward/backward pass (unlike generation)
right padding is correct, because causal attention means trailing pads cannot
influence earlier positions, and it keeps each row's prompt boundary at its own
index.  The response mask is built from absolute start indices so the same code
is valid for left-padded batches too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aopd.data.rollouts import Rollout


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise ImportError("Batch building requires PyTorch.") from exc
    return torch


def _rollout_ids(rollout: Rollout) -> list[int]:
    if rollout.input_ids is None:
        raise ValueError(
            f"Rollout {rollout.rollout_id!r} has no token ids; collect with "
            "retain_token_ids=True before training on it."
        )
    ids = rollout.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return [int(token) for token in ids]


def build_token_batch(
    rollouts: Sequence[Rollout],
    *,
    pad_token_id: int,
    eos_token_id: int | None = None,
    device: Any | None = None,
    max_length: int | None = None,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build a right-padded, shifted causal-LM batch from selected rollouts.

    Returns the tensors the trainer consumes directly:

    ``model_input_ids`` / ``model_attention_mask``
        the sequence minus its final token, i.e. the model's input
    ``labels``
        the sequence minus its first token, aligned to the model's logits
    ``response_mask``
        boolean, True exactly at generated (non-prompt, non-pad) label
        positions
    """

    torch = _require_torch()
    if not rollouts:
        raise ValueError("Cannot build a batch from an empty rollout selection.")

    sequences = [_rollout_ids(rollout) for rollout in rollouts]
    prompt_lengths = []
    for rollout, ids in zip(rollouts, sequences):
        length = rollout.prompt_length
        if length is None:
            raise ValueError(
                f"Rollout {rollout.rollout_id!r} has no prompt_length; the response "
                "mask cannot be built without it."
            )
        if length >= len(ids):
            raise ValueError(
                f"Rollout {rollout.rollout_id!r} has prompt_length {length} but only "
                f"{len(ids)} tokens: it contains no response to train on."
            )
        prompt_lengths.append(int(length))

    if max_length is not None:
        trimmed: list[list[int]] = []
        for ids, prompt_length in zip(sequences, prompt_lengths):
            if len(ids) > max_length:
                if prompt_length >= max_length:
                    raise ValueError(
                        "max_length truncates a prompt entirely; raise max_length "
                        "or drop the example."
                    )
                ids = ids[:max_length]
            trimmed.append(ids)
        sequences = trimmed

    batch_size = len(sequences)
    width = max(len(ids) for ids in sequences)
    input_ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    for row, ids in enumerate(sequences):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1

    starts = torch.tensor(prompt_lengths, dtype=torch.long)
    model_input_ids = input_ids[:, :-1]
    model_attention_mask = attention_mask[:, :-1]
    labels = input_ids[:, 1:]
    label_attention = attention_mask[:, 1:]

    positions = torch.arange(labels.shape[1]).unsqueeze(0)
    # `starts - 1` because label j corresponds to input position j+1.
    response_mask = label_attention.bool() & (positions >= (starts - 1).unsqueeze(1))

    batch: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "model_input_ids": model_input_ids,
        "model_attention_mask": model_attention_mask,
        "labels": labels,
        "response_mask": response_mask,
        "prompt_lengths": starts,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "response_tokens": int(response_mask.sum().item()),
    }
    if weights is not None:
        if len(weights) != batch_size:
            raise ValueError("weights must align one-to-one with rollouts.")
        batch["weights"] = torch.tensor(list(weights), dtype=torch.float32)
    if device is not None:
        batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    return batch


def iter_micro_batches(
    rollouts: Sequence[Rollout],
    *,
    micro_batch_size: int,
    max_tokens: int | None = None,
) -> list[list[Rollout]]:
    """Split a selection into micro-batches, optionally capped by token count.

    Rollouts are sorted by length so each micro-batch pads to a similar width;
    with 18k-token thinking traces, mixing a 600-token and an 18k-token rollout
    in one batch wastes almost the entire forward pass on padding.
    """

    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive.")
    ordered = sorted(rollouts, key=lambda rollout: len(_rollout_ids(rollout)))
    batches: list[list[Rollout]] = []
    current: list[Rollout] = []
    current_width = 0
    for rollout in ordered:
        width = max(current_width, len(_rollout_ids(rollout)))
        would_exceed_tokens = (
            max_tokens is not None and current and width * (len(current) + 1) > max_tokens
        )
        if current and (len(current) >= micro_batch_size or would_exceed_tokens):
            batches.append(current)
            current, current_width = [], 0
            width = len(_rollout_ids(rollout))
        current.append(rollout)
        current_width = width
    if current:
        batches.append(current)
    return batches


def batch_builder(
    pad_token_id: int,
    *,
    eos_token_id: int | None = None,
    device: Any | None = None,
    max_length: int | None = None,
):
    """Return a ``batch_builder`` callable for ``OPDTrainer.fit_rollout_rounds``."""

    def build(selected: Sequence[Rollout]) -> Mapping[str, Any]:
        return build_token_batch(
            selected,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            device=device,
            max_length=max_length,
        )

    return build


__all__ = ["batch_builder", "build_token_batch", "iter_micro_batches"]
