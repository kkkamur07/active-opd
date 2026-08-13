"""Frozen Qwen3.5-9B teacher: one generation per prompt, logits as numpy."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .student import generate_trajectories


def teacher_logits(model, input_ids: np.ndarray) -> np.ndarray:
    """No-grad forward. Returns float16 logits ``[n, seq, vocab]``."""

    device = next(model.parameters()).device
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits

    return np.ascontiguousarray(logits.to(torch.float16).cpu().numpy())


def generate_teacher(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 8192,
) -> dict[str, Any]:
    """One completion plus teacher logits on that sequence."""

    batch = generate_trajectories(model, tokenizer, prompt, n=1, max_new_tokens=max_new_tokens)
    length = int(batch["prompt_length"]) + int(batch["response_lengths"][0])
    batch["logits"] = teacher_logits(model, batch["input_ids"][:, :length])
    return batch
