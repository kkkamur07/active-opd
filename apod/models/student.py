"""Qwen3.5-2B student: thinking-mode generation of K trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def encode_prompt(tokenizer, prompt: str, *, enable_thinking: bool = True):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer(text, return_tensors="pt")


def generate_trajectories(
    model,
    tokenizer,
    prompt: str,
    n: int = 16,
    max_new_tokens: int = 8192,
) -> dict[str, Any]:
    """Sample ``n`` completions. Token ids are a packed ``int32`` array ``[n, seq]``."""

    inputs = encode_prompt(tokenizer, prompt)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[-1])

    with torch.no_grad():
        sequences = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_ids = np.ascontiguousarray(sequences.cpu().numpy(), dtype=np.int32)
    response_ids = input_ids[:, prompt_length:]

    # Rows are right-padded with eos; the first eos marks the end of a response.
    # A row with no eos at all ran into max_new_tokens, i.e. was truncated.
    hits = response_ids == tokenizer.eos_token_id
    has_eos = hits.any(axis=1)
    lengths = np.where(has_eos, np.argmax(hits, axis=1) + 1, response_ids.shape[1])

    return {
        "prompt": prompt,
        "prompt_length": np.int32(prompt_length),
        "input_ids": input_ids,
        "response_lengths": lengths.astype(np.int32),
        "truncated": ~has_eos,
        "responses": np.asarray(
            tokenizer.batch_decode(response_ids, skip_special_tokens=True), dtype=object
        ),
    }
