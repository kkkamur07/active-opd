"""vLLM offline engine: sample n traces per prompt, packed like the HF path.

vLLM is imported lazily inside the functions so that importing ``apod.models``
still works on a box without vLLM (or without a GPU at all).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np


def _filter_engine_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop kwargs this vLLM build does not know about.

    ``language_model_only`` and ``max_cudagraph_capture_size`` come and go
    between releases; an unknown kwarg is a hard TypeError from ``LLM.__init__``,
    which is a bad way to lose a job that is otherwise fine without them.
    """

    from vllm import EngineArgs

    known = {field.name for field in dataclasses.fields(EngineArgs)}
    kept = {key: value for key, value in kwargs.items() if key in known}
    return kept, sorted(set(kwargs) - set(kept))


def build_llm(
    model_id: str,
    *,
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.90,
    enable_prefix_caching: bool = True,
    language_model_only: bool = True,
    max_cudagraph_capture_size: int | None = None,
    seed: int | None = None,
    **extra: Any,
):
    """One engine per process, one GPU per process. Never tensor-parallel here.

    Two 2B replicas (one per card, ``CUDA_VISIBLE_DEVICES=0`` / ``=1``) beat one
    TP=2 replica: the model is 4 GB, so the split buys nothing and costs
    all-reduces on every layer.
    """

    from vllm import LLM

    kwargs: dict[str, Any] = {
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": enable_prefix_caching,
        "language_model_only": language_model_only,
        **extra,
    }
    if max_cudagraph_capture_size is not None:
        kwargs["max_cudagraph_capture_size"] = max_cudagraph_capture_size
    if seed is not None:
        kwargs["seed"] = seed

    kwargs, dropped = _filter_engine_kwargs(kwargs)
    if dropped:
        print(f"[vllm] installed build does not accept {dropped}; continuing without them")

    return LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        **kwargs,
    )


def render_prompt(tokenizer, prompt: str, *, enable_thinking: bool = True) -> str:
    """Same chat template as the HF path, left as text for vLLM to tokenize."""

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_sampling_params(
    *,
    n: int,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    seed: int | None = None,
):
    """Qwen thinking-mode defaults. ``presence_penalty`` is what stops the loop.

    With penalty 0.0 the 2B model repeats itself until it hits the token cap;
    that is where the 61% truncation rate came from, not the cap being too low.
    """

    from vllm import SamplingParams

    return SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        seed=seed,
    )


def _pack(request_output, prompt: str, pad_id: int) -> dict[str, Any]:
    """One RequestOutput (n completions, shared prompt) -> the save_npz dict."""

    prompt_ids = list(request_output.prompt_token_ids)
    prompt_length = len(prompt_ids)
    completions = list(request_output.outputs)

    lengths = np.asarray([len(c.token_ids) for c in completions], dtype=np.int32)
    width = prompt_length + int(lengths.max()) if lengths.size else prompt_length

    input_ids = np.full((len(completions), width), pad_id, dtype=np.int32)
    for row, completion in enumerate(completions):
        input_ids[row, :prompt_length] = prompt_ids
        input_ids[row, prompt_length : prompt_length + len(completion.token_ids)] = completion.token_ids

    return {
        "prompt": prompt,
        "prompt_length": np.int32(prompt_length),
        "input_ids": input_ids,
        "response_lengths": lengths,
        # finish_reason == "length" means the sampler ran into max_tokens.
        "truncated": np.asarray([c.finish_reason == "length" for c in completions], dtype=bool),
        "finish_reasons": np.asarray([str(c.finish_reason) for c in completions], dtype=object),
        "responses": np.asarray([c.text for c in completions], dtype=object),
    }


def generate_trajectories_vllm(
    llm,
    tokenizer,
    prompts: Sequence[str],
    *,
    n: int = 16,
    max_tokens: int = 16384,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    chunk_size: int = 8,
    seed: int | None = None,
    enable_thinking: bool = True,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(position, batch)`` in prompt order, one batch per prompt.

    Prompts are submitted ``chunk_size`` at a time so a dying run keeps the GPU
    hours it already spent, while still giving the scheduler ``chunk_size * n``
    concurrent sequences to fill the batch with. One ``generate`` call per chunk:
    the n completions share a single prefill.
    """

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    texts = [render_prompt(tokenizer, prompt, enable_thinking=enable_thinking) for prompt in prompts]

    for start in range(0, len(texts), chunk_size):
        stop = min(start + chunk_size, len(texts))
        params = build_sampling_params(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            # Per-chunk seed keeps a resumed run reproducible without making
            # every chunk sample the identical stream.
            seed=None if seed is None else seed + start,
        )
        outputs = llm.generate(texts[start:stop], params)
        for offset, request_output in enumerate(outputs):
            yield start + offset, _pack(request_output, prompts[start + offset], pad_id)
