"""vLLM offline engine: sample n traces per prompt, packed like the HF path.

vLLM is imported lazily inside the functions so that importing ``apod.models``
still works on a box without vLLM (or without a GPU at all).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata as metadata
import os
import sysconfig
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
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


def _ensure_cuda_home() -> str | None:
    """Point FlashInfer's JIT at the CUDA toolkit shipped in the wheels.

    vLLM samples ``top_k`` through FlashInfer, which compiles its sampling
    kernel at engine start and locates nvcc via ``CUDA_HOME``/``CUDA_PATH``,
    falling back to ``/usr/local/cuda``. This box installs the driver only, on
    purpose (see docs/guide.md), so that path does not exist and engine init
    dies with "Could not find nvcc and default cuda_home='/usr/local/cuda'
    doesn't exist" -- but only once a request actually uses top_k, which is the
    recommended Qwen setting, so it fails mid-run rather than at import.

    nvcc is nonetheless present: the ``nvidia-cuda-nvcc`` wheel ships a complete
    ``bin``/``include``/``nvvm`` tree inside site-packages. Point at that rather
    than making the caller export a variable, and leave an existing CUDA_HOME
    alone so a real toolkit still wins.
    """

    for name in ("CUDA_HOME", "CUDA_PATH"):
        existing = os.environ.get(name)
        if existing and (Path(existing) / "bin" / "nvcc").exists():
            return existing

    site = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    # cu13 today; glob so a future cu14 wheel set needs no edit here.
    for candidate in sorted(site.glob("cu*"), reverse=True):
        if (candidate / "bin" / "nvcc").exists():
            os.environ.setdefault("CUDA_HOME", str(candidate))
            os.environ.setdefault("CUDA_PATH", str(candidate))
            return str(candidate)

    # Nothing found: let vLLM raise its own error rather than guessing.
    return None


def _minor(version: str) -> tuple[int, int]:
    parts = version.split(".")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


def _disable_flashinfer_sampler_if_toolkit_mismatched() -> str | None:
    """Fall back to torch-native top-k/top-p when the JIT toolkit is mixed.

    vLLM samples top_k/top_p through FlashInfer, which compiles its kernel at
    engine start. That build includes CCCL, and CCCL hard-errors with "CUDA
    compiler and CUDA toolkit headers are incompatible" when nvcc's version does
    not match the CUDA headers next to it. The cu130 wheel set is internally
    mixed: nvidia-cuda-nvcc is 13.3.x while nvidia-cuda-runtime (which supplies
    the headers) is 13.0.x, so the build fails on this machine.

    A mixed 13.x set is fine at *runtime* -- minor version compatibility covers
    it -- but not for compiling against, which is the distinction that matters
    here. The native sampler applies the same top-k/top-p filter followed by a
    multinomial draw, so nothing about the sampling distribution changes; only
    the kernel does. Disabling it is preferable to pinning nvcc back to 13.0,
    which would perturb the whole locked dependency set for a sampling kernel.

    Respects an explicit VLLM_USE_FLASHINFER_SAMPLER from the caller.
    """

    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ:
        return None

    try:
        nvcc = metadata.version("nvidia-cuda-nvcc")
        runtime = metadata.version("nvidia-cuda-runtime")
    except metadata.PackageNotFoundError:
        return None

    if _minor(nvcc) == _minor(runtime):
        return None

    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    reason = f"nvcc {nvcc} vs cuda-runtime {runtime}"
    print(f"[vllm] mixed CUDA toolkit ({reason}); using the native top-k/top-p sampler")
    return reason


def _config_cache_key(model_id: str) -> tuple[str, Path] | None:
    """(config-content hash, vLLM cache root) for a local checkpoint, else None.

    Both the served-model alias and the compile cache below are keyed by
    config.json content rather than checkpoint path; hub model ids already
    have a stable path and return None.
    """

    config_path = Path(model_id) / "config.json"
    if not config_path.is_file():
        return None
    key = hashlib.sha256(config_path.read_bytes()).hexdigest()[:10]
    cache_root = Path(
        os.environ.get("VLLM_CACHE_ROOT", str(Path.home() / ".cache" / "vllm"))
    )
    return key, cache_root


def _stable_model_alias(model_id: str) -> str:
    """Serve local checkpoints through a config-keyed symlink.

    vLLM's AOT-compile cache key hashes ``ModelConfig.model`` -- the path
    string -- so every round's new checkpoint dir re-pays ~125 s of AOT
    compile + warmup even when the pinned compile dir (below) is warm
    (measured: 137.6 s init on a warm pinned cache vs 14.0 s fully warm).
    Compiled artifacts depend on architecture and shapes, never weights, so
    give vLLM a path that is STABLE across rounds: a symlink named by the
    config.json content hash, atomically re-pointed at the current
    checkpoint. Hub model ids pass through untouched.
    """

    keyed = _config_cache_key(model_id)
    if keyed is None:
        return model_id
    key, cache_root = keyed
    alias = cache_root / "apod_model_alias" / key
    alias.parent.mkdir(parents=True, exist_ok=True)
    tmp = alias.parent / f"{key}.tmp.{os.getpid()}"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(Path(model_id).resolve())
    os.replace(tmp, alias)  # atomic re-point; concurrent shards write the same target
    return str(alias)


def _stable_compile_cache_dir(model_id: str) -> str | None:
    """A compile-cache dir keyed by config content, not checkpoint path.

    vLLM's default torch.compile cache key includes the model *path*
    (ModelConfig.compute_hash hashes ``model``), so every round's checkpoint
    directory is a cache miss and each engine re-pays ~130s of compile+warmup
    (measured: 153-156s init vs 13-14s warm). The computation graph only
    depends on the config, which is byte-identical across our round
    checkpoints -- so for local checkpoints, pin ``compilation_config.cache_dir``
    to a directory named by the config.json hash. Any real config change gets
    a fresh dir. Hub models keep vLLM's default (their path IS stable).
    """

    keyed = _config_cache_key(model_id)
    if keyed is None:
        return None
    key, cache_root = keyed
    return str(cache_root / "apod_compile_cache" / key)


def build_llm(
    model_id: str,
    *,
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.90,
    enable_prefix_caching: bool = True,
    language_model_only: bool = True,
    max_cudagraph_capture_size: int | None = None,
    seed: int | None = None,
    fast_presence_penalty: bool = True,
    **extra: Any,
):
    """One engine per process, one GPU per process. Never tensor-parallel here.

    Two 2B replicas (one per card, ``CUDA_VISIBLE_DEVICES=0`` / ``=1``) beat one
    TP=2 replica: the model is 4 GB, so the split buys nothing and costs
    all-reduces on every layer.
    """

    from vllm import LLM

    from apod.models.vllm_qwen35 import register

    # Our round checkpoints declare Qwen3_5ForCausalLM, which this vLLM build
    # neither routes nor loads correctly; see apod/models/vllm_qwen35.py.
    register()
    _ensure_cuda_home()
    _disable_flashinfer_sampler_if_toolkit_mismatched()

    kwargs: dict[str, Any] = {
        # vLLM's LLM class silences engine stats by default. They are the only
        # live view of how many sequences are actually in flight, which is what
        # tells apart "the GPU is saturated" from "we under-submitted and the
        # chunk is draining". tqdm only advances when a whole request finishes,
        # so it cannot show that.
        "disable_log_stats": False,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": enable_prefix_caching,
        "language_model_only": language_model_only,
        **extra,
    }
    if max_cudagraph_capture_size is not None:
        kwargs["max_cudagraph_capture_size"] = max_cudagraph_capture_size
    compile_cache_dir = _stable_compile_cache_dir(model_id)
    if compile_cache_dir is not None:
        kwargs["compilation_config"] = {"cache_dir": compile_cache_dir}
        print(f"[vllm] pinned compile cache: {compile_cache_dir}")
    checkpoint_path = Path(model_id).resolve() if Path(model_id).is_dir() else None
    model_id = _stable_model_alias(model_id)
    if seed is not None:
        kwargs["seed"] = seed
    if fast_presence_penalty:
        from apod.models.presence_penalty import IncrementalPresencePenalty

        kwargs["logits_processors"] = [IncrementalPresencePenalty]

    kwargs, dropped = _filter_engine_kwargs(kwargs)
    if dropped:
        print(f"[vllm] installed build does not accept {dropped}; continuing without them")
    if fast_presence_penalty and "logits_processors" in dropped:
        # build_sampling_params routes the penalty through extra_args and leaves
        # the native field at 0.0. Without the processor that penalty would be
        # silently dropped and every trace would be generated unpenalised, which
        # is a wrong experiment rather than a slow one. Fail instead.
        raise RuntimeError(
            "this vLLM build does not accept logits_processors; "
            "pass fast_presence_penalty=False to use the native penalty path"
        )

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        **kwargs,
    )
    if checkpoint_path is not None:
        # The alias is one mutable pointer shared by every checkpoint load. The
        # pipeline is strictly sequential, but if that ever breaks, an engine
        # would silently serve another round's weights -- so hard-fail instead.
        served = Path(model_id).resolve()
        if served != checkpoint_path:
            raise RuntimeError(
                f"model alias was re-pointed during engine construction: "
                f"expected {checkpoint_path}, now serving {served}"
            )
    return llm


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
    max_tokens: int | None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    seed: int | None = None,
    fast_presence_penalty: bool = True,
):
    """Qwen thinking-mode defaults.

    NOTE (ADR 0004): the PIPELINE runs presence_penalty 0.0 -- generation,
    entropy scoring, and the GKD objective must share one distribution. The
    1.5 default here reaches only the standalone measurement scripts, which
    historically wanted the penalty on. The pre-ADR observation stands as
    context: penalty 0.0 once produced a 61% truncation rate on the base 2B
    at exploratory settings (though smoke2 measured 0/96 repetition loops at
    temperature 1.0 / top_p 0.95 / top_k 20 with penalty off).

    ``max_tokens=None`` means no generation cap: a trace runs until the model
    emits EOS, or until the prompt plus response reaches the engine's
    ``max_model_len``. That is the honest setting for measuring how long traces
    actually are, because any finite cap manufactures truncations that are an
    artifact of the cap rather than of the model.
    """

    from vllm import SamplingParams

    from apod.models.presence_penalty import EXTRA_ARGS_KEY

    extra_args = None
    if fast_presence_penalty and presence_penalty:
        # Hand the penalty to IncrementalPresencePenalty and leave the native
        # field at 0.0, which is what lets vLLM skip its O(batch x tokens)
        # per-step penalty rebuild. Requires build_llm(fast_presence_penalty=True).
        extra_args = {EXTRA_ARGS_KEY: presence_penalty}
        presence_penalty = 0.0

    return SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        seed=seed,
        extra_args=extra_args,
    )


def _pack(request_output, prompt: str, pad_id: int) -> dict[str, Any]:
    """One RequestOutput (n completions, shared prompt) -> the save_npz dict."""

    prompt_ids = list(request_output.prompt_token_ids)
    prompt_length = len(prompt_ids)
    completions = list(request_output.outputs)

    lengths = np.asarray([len(c.token_ids) for c in completions], dtype=np.int32)
    width = prompt_length + int(lengths.max(initial=0))

    input_ids = np.full((len(completions), width), pad_id, dtype=np.int32)
    input_ids[:, :prompt_length] = prompt_ids
    for row, completion in enumerate(completions):
        input_ids[row, prompt_length : prompt_length + len(completion.token_ids)] = completion.token_ids

    return {
        "prompt": prompt,
        "prompt_length": np.int32(prompt_length),
        "input_ids": input_ids,
        "response_lengths": lengths,
        # finish_reason == "length" means the sampler ran out of room: either
        # max_tokens if one was set, or max_model_len when it was not.
        "truncated": np.asarray([c.finish_reason == "length" for c in completions], dtype=bool),
        "finish_reasons": np.asarray([str(c.finish_reason) for c in completions], dtype=object),
        "responses": np.asarray([c.text for c in completions], dtype=object),
    }


def _count_chunk_tokens(outputs) -> dict[str, int]:
    """Exact token counts for one ``llm.generate`` call, read off the id lists.

    ``prompt_token_ids`` and ``CompletionOutput.token_ids`` are the real ids the
    engine processed, so nothing here is a character-count estimate. The prompt
    is counted once per request rather than once per completion: with ``n > 1``
    vLLM prefills the prompt a single time and forks the KV cache for the n
    samples, so charging it n times would inflate the prefill number n-fold.
    """

    counts = {"prompt_tokens": 0, "cached_prompt_tokens": 0, "generated_tokens": 0, "sequences": 0}
    for request_output in outputs:
        counts["prompt_tokens"] += len(request_output.prompt_token_ids or ())
        # Prefix-cache hits are prompt tokens that were never recomputed. Older
        # builds do not set the field, hence the getattr.
        counts["cached_prompt_tokens"] += int(getattr(request_output, "num_cached_tokens", None) or 0)
        for completion in request_output.outputs:
            counts["generated_tokens"] += len(completion.token_ids)
            counts["sequences"] += 1
    return counts


@dataclasses.dataclass
class _ThroughputMeter:
    """Running totals across every chunk of one generation run."""

    chunks: int = 0
    prompts: int = 0
    seconds: float = 0.0
    counts: Counter = dataclasses.field(default_factory=Counter)

    def add(self, *, prompts: int, seconds: float, counts: dict[str, int]) -> None:
        self.chunks += 1
        self.prompts += prompts
        self.seconds += seconds
        self.counts.update(counts)

    def totals(self) -> dict[str, Any]:
        return _rates(
            seconds=self.seconds,
            counts=self.counts,
            extra={"chunks": self.chunks, "prompts": self.prompts},
        )


def _rates(
    *,
    seconds: float,
    counts: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Counts plus the per-second rates they imply over ``seconds``.

    Read the rates as *amortised over the whole window*, not as isolated engine
    phases. vLLM's continuous batching interleaves the prefill of one request
    with the decode of another inside a single ``generate`` call, so
    ``prompt_tokens_per_s`` is "prompt tokens pushed through per second of wall
    clock", not the rate a dedicated prefill pass would hit; the same window is
    simultaneously being charged for decode. ``generated_tokens_per_s`` is the
    number to quote for decode throughput, and ``total_tokens_per_s`` is the
    only one that sums cleanly.
    """

    span = max(float(seconds), 1e-9)
    prompt_tokens = counts["prompt_tokens"]
    generated_tokens = counts["generated_tokens"]
    sequences = counts["sequences"]
    total_tokens = prompt_tokens + generated_tokens
    metrics: dict[str, Any] = {
        "seconds": float(seconds),
        "prompt_tokens": int(prompt_tokens),
        "cached_prompt_tokens": int(counts["cached_prompt_tokens"]),
        "generated_tokens": int(generated_tokens),
        "total_tokens": int(total_tokens),
        "sequences": int(sequences),
        "prompt_tokens_per_s": prompt_tokens / span,
        "generated_tokens_per_s": generated_tokens / span,
        "total_tokens_per_s": total_tokens / span,
        "sequences_per_s": sequences / span,
        "mean_generated_tokens_per_sequence": generated_tokens / sequences if sequences else 0.0,
    }
    if extra:
        metrics.update(extra)
    return metrics


def generate_trajectories_vllm(
    llm,
    tokenizer,
    prompts: Sequence[str],
    *,
    n: int = 16,
    max_tokens: int | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    chunk_size: int = 8,
    seed: int | None = None,
    enable_thinking: bool = True,
    on_chunk: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(position, batch)`` in prompt order, one batch per prompt.

    Prompts are submitted ``chunk_size`` at a time so a dying run keeps the GPU
    hours it already spent, while still giving the scheduler ``chunk_size * n``
    concurrent sequences to fill the batch with. One ``generate`` call per chunk:
    the n completions share a single prefill.

    Throughput is measured per chunk and reported two ways, both additive to the
    existing ``(position, batch)`` contract so older callers keep working:

    * ``batch["throughput"]`` -- the chunk's metrics dict, with a ``cumulative``
      entry covering the run so far. Every batch from one chunk carries the same
      dict object, because the timing is a property of the ``generate`` call, not
      of the individual prompt. ``save_npz`` only writes the keys it knows, so
      the extra key never reaches disk as an array.
    * ``on_chunk(metrics)`` -- called once per chunk, right after its
      ``generate`` returns and before the chunk's batches are yielded, for
      progress printing.

    The clock covers only ``llm.generate``; prompt rendering and tokenization
    happen once up front and are outside it.
    """

    # Not ``pad_token_id or eos_token_id``: token id 0 is falsy but a real id.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    texts = [render_prompt(tokenizer, prompt, enable_thinking=enable_thinking) for prompt in prompts]
    meter = _ThroughputMeter()

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
        began = time.perf_counter()
        outputs = llm.generate(texts[start:stop], params)
        seconds = time.perf_counter() - began

        counts = _count_chunk_tokens(outputs)
        meter.add(prompts=stop - start, seconds=seconds, counts=counts)
        metrics = _rates(
            seconds=seconds,
            counts=counts,
            extra={
                "chunk_index": meter.chunks - 1,
                "chunk_start": start,
                "chunk_prompts": stop - start,
            },
        )
        metrics["cumulative"] = meter.totals()
        if on_chunk is not None:
            on_chunk(metrics)

        for offset, request_output in enumerate(outputs):
            batch = _pack(request_output, prompts[start + offset], pad_id)
            batch["throughput"] = metrics
            yield start + offset, batch
