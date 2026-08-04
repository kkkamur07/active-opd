"""On-policy student rollout collection.

Two properties matter here and are enforced rather than assumed:

* **Truncation is recorded at generation time.** A rollout that used its whole
  ``max_new_tokens`` budget without emitting EOS did not finish reasoning, and
  the verifier needs to know that from the generator rather than guessing from
  the decoded text (a trace can close ``</think>`` and still be cut off before
  writing its answer).
* **Padding never reaches the trainer.** ``generate`` right-pads the ``K``
  returned sequences to a common length; those pads were previously stored
  verbatim, counted as generated tokens, and forwarded through both models.
  Each rollout now stores its own trimmed ids plus a real attention mask.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aopd.data.datasets import MathExample
from aopd.data.rollouts import Rollout
from aopd.models.common import GenerationOptions


@dataclass(frozen=True)
class RolloutCollectionConfig:
    """Generation controls for one rollout round."""

    num_rollouts_per_prompt: int = 8
    max_new_tokens: int = 1024
    retain_token_ids: bool = True
    #: Prompts generated per ``generate`` call. Batched decoding is a large
    #: throughput win (generation dominates wall clock) and requires left
    #: padding, which the response mask now handles via absolute start indices.
    prompt_batch_size: int = 1

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> RolloutCollectionConfig:
        names = {"num_rollouts_per_prompt", "max_new_tokens", "retain_token_ids", "prompt_batch_size"}
        values = {name: config[name] for name in names if name in config}
        if "retain_prompt_tokens" in config and "retain_token_ids" not in values:
            values["retain_token_ids"] = config["retain_prompt_tokens"]
        return cls(**values)


def _as_sequence_batch(sequences: Any) -> list[Any]:
    if isinstance(sequences, (str, bytes)):
        return [sequences]
    if hasattr(sequences, "ndim") and sequences.ndim == 1:
        return [sequences]
    if isinstance(sequences, Sequence):
        return list(sequences)
    if hasattr(sequences, "__len__") and hasattr(sequences, "__getitem__"):
        return [sequences[index] for index in range(len(sequences))]
    return [sequences]


def _trim_response(
    sequence: Any,
    prompt_length: int,
    *,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> tuple[Any, int, bool]:
    """Return ``(trimmed_ids, response_length, hit_eos)`` for one sequence.

    Trailing pad tokens added to align the ``K`` returned sequences are
    removed. EOS is kept: it is the token the student most needs to learn.
    """

    try:
        ids = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
    except TypeError:  # pragma: no cover - non-sequence generation output
        return sequence, 0, False

    response = ids[prompt_length:]
    hit_eos = False
    if eos_token_id is not None and eos_token_id in response:
        end = response.index(eos_token_id) + 1
        response = response[:end]
        hit_eos = True
    elif pad_token_id is not None:
        while response and response[-1] == pad_token_id:
            response.pop()
    return ids[:prompt_length] + response, len(response), hit_eos


class RolloutCollector:
    """Collect K independent student responses for each prompt."""

    def __init__(
        self,
        student: Any,
        config: RolloutCollectionConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.student = student
        if isinstance(config, RolloutCollectionConfig):
            self.config = config
        else:
            self.config = RolloutCollectionConfig.from_mapping(config or {})
        if self.config.num_rollouts_per_prompt <= 0:
            raise ValueError("num_rollouts_per_prompt must be positive.")
        if self.config.prompt_batch_size <= 0:
            raise ValueError("prompt_batch_size must be positive.")

    def collect(
        self,
        prompts: Iterable[str | MathExample],
        *,
        generation: GenerationOptions | Mapping[str, Any] | None = None,
        references: Sequence[str | None] | None = None,
        round_index: int = 0,
    ) -> list[Rollout]:
        """Generate and decode rollouts without materializing a dataset."""

        generation_options = self._generation_options(generation)
        examples = list(prompts)
        if references is not None and len(references) != len(examples):
            raise ValueError("references must align one-to-one with prompts.")

        pad_token_id, eos_token_id = self._special_token_ids(generation_options)
        records: list[Rollout] = []
        batch_size = self.config.prompt_batch_size

        for start in range(0, len(examples), batch_size):
            window = examples[start : start + batch_size]
            texts = [
                item.prompt if isinstance(item, MathExample) else str(item)
                for item in window
            ]
            prompt_lengths, sequences = self._generate_window(texts, generation_options)
            per_prompt = self.config.num_rollouts_per_prompt
            flat = _as_sequence_batch(sequences)
            if len(flat) != len(window) * per_prompt:
                raise RuntimeError(
                    f"generate returned {len(flat)} sequences for {len(window)} prompts "
                    f"x {per_prompt} rollouts."
                )
            for offset, item in enumerate(window):
                example_index = start + offset
                reference = (
                    item.reference_answer
                    if isinstance(item, MathExample)
                    else references[example_index]
                    if references is not None
                    else None
                )
                prompt_length = prompt_lengths[offset]
                for rollout_index in range(per_prompt):
                    sequence = flat[offset * per_prompt + rollout_index]
                    records.append(
                        self._build_rollout(
                            sequence=sequence,
                            prompt=texts[offset],
                            reference=reference,
                            prompt_length=prompt_length,
                            example_index=example_index,
                            rollout_index=rollout_index,
                            round_index=round_index,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            max_new_tokens=generation_options.max_new_tokens,
                            problem_id=getattr(item, "problem_id", None),
                        )
                    )
        return records

    def _build_rollout(
        self,
        *,
        sequence: Any,
        prompt: str,
        reference: str | None,
        prompt_length: int,
        example_index: int,
        rollout_index: int,
        round_index: int,
        pad_token_id: int | None,
        eos_token_id: int | None,
        max_new_tokens: int,
        problem_id: str | None,
    ) -> Rollout:
        if isinstance(sequence, str):
            return Rollout(
                prompt=prompt,
                response=sequence,
                reference_answer=reference,
                rollout_id=f"r{round_index}:{example_index}:{rollout_index}",
                prompt_length=prompt_length,
                metadata={
                    "prompt_index": example_index,
                    "rollout_index": rollout_index,
                    "round_index": round_index,
                    "problem_id": problem_id,
                    "num_rollouts_per_prompt": self.config.num_rollouts_per_prompt,
                },
            )

        trimmed, response_length, hit_eos = _trim_response(
            sequence,
            prompt_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        truncated = (not hit_eos) and response_length >= max_new_tokens
        response = self._decode(trimmed, prompt_length)
        return Rollout(
            prompt=prompt,
            response=response,
            reference_answer=reference,
            rollout_id=f"r{round_index}:{example_index}:{rollout_index}",
            prompt_length=prompt_length,
            input_ids=trimmed if self.config.retain_token_ids else None,
            attention_mask=[1] * len(trimmed) if self.config.retain_token_ids else None,
            truncated=truncated,
            response_length=response_length,
            metadata={
                "prompt_index": example_index,
                "rollout_index": rollout_index,
                "round_index": round_index,
                "problem_id": problem_id,
                "hit_eos": hit_eos,
                "num_rollouts_per_prompt": self.config.num_rollouts_per_prompt,
            },
        )

    def _generate_window(
        self,
        texts: Sequence[str],
        generation: GenerationOptions,
    ) -> tuple[list[int], Any]:
        """Generate for a window of prompts, returning per-prompt prompt lengths.

        Batched generation pads on the left (an HF requirement for
        decoder-only models), so every row shares one prompt end offset.
        """

        if len(texts) == 1:
            prompt_length = _prompt_length(self.student, texts[0], generation)
            sequences = self.student.generate(texts[0], generation=generation)
            return [prompt_length], sequences

        inputs = self._encode_batch(texts, generation)
        if inputs is None:
            # Wrapper does not support batched encoding; fall back per prompt.
            lengths: list[int] = []
            collected: list[Any] = []
            for text in texts:
                lengths.append(_prompt_length(self.student, text, generation))
                collected.extend(_as_sequence_batch(
                    self.student.generate(text, generation=generation)
                ))
            return lengths, collected
        padded_length = int(inputs["input_ids"].shape[-1])
        sequences = self.student.generate(inputs, generation=generation)
        return [padded_length] * len(texts), sequences

    def _encode_batch(
        self,
        texts: Sequence[str],
        generation: GenerationOptions,
    ) -> Mapping[str, Any] | None:
        tokenizer = getattr(self.student, "tokenizer", None)
        encode = getattr(tokenizer, "encode_batch", None)
        if not callable(encode):
            return None
        try:
            return encode(texts, enable_thinking=generation.enable_thinking)
        except (AttributeError, TypeError, NotImplementedError):  # pragma: no cover
            return None

    def _decode(self, sequence: Any, prompt_length: int) -> str:
        tokenizer = getattr(self.student, "tokenizer", None)
        if tokenizer is None:
            return str(sequence)
        decode = getattr(tokenizer, "decode", None)
        if decode is None:
            return str(sequence)
        return str(decode(sequence[prompt_length:]))

    def _special_token_ids(
        self,
        generation: GenerationOptions,
    ) -> tuple[int | None, int | None]:
        tokenizer = getattr(self.student, "tokenizer", None)
        pad = generation.pad_token_id
        eos = generation.eos_token_id
        if tokenizer is not None:
            pad = pad if pad is not None else getattr(tokenizer, "pad_token_id", None)
            eos = eos if eos is not None else getattr(tokenizer, "eos_token_id", None)
        return pad, eos

    def _generation_options(
        self,
        generation: GenerationOptions | Mapping[str, Any] | None,
    ) -> GenerationOptions:
        if generation is None:
            options = GenerationOptions(
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=self.config.num_rollouts_per_prompt,
            )
        elif isinstance(generation, GenerationOptions):
            options = generation
        else:
            options = GenerationOptions.from_mapping(generation)
        conflicts = []
        if options.max_new_tokens != self.config.max_new_tokens:
            conflicts.append(
                f"max_new_tokens ({options.max_new_tokens} vs {self.config.max_new_tokens})"
            )
        if options.num_return_sequences != self.config.num_rollouts_per_prompt:
            conflicts.append(
                "num_return_sequences "
                f"({options.num_return_sequences} vs {self.config.num_rollouts_per_prompt})"
            )
        if conflicts:
            raise ValueError(
                "Generation options conflict with the rollout config: "
                + "; ".join(conflicts)
                + ". Set them consistently rather than relying on a silent override."
            )
        return options


def _shape_last(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape:
        return int(shape[-1])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return None


def _prompt_length(student: Any, prompt: str, generation: GenerationOptions) -> int:
    prepare_inputs = getattr(student, "prepare_inputs", None)
    if not callable(prepare_inputs):
        return 0
    inputs = prepare_inputs(prompt, generation=generation)
    attention = inputs.get("attention_mask") if hasattr(inputs, "get") else None
    if attention is not None and hasattr(attention, "sum"):
        try:
            return int(attention.sum().item())
        except (AttributeError, TypeError, ValueError):
            pass
    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    return _shape_last(attention) or _shape_last(input_ids) or 0


def collect_rollouts(
    student: Any,
    prompts: Iterable[str | MathExample],
    *,
    config: RolloutCollectionConfig | Mapping[str, Any] | None = None,
    generation: GenerationOptions | Mapping[str, Any] | None = None,
    references: Sequence[str | None] | None = None,
    round_index: int = 0,
) -> list[Rollout]:
    """Functional convenience wrapper around ``RolloutCollector``."""

    return RolloutCollector(student, config).collect(
        prompts,
        generation=generation,
        references=references,
        round_index=round_index,
    )


__all__ = [
    "RolloutCollectionConfig",
    "RolloutCollector",
    "collect_rollouts",
]
