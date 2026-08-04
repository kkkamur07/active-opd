"""Shared contracts for lazily loaded causal-language-model wrappers.

The wrappers in this module deliberately separate configuration from loading.
Constructing a wrapper is therefore safe in config resolution, tests, and
documentation builds; network access only happens when ``load`` is called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Any, Literal

type ModelInput = str | Sequence[Mapping[str, Any]]
type DTypeName = Literal["auto", "bfloat16", "float16", "float32"]
type QuantizationName = Literal["none", "4bit", "8bit"]
type DeviceMap = str | Mapping[str, int | str] | None


class ModelNotLoadedError(RuntimeError):
    """Raised when an operation requires a model or tokenizer that is absent."""


@dataclass(frozen=True)
class ModelLoadOptions:
    """Configuration shared by the teacher and student loaders.

    ``quantization`` is intentionally represented as a small string union so
    it maps directly to Hydra/YAML values. BitsAndBytes is imported only when
    a quantized model is actually loaded.
    """

    model_id: str
    revision: str | None = None
    dtype: DTypeName = "bfloat16"
    quantization: QuantizationName = "none"
    device_map: DeviceMap = None
    max_memory: Mapping[str | int, str] | None = None
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = False
    attn_implementation: str | None = "sdpa"
    gradient_checkpointing: bool = False
    optimizer_8bit: bool = False
    use_cache: bool = False
    bnb_4bit_compute_dtype: DTypeName | None = None
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> ModelLoadOptions:
        """Create options from a Hydra/OmegaConf-compatible mapping."""

        names = {field.name for field in fields(cls)}
        values = {name: config[name] for name in names if name in config}
        if "max_memory" in values and values["max_memory"] is not None:
            values["max_memory"] = {
                int(key.removeprefix("cuda:"))
                if isinstance(key, str)
                and key.removeprefix("cuda:").isdigit()
                else key: value
                for key, value in dict(values["max_memory"]).items()
            }
        return cls(**values)

    def pretrained_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for ``AutoModelForCausalLM.from_pretrained``."""

        kwargs: dict[str, Any] = {
            "torch_dtype": resolve_torch_dtype(self.dtype),
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.max_memory is not None:
            kwargs["max_memory"] = dict(self.max_memory)
        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation
        quantization_config = self.quantization_config()
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        return kwargs

    def quantization_config(self) -> Any | None:
        """Build a BitsAndBytes config, importing optional dependencies lazily."""

        if self.quantization == "none":
            return None

        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Quantized loading requires transformers with BitsAndBytesConfig "
                "support and the optional 'quantization' dependency."
            ) from exc

        if self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)

        compute_dtype = self.bnb_4bit_compute_dtype or self.dtype
        if compute_dtype == "auto":
            compute_dtype = "float16"
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolve_torch_dtype(compute_dtype),
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )


def resolve_torch_dtype(dtype: DTypeName | Any) -> Any:
    """Resolve a config dtype name without importing torch at module import time."""

    if dtype == "auto" or dtype is None:
        return "auto"
    if not isinstance(dtype, str):
        return dtype
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - model loading needs torch
        raise ImportError("PyTorch is required to resolve model dtypes.") from exc
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported dtype {dtype!r}; use auto, bfloat16, float16, or float32."
        ) from exc


@dataclass(frozen=True)
class GenerationOptions:
    """Generation controls shared by rollout collection and evaluation."""

    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    enable_thinking: bool = True
    eos_token_id: int | None = None
    pad_token_id: int | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> GenerationOptions:
        names = {field.name for field in fields(cls)}
        return cls(**{name: config[name] for name in names if name in config})

    def to_generate_kwargs(self, contract: TokenizerContract) -> dict[str, Any]:
        """Translate options into ``generate`` kwargs with safe token defaults."""

        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_return_sequences": self.num_return_sequences,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.pad_token_id
            if self.pad_token_id is not None
            else contract.pad_token_id,
        }
        if self.eos_token_id is not None:
            kwargs["eos_token_id"] = self.eos_token_id
        elif contract.eos_token_id is not None:
            kwargs["eos_token_id"] = contract.eos_token_id
        if self.do_sample:
            kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )
        return kwargs


@dataclass
class TokenizerContract:
    """Small tokenizer adapter shared by teacher and student wrappers."""

    tokenizer: Any

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> TokenizerContract:
        """Load a tokenizer explicitly; this is the first network boundary."""

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("Transformers is required to load a tokenizer.") from exc

        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision is not None:
            kwargs["revision"] = revision
        return cls.from_tokenizer(AutoTokenizer.from_pretrained(model_id, **kwargs))

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> TokenizerContract:
        """Wrap an existing tokenizer and ensure batching has a pad token."""

        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
        return cls(tokenizer=tokenizer)

    @property
    def pad_token_id(self) -> int | None:
        return getattr(self.tokenizer, "pad_token_id", None)

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self.tokenizer, "eos_token_id", None)

    def assert_compatible_with(self, other: TokenizerContract) -> None:
        """Raise when two tokenizers cannot be used for shared token IDs."""

        checks = (
            (
                "vocab_size",
                getattr(self.tokenizer, "vocab_size", None),
                getattr(other.tokenizer, "vocab_size", None),
            ),
            ("pad_token_id", self.pad_token_id, other.pad_token_id),
            ("eos_token_id", self.eos_token_id, other.eos_token_id),
        )
        mismatches = {
            name: (left, right)
            for name, left, right in checks
            if left is not None and right is not None and left != right
        }
        get_vocab = getattr(self.tokenizer, "get_vocab", None)
        other_get_vocab = getattr(other.tokenizer, "get_vocab", None)
        if (
            callable(get_vocab)
            and callable(other_get_vocab)
            and get_vocab() != other_get_vocab()
        ):
            mismatches["vocabulary"] = ("different", "different")
        if mismatches:
            details = ", ".join(
                f"{name}={left!r}/{right!r}"
                for name, (left, right) in mismatches.items()
            )
            raise ValueError(f"Tokenizer contracts are incompatible: {details}.")

    def format_prompt(
        self,
        prompt_or_messages: ModelInput,
        *,
        enable_thinking: bool = True,
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply the Qwen chat template while retaining a plain-text fallback."""

        if isinstance(prompt_or_messages, str):
            messages: Sequence[Mapping[str, Any]] = [
                {"role": "user", "content": prompt_or_messages}
            ]
        else:
            messages = prompt_or_messages

        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            if isinstance(prompt_or_messages, str):
                return prompt_or_messages
            return "\n".join(str(message.get("content", "")) for message in messages)

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        try:
            return apply_chat_template(messages, **kwargs)
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            kwargs.pop("enable_thinking")
            return apply_chat_template(messages, **kwargs)

    def encode_prompt(
        self,
        prompt_or_messages: ModelInput,
        *,
        enable_thinking: bool = True,
        add_generation_prompt: bool = True,
        return_tensors: str = "pt",
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> Mapping[str, Any]:
        """Format and tokenize one prompt for model generation."""

        text = self.format_prompt(
            prompt_or_messages,
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
        )
        kwargs: dict[str, Any] = {
            "return_tensors": return_tensors,
            "padding": padding,
            "truncation": truncation,
        }
        if max_length is not None:
            kwargs["max_length"] = max_length
        return self.tokenizer(text, **kwargs)

    def encode_batch(
        self,
        prompts: Sequence[ModelInput],
        *,
        enable_thinking: bool = True,
        add_generation_prompt: bool = True,
        return_tensors: str = "pt",
    ) -> Mapping[str, Any]:
        """Tokenize several prompts into one left-padded batch.

        Decoder-only generation requires left padding: with right padding the
        pads sit between the prompt and the first generated token and the model
        continues from a pad. Left padding keeps every row's prompt flush
        against the generation boundary, so one absolute response-start index
        is valid for the whole batch.
        """

        texts = [
            self.format_prompt(
                prompt,
                enable_thinking=enable_thinking,
                add_generation_prompt=add_generation_prompt,
            )
            for prompt in prompts
        ]
        previous_side = getattr(self.tokenizer, "padding_side", None)
        try:
            if previous_side is not None:
                self.tokenizer.padding_side = "left"
            return self.tokenizer(
                texts,
                return_tensors=return_tensors,
                padding=True,
                truncation=False,
            )
        finally:
            if previous_side is not None:
                self.tokenizer.padding_side = previous_side

    def decode(self, token_ids: Any, *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )


class CausalLMWrapper:
    """Lazy, role-neutral interface around a Transformers causal LM."""

    is_trainable: bool = False

    def __init__(
        self,
        options: ModelLoadOptions,
        *,
        tokenizer: TokenizerContract | Any | None = None,
    ) -> None:
        self.options = options
        self._tokenizer = (
            tokenizer
            if isinstance(tokenizer, TokenizerContract)
            else TokenizerContract.from_tokenizer(tokenizer)
            if tokenizer is not None
            else None
        )
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            raise ModelNotLoadedError(
                f"{type(self).__name__}.load() must be called before using the model."
            )
        return self._model

    @property
    def tokenizer(self) -> TokenizerContract:
        if self._tokenizer is None:
            raise ModelNotLoadedError(
                f"{type(self).__name__}.load() must be called before using the tokenizer."
            )
        return self._tokenizer

    def load_tokenizer(self) -> TokenizerContract:
        """Load only the tokenizer, useful for prompt formatting and compatibility."""

        if self._tokenizer is None:
            self._tokenizer = TokenizerContract.from_pretrained(
                self.options.model_id,
                revision=self.options.revision,
                trust_remote_code=self.options.trust_remote_code,
            )
        return self._tokenizer

    def load(self) -> CausalLMWrapper:
        """Load tokenizer and model; construction itself never downloads anything."""

        if self._model is not None:
            return self

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("Transformers is required to load a causal LM.") from exc

        self.load_tokenizer()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.options.model_id,
            **self.options.pretrained_kwargs(),
        )
        if self.is_trainable and self.options.device_map is None:
            try:
                import torch
            except ImportError:  # pragma: no cover - model loading already needs torch
                torch = None
            if torch is not None and torch.cuda.is_available():
                self._model.to("cuda")
        self._configure_loaded_model()
        return self

    def _configure_loaded_model(self) -> None:
        if self.options.gradient_checkpointing:
            enable_checkpointing = getattr(
                self.model,
                "gradient_checkpointing_enable",
                None,
            )
            if enable_checkpointing is not None:
                enable_checkpointing()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = self.options.use_cache

    def prepare_inputs(
        self,
        prompt_or_inputs: ModelInput | Mapping[str, Any],
        *,
        generation: GenerationOptions | None = None,
    ) -> Mapping[str, Any]:
        """Return tokenized inputs, preserving already-tokenized batches."""

        if isinstance(prompt_or_inputs, Mapping):
            return prompt_or_inputs
        generation = generation or GenerationOptions()
        return self.tokenizer.encode_prompt(
            prompt_or_inputs,
            enable_thinking=generation.enable_thinking,
        )

    def assert_tokenizer_compatible(self, other: CausalLMWrapper) -> None:
        """Validate that two wrappers can share sampled token IDs."""

        self.tokenizer.assert_compatible_with(other.tokenizer)

    def generate(
        self,
        prompt_or_inputs: ModelInput | Mapping[str, Any],
        *,
        generation: GenerationOptions | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> Any:
        """Generate token IDs from text/messages or an existing tokenized batch."""

        generation_options = self._generation_options(generation, overrides)
        inputs = self.prepare_inputs(prompt_or_inputs, generation=generation_options)
        inputs = self._move_inputs_to_model_device(inputs)
        was_training = bool(getattr(self.model, "training", False))
        config = getattr(self.model, "config", None)
        previous_use_cache = getattr(config, "use_cache", None)
        if was_training:
            self.model.eval()
        if config is not None and previous_use_cache is False:
            config.use_cache = True
        try:
            with self._inference_context():
                return self.model.generate(
                    **dict(inputs),
                    **generation_options.to_generate_kwargs(self.tokenizer),
                )
        finally:
            if config is not None and previous_use_cache is not None:
                config.use_cache = previous_use_cache
            if was_training:
                self.model.train()

    def generate_text(
        self,
        prompt_or_inputs: ModelInput | Mapping[str, Any],
        *,
        generation: GenerationOptions | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> list[str]:
        """Generate and decode one or more sequences."""

        sequences = self.generate(
            prompt_or_inputs,
            generation=generation,
            **overrides,
        )
        if getattr(sequences, "ndim", 1) == 1:
            sequences = sequences.unsqueeze(0)
        return [self.tokenizer.decode(sequence) for sequence in sequences]

    def forward(self, **inputs: Any) -> Any:
        """Forward a tokenized batch for later trainer/loss integration."""

        with self._inference_context() if not self.is_trainable else nullcontext():
            return self.model(**inputs)

    def unload(self) -> None:
        """Release references and clear CUDA cache when available."""

        self._model = None
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generation_options(
        self,
        generation: GenerationOptions | Mapping[str, Any] | None,
        overrides: Mapping[str, Any],
    ) -> GenerationOptions:
        if generation is None:
            options = GenerationOptions()
        elif isinstance(generation, GenerationOptions):
            options = generation
        else:
            options = GenerationOptions.from_mapping(generation)
        if not overrides:
            return options
        values = {field.name: getattr(options, field.name) for field in fields(options)}
        unknown = set(overrides).difference(values)
        if unknown:
            raise TypeError(f"Unknown generation option(s): {sorted(unknown)}")
        return GenerationOptions(**{**values, **overrides})

    def _move_inputs_to_model_device(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        device = getattr(self.model, "device", None)
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except (AttributeError, StopIteration):
                return inputs
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _inference_context() -> Any:
        try:
            import torch
        except ImportError:
            return nullcontext()
        return torch.inference_mode()
