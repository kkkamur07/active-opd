"""Student model wrapper for full-parameter Qwen3 training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import (
    CausalLMWrapper,
    ModelLoadOptions,
    TokenizerContract,
)

DEFAULT_STUDENT_MODEL_ID = "Qwen/Qwen3-1.7B"


class StudentModel(CausalLMWrapper):
    """Lazy Qwen3 student with memory-safe full-parameter training defaults.

    The first experiment trains the student weights directly, so student
    quantization is rejected. Memory is controlled through BF16, activation
    checkpointing, gradient accumulation in the later trainer, and optionally
    an 8-bit optimizer.
    """

    is_trainable = True

    def __init__(
        self,
        options: ModelLoadOptions | Mapping[str, Any] | None = None,
        *,
        tokenizer: TokenizerContract | Any | None = None,
    ) -> None:
        resolved = self._resolve_options(options)
        if resolved.quantization != "none":
            raise ValueError(
                "StudentModel uses full-parameter training and must use "
                "quantization='none'. Use quantization on the frozen teacher."
            )
        super().__init__(resolved, tokenizer=tokenizer)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        tokenizer: TokenizerContract | Any | None = None,
    ) -> StudentModel:
        """Construct a student from a Hydra ``model.student`` mapping."""

        return cls(ModelLoadOptions.from_mapping(config), tokenizer=tokenizer)

    @staticmethod
    def _resolve_options(
        options: ModelLoadOptions | Mapping[str, Any] | None,
    ) -> ModelLoadOptions:
        if options is None:
            return ModelLoadOptions(
                model_id=DEFAULT_STUDENT_MODEL_ID,
                dtype="bfloat16",
                quantization="none",
                device_map=None,
                gradient_checkpointing=True,
                optimizer_8bit=True,
                use_cache=False,
            )
        if isinstance(options, ModelLoadOptions):
            return options
        return ModelLoadOptions.from_mapping(options)

    def prepare_for_training(self) -> StudentModel:
        """Enable training mode and checkpointing after an explicit ``load``."""

        self.load()
        self.model.train()
        if self.options.gradient_checkpointing:
            enable_input_grads = getattr(
                self.model,
                "enable_input_require_grads",
                None,
            )
            if enable_input_grads is not None:
                enable_input_grads()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        return self
