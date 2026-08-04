"""Frozen teacher model wrapper for Qwen3 distillation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import (
    CausalLMWrapper,
    ModelLoadOptions,
    TokenizerContract,
)

DEFAULT_TEACHER_MODEL_ID = "Qwen/Qwen3-4B"


class TeacherModel(CausalLMWrapper):
    """Lazy, frozen teacher with optional 4-bit BitsAndBytes loading."""

    is_trainable = False

    def __init__(
        self,
        options: ModelLoadOptions | Mapping[str, Any] | None = None,
        *,
        tokenizer: TokenizerContract | Any | None = None,
    ) -> None:
        super().__init__(self._resolve_options(options), tokenizer=tokenizer)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        tokenizer: TokenizerContract | Any | None = None,
    ) -> TeacherModel:
        """Construct a teacher from a Hydra ``model.teacher`` mapping."""

        return cls(ModelLoadOptions.from_mapping(config), tokenizer=tokenizer)

    @staticmethod
    def _resolve_options(
        options: ModelLoadOptions | Mapping[str, Any] | None,
    ) -> ModelLoadOptions:
        if options is None:
            return ModelLoadOptions(
                model_id=DEFAULT_TEACHER_MODEL_ID,
                dtype="bfloat16",
                quantization="4bit",
                device_map="auto",
                max_memory={"cuda:0": "22GiB", "cpu": "32GiB"},
                gradient_checkpointing=False,
                optimizer_8bit=False,
                use_cache=True,
            )
        if isinstance(options, ModelLoadOptions):
            return options
        return ModelLoadOptions.from_mapping(options)

    def _configure_loaded_model(self) -> None:
        super()._configure_loaded_model()
        self.model.eval()
        self.model.requires_grad_(False)
