"""vLLM adapter for checkpoints saved by transformers as Qwen3_5ForCausalLM.

Four gaps in vLLM 0.26.0 break reloading our own round checkpoints. Its
``Qwen3_5ForCausalLM`` exists only as the inner language model of the
multimodal class and was never wired up to run standalone:

1. The architecture string ``Qwen3_5ForCausalLM`` is not in vLLM's registry,
   so ``_normalize_arch`` suffix-swaps it to ``Qwen3_5ForConditionalGeneration``
   (the multimodal class), which crashes on a text-only config with
   ``'Qwen3_5TextConfig' object has no attribute 'vision_config'``.
2. transformers 5.x saves qwen3_5_text weights under ``model.language_model.*``
   (its registered checkpoint-conversion format, kept compatible with the
   multimodal repo layout), while vLLM's text class expects flat ``model.*``
   and its ``load_weights`` applies no rename.
3. The text class does not declare ``IsHybrid``, so the hybrid-model config
   hook that sizes the GDN/mamba state caches never runs and engine init dies
   on ``assert mamba_block_size is not None``. The multimodal class carries
   the interface and its state-shape classmethods, which read only
   ``hf_text_config`` -- exactly our config -- so we borrow them.
4. The text config carries interleaved M-RoPE parameters (``mrope_section``),
   so the runner requires ``SupportsMRoPE``; for text-only input the position
   grid is just the plain index repeated over T/H/W with delta 0.

``register()`` fixes all three: it maps the architecture string to this
subclass (the registry's exact-name match short-circuits the suffix swap) and
installs the same per-arch config hook the multimodal path uses (it copies
``mamba_ssm_dtype: float32`` from the HF config into the cache config, so
checkpoint rollouts keep the base model's SSM-state numerics). Base-repo
loads are unaffected -- they use the multimodal architecture string.
"""

from __future__ import annotations


def register() -> None:
    """Idempotently register Qwen3_5ForCausalLM with vLLM.

    Must run before engine construction. The string form keeps the import
    lazy, and registration propagates to the EngineCore workers because vLLM
    defaults to fork on this setup (no CUDA initialized beforehand).
    """

    from vllm import ModelRegistry
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        Qwen3_5ForConditionalGenerationConfig,
    )

    if "Qwen3_5ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Qwen3_5ForCausalLM", "apod.models.vllm_qwen35:Qwen3_5TextForCausalLM"
        )
    MODELS_CONFIG_MAP.setdefault(
        "Qwen3_5ForCausalLM", Qwen3_5ForConditionalGenerationConfig
    )


def __getattr__(name: str):
    # Defined via module __getattr__ so importing this module (for register())
    # never imports vLLM's model code in the parent process; only the worker
    # resolving the registry string pays for it.
    if name != "Qwen3_5TextForCausalLM":
        raise AttributeError(name)

    import torch
    from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )
    from vllm.model_executor.models.utils import WeightsMapper

    class Qwen3_5TextForCausalLM(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "model."}
        )

        # The multimodal class's mamba-state classmethods ignore ``cls`` and
        # read only hf_text_config, so the CG-bound methods work as-is here.
        get_mamba_state_dtype_from_config = (
            Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config
        )
        get_mamba_state_shape_from_config = (
            Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config
        )
        get_mamba_state_copy_func = (
            Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func
        )

        def load_weights(self, weights):
            return super().load_weights(self.hf_to_vllm_mapper.apply(weights))

        def get_mrope_input_positions(self, input_tokens, mm_features):
            # The text backbone is trained with interleaved M-RoPE (the config
            # carries mrope_section even standalone). With no vision features,
            # all three T/H/W rows are the plain position index and the decode
            # delta is 0 -- identical to what the multimodal class computes
            # for pure-text prompts.
            if mm_features:
                raise ValueError("text-only checkpoint got multimodal features")
            n = len(input_tokens)
            positions = torch.arange(n, dtype=torch.int64).unsqueeze(0).repeat(3, 1)
            return positions, 0

    globals()[name] = Qwen3_5TextForCausalLM  # cache: stable class identity
    return Qwen3_5TextForCausalLM
