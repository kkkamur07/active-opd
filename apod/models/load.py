"""Load a Hugging Face causal LM. Teacher loading always freezes weights."""

from __future__ import annotations

import torch


def _require_cuda(device_map: str) -> None:
    """Fail loudly when an accelerator was asked for but torch cannot see one.

    ``device_map="auto"`` silently places the model on CPU when CUDA is
    unavailable, which turns a broken install into a run that merely looks slow.
    Pass ``device_map="cpu"`` to opt into CPU on purpose.
    """

    if device_map == "cpu" or torch.cuda.is_available():
        return

    raise RuntimeError(
        f"device_map={device_map!r} needs CUDA but torch.cuda.is_available() is False "
        f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). "
        "Generation would silently run on CPU. Check that the installed torch build "
        "matches the driver reported by nvidia-smi, or pass device_map='cpu'."
    )


def load_lm(model_id: str, *, frozen: bool = False, device_map: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _require_cuda(device_map)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    if frozen:
        model.eval()
        model.requires_grad_(False)

    return tokenizer, model
