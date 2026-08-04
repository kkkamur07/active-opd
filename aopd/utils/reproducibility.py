"""Reproducibility and device-memory helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch when those dependencies are available."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def peak_cuda_memory(reset: bool = False) -> int | None:
    """Return peak allocated CUDA bytes, or ``None`` on CPU-only machines."""

    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.max_memory_allocated())


def configure_cuda_memory(*, allow_tf32: bool = True) -> None:
    """Apply safe global CUDA math settings when a GPU is available."""

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


__all__ = ["configure_cuda_memory", "peak_cuda_memory", "seed_everything"]
