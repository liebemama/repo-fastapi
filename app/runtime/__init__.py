# app/runtime/__init__.py
from __future__ import annotations

import torch

from app.core.config import get_settings


def pick_device() -> torch.device:
    """
    Choose runtime device based on settings and availability.
    """
    s = get_settings()
    dev_str = str(getattr(s, "DEVICE", "cuda:0")).lower()

    # Prefer explicit cuda if available
    if dev_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(dev_str)
        # fallback if requested cuda but not available
        return torch.device("cpu")

    # Apple Metal
    if dev_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # CPU
    return torch.device("cpu")


def pick_dtype(device: str | torch.device) -> torch.dtype:
    """
    Heuristic: fp16 on CUDA, fp32 otherwise.
    """
    d = str(device).lower()
    if d.startswith("cuda"):
        return torch.float16
    # mps half precision is still spotty → prefer fp32 for stability
    return torch.float32
