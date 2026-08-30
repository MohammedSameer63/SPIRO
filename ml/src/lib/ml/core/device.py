"""
SPIRO ML — Device Resolution
Centralises GPU/CPU/MPS selection so every component uses the same logic.
"""
from __future__ import annotations

import torch
from lib.ml.core.logger import get_logger

log = get_logger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    """
    Resolve the best available torch.device.

    Parameters
    ----------
    requested : str
        "auto" | "cpu" | "cuda" | "cuda:0" | "mps"

    Returns
    -------
    torch.device
    """
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)

    log.info(f"Using device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(device)}")
        log.info(
            f"  Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB"
        )
    return device


def get_device_info() -> dict:
    """Return a dict with full device environment info."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
        "mps_available": (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    return info
