"""Shared device and mixed-precision runtime helpers."""

from contextlib import nullcontext

import torch


def select_device(name: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def select_precision(device: torch.device) -> str:
    """Choose the default numerical precision for a device."""
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def autocast_context(device: torch.device, precision: str):
    """Return CUDA autocast or a no-op context on other devices."""
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(enabled: bool):
    """Create a CUDA GradScaler across supported PyTorch AMP APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)
