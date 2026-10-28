"""Explicit device/precision policy; no mid-run fallback."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any
import platform
import importlib.metadata
import torch


def select_device(requested: str = "auto") -> torch.device:
    """Select once; an explicitly unavailable accelerator is an error."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available[requested]:
        raise RuntimeError(f"Requested {requested} is unavailable; no CPU fallback was applied")
    return torch.device(requested)


def precision_context(device: torch.device, precision: str) -> Any:
    """Use supported CUDA autocast; portable default is FP32."""
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("Reduced precision is currently validated only for CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support BF16")
    if precision not in {"fp16", "bf16"}:
        raise ValueError("Unknown precision")
    return torch.amp.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def synchronize(device: torch.device) -> None:
    """Synchronize accelerator work before measured wall-clock boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def environment(device: torch.device) -> dict[str, Any]:
    """Record actual runtime capabilities without inferring a GPU allocation."""
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "device": str(device),
            "cuda": torch.version.cuda, "mps_available": torch.backends.mps.is_available(),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "Pillow", "PyYAML", "tqdm")}}
