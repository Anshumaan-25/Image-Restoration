"""Explicit 8-bit RGB input and clamped PNG export."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image, ImageCms, ImageOps
import torch


def file_sha256(path: str | Path) -> str:
    """Hash file content without loading a dataset archive into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: str | Path) -> torch.Tensor:
    """Read RGB PNG/JPEG as float32 CHW; reject implicit color conversions."""
    with Image.open(path) as original:
        if original.format not in {"PNG", "JPEG"} or original.mode != "RGB":
            raise ValueError(f"Expected 8-bit RGB PNG/JPEG, got {original.format}/{original.mode}: {path}")
        # Pillow can downcast 16-bit RGB PNG silently; inspect the PNG IHDR.
        if original.format == "PNG":
            with Path(path).open("rb") as handle:
                header = handle.read(26)
            if len(header) < 26 or header[24] != 8:
                raise ValueError("Only 8-bit PNG inputs are supported")
            gamma = original.info.get("gamma")
            if gamma is not None and abs(float(gamma) - 0.45455) > .01:
                raise ValueError("Non-sRGB PNG gamma requires an explicit conversion")
        profile = original.info.get("icc_profile")
        if profile:
            description = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(profile)))
            if "srgb" not in description.lower().replace(" ", ""):
                raise ValueError("Non-sRGB ICC profile requires an explicit conversion")
        image = ImageOps.exif_transpose(original)
        array = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255)


def save_rgb(tensor: torch.Tensor, path: str | Path, overwrite: bool = False) -> None:
    """Export a finite CHW/one-image BCHW tensor; refuse accidental overwrites."""
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3 or not torch.isfinite(tensor).all():
        raise ValueError("Export requires a finite RGB CHW image")
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        raise ValueError("Restored output must be PNG")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().float().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(destination)
