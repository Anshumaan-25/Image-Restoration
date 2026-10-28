"""Named RGB degradation operators with explicit CPU randomness."""
from __future__ import annotations

from io import BytesIO
import math
import numpy as np
from PIL import Image, features
import torch
import torch.nn.functional as F


def validate_image(x: torch.Tensor) -> None:
    """Require finite, normalized CPU RGB CHW clean images."""
    if x.device.type != "cpu" or x.ndim != 3 or x.shape[0] != 3 or min(x.shape) < 1:
        raise ValueError("Degradation expects CPU RGB CHW")
    if not x.is_floating_point() or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Clean images must be finite floats in [0,1]")


def apply_gaussian_noise(x: torch.Tensor, sigma: float, generator: torch.Generator | None = None,
                         clipped: bool = False) -> torch.Tensor:
    """AWGN: independent epsilon * sigma/255; unclipped by default."""
    validate_image(x)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    output = x + torch.randn(x.shape, generator=generator, dtype=x.dtype) * (sigma / 255)
    return output.clamp(0, 1) if clipped else output


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """IEC sRGB inverse transfer, assuming normalized sRGB samples."""
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """sRGB transfer after the declared linear saturation boundary."""
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(0).pow(1 / 2.4) - 0.055)


def apply_poisson_noise(x: torch.Tensor, peak_photons: float,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """Approximate linear-RGB shot noise; not a raw camera simulation."""
    validate_image(x)
    if not math.isfinite(peak_photons) or peak_photons <= 0:
        raise ValueError("peak_photons must be positive")
    noisy = torch.poisson(srgb_to_linear(x) * peak_photons, generator=generator) / peak_photons
    return linear_to_srgb(noisy.clamp(0, 1))


def jpeg_roundtrip(x: torch.Tensor, quality: int = 20) -> torch.Tensor:
    """Real Pillow JPEG RGB 4:4:4 encode/decode with pinned explicit options."""
    validate_image(x)
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be integer 1..100")
    array = x.mul(255).round().byte().permute(1, 2, 0).numpy()
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="JPEG", quality=quality, subsampling=0,
                               optimize=False, progressive=False)
    stream.seek(0)
    with Image.open(stream) as decoded:
        restored = np.array(decoded.convert("RGB"), copy=True)
    return torch.from_numpy(restored).permute(2, 0, 1).float() / 255


def bicubic_fixture(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Torch antialiased LR fixture; explicitly not MATLAB benchmark bicubic."""
    validate_image(x)
    if scale not in (2, 4) or x.shape[-2] % scale or x.shape[-1] % scale:
        raise ValueError("Fixture dimensions must be divisible by scale 2/4")
    return F.interpolate(x[None], size=(x.shape[-2] // scale, x.shape[-1] // scale),
                         mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)


def codec_metadata() -> dict[str, str]:
    """Record actual codec versions; no assumption of cross-codec equivalence."""
    import PIL
    return {"pillow": PIL.__version__, "jpeg_library": str(features.version_codec("jpg")),
            "subsampling": "4:4:4", "optimize": "false", "progressive": "false"}
