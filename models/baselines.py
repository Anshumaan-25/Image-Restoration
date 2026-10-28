"""Compact explicitly non-adversarial baseline models; no weight downloads."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from models.common import validate_batch
from utils.config import ModelConfig


class DnCNNRGB(nn.Module):
    """DnCNN-style RGB residual estimator; L1 training is a project variant."""

    def __init__(self, width: int = 64, depth: int = 20) -> None:
        super().__init__()
        if width < 1 or depth < 3:
            raise ValueError("width >=1 and depth >=3 required")
        layers: list[nn.Module] = [nn.Conv2d(3, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()])
        layers.append(nn.Conv2d(width, 3, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map noisy BCHW RGB to an unclamped same-size RGB estimate."""
        validate_batch(x)
        return x - self.body(x)


class Identity(nn.Module):
    """Input-quality reference for denoising and JPEG tasks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the same finite RGB BCHW tensor."""
        validate_batch(x)
        return x


class Bicubic(nn.Module):
    """Torch bicubic SR reference, not a MATLAB benchmark degradation operator."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("SR scale must be 2 or 4")
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample RGB BCHW to Bx3xsHxsW without clamping."""
        validate_batch(x)
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


def build_model(spec: ModelConfig) -> nn.Module:
    """Construct an implemented model only; never download weights."""
    if spec.name == "dncnn_rgb_l1" and spec.scale == 1:
        return DnCNNRGB(spec.width, spec.depth)
    if spec.name == "identity" and spec.scale == 1:
        return Identity()
    if spec.name == "bicubic":
        return Bicubic(spec.scale)
    if spec.name == "edsr_baseline":
        from models.edsr import EDSRBaseline
        return EDSRBaseline(spec.scale, spec.width, spec.blocks)
    if spec.name == "restormer_denoising" and spec.scale == 1:
        from models.transformers import RestormerDenoising
        return RestormerDenoising()
    if spec.name == "swinir_light_sr" and spec.scale in (2,4):
        from models.transformers import SwinIRLightSR
        return SwinIRLightSR(spec.scale)
    raise ValueError(f"Unsupported model/scale: {spec}")
