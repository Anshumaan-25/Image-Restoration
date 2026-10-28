"""EDSR baseline topology with normalized RGB public inputs and no weight download.

Independent implementation of Lim et al., CVPR Workshops 2017. The official
EDSR-PyTorch source documents the residual trunk/PixelShuffle/mean conventions:
https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/edsr.py
This module is trained from scratch; upstream checkpoints are not imported.
"""
import torch
from torch import nn
from models.common import validate_batch


class ResidualBlock(nn.Module):
    """Two convolutions, ReLU and identity; no batch normalization."""
    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return residual-scale-one feature update."""
        return x + self.body(x)


class EDSRBaseline(nn.Module):
    """16 blocks, 64 features by default; separate x2 and x4 heads."""
    def __init__(self, scale: int, width: int = 64, blocks: int = 16) -> None:
        super().__init__()
        if scale not in (2, 4) or width < 1 or blocks < 1:
            raise ValueError("EDSR requires scale 2/4, positive width and blocks")
        self.register_buffer("mean", torch.tensor((.4488, .4371, .4040)).view(1, 3, 1, 1))
        self.head = nn.Conv2d(3, width, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)], nn.Conv2d(width, width, 3, padding=1))
        tail = []
        for _ in range(1 if scale == 2 else 2):
            tail.extend([nn.Conv2d(width, 4*width, 3, padding=1), nn.PixelShuffle(2)])
        tail.append(nn.Conv2d(width, 3, 3, padding=1))
        self.tail = nn.Sequential(*tail)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map arbitrary positive LR H/W to RGB sH/sW, without clamping."""
        validate_batch(x)
        features = self.head(x - self.mean)
        return self.tail(features + self.body(features)) + self.mean
