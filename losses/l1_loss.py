"""Pixel fidelity objectives without output clamping."""
import torch
import math
from torch import nn


class CharbonnierLoss(nn.Module):
    """Smooth L1 approximation, shifted so identical tensors have zero loss."""

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean sqrt(error²+epsilon²)-epsilon with FP32 reductions."""
        return ((pred.float() - target.float()).square().add(self.epsilon**2).sqrt() - self.epsilon).mean()
