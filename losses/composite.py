"""Pixel-only baseline criterion; optional pretrained features are not fabricated."""
import torch
from torch import nn
from losses.l1_loss import CharbonnierLoss


class CompositeLoss(nn.Module):
    """Compatible loss/log interface, with perceptual weight zero by default."""

    def __init__(self, kind: str = "l1", epsilon: float = 1e-3,
                 perceptual_weight: float = 0.0) -> None:
        super().__init__()
        if perceptual_weight:
            raise ValueError("Perceptual features are staged and require provisioned verified weights")
        if kind not in {"l1", "charbonnier"}:
            raise ValueError("Unsupported primary pixel objective")
        self.pixel = nn.L1Loss() if kind == "l1" else CharbonnierLoss(epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return a differentiable scalar and detached unweighted log values."""
        if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError("Loss requires matching RGB BCHW tensors")
        if min(pred.shape) < 1 or not pred.is_floating_point() or not target.is_floating_point():
            raise ValueError("Loss requires nonempty floating tensors")
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("Nonfinite loss inputs")
        loss = self.pixel(pred.float(), target.float())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite FP32 loss reduction")
        return loss, {"pixel": float(loss.detach())}
