"""Two-view prediction agreement; gradients flow through both predictions."""
import torch


def phase_consistency(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return FP32 mean absolute disagreement for matching finite RGB batches."""
    if a.shape != b.shape or a.ndim != 4 or a.shape[1] != 3 or min(a.shape) < 1:
        raise ValueError("Consistency requires matching nonempty RGB BCHW predictions")
    if not a.is_floating_point() or not b.is_floating_point() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Consistency predictions must be finite floating tensors")
    return (a.float() - b.float()).abs().mean()
