"""Public tensor validation shared by the small baseline models."""
import torch


def validate_batch(x: torch.Tensor) -> None:
    """Require nonempty finite floating RGB BCHW; do not impose prediction range."""
    if x.ndim != 4 or x.shape[1] != 3 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError("Expected nonempty floating RGB BCHW input")
    if not torch.isfinite(x).all():
        raise ValueError("Model input contains nonfinite values")
