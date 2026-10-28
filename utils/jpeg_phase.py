"""Aligned JPEG phase views; no shifts, wrapped pixels or double compression."""
from __future__ import annotations

import torch
from utils.degradation import jpeg_roundtrip
from utils.reproducibility import stable_seed

DEV_PHASES = tuple((u, v) for u in (0, 2, 4, 6) for v in (0, 2, 4, 6))
ALL_PHASES = tuple((u, v) for u in range(8) for v in range(8))


def phase_pair(seed: int, ordinal: int, mode: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Each stratified cycle covers all 64 first phases; paired offset is 32."""
    if type(ordinal) is not int or ordinal < 0 or mode not in {"random", "stratified"}:
        raise ValueError("Nonnegative ordinal and random/stratified mode required")
    generator = torch.Generator().manual_seed(stable_seed(seed, "phase", ordinal if mode == "random" else ordinal // 64))
    if mode == "random":
        a, b = torch.randint(64, (2,), generator=generator).tolist()
    else:
        a = int(torch.randperm(64, generator=generator)[ordinal % 64])
        b = (a + 32) % 64
    return ALL_PHASES[a], ALL_PHASES[b]


def clean_context(image: torch.Tensor, patch: int, generator: torch.Generator | None = None,
                  augment: bool = False) -> torch.Tensor:
    """Select P+32 context, centered for evaluation, geometric changes before JPEG."""
    if type(patch) is not int or patch < 1 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Positive patch and RGB CHW image required")
    size = patch + 32
    if min(image.shape[-2:]) < size:
        raise ValueError(f"Phase study requires source dimensions >= {size}")
    if generator is None:
        top, left = ((image.shape[-2] - size) // 2, (image.shape[-1] - size) // 2)
    else:
        top = int(torch.randint(image.shape[-2] - size + 1, (), generator=generator))
        left = int(torch.randint(image.shape[-1] - size + 1, (), generator=generator))
    context = image[:, top:top+size, left:left+size]
    if augment:
        if generator is None:
            raise ValueError("Augmentation requires explicit randomness")
        context = torch.rot90(context, int(torch.randint(4, (), generator=generator)), (-2, -1))
        if bool(torch.randint(2, (), generator=generator)):
            context = context.flip(-1)
    return context.contiguous()


def phase_view(context: torch.Tensor, patch: int, phase: tuple[int, int], quality: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode P+8 region then extract P pixels with identical clean target content."""
    if len(phase) != 2 or any(type(v) is not int or not 0 <= v < 8 for v in phase):
        raise ValueError("JPEG phase coordinates must be integers in 0..7")
    if type(patch) is not int or patch < 1 or context.shape != (3, patch+32, patch+32):
        raise ValueError("Expected RGB context of shape (3,P+32,P+32)")
    u, v = phase
    region = context[:, 16-u:16-u+patch+8, 16-v:16-v+patch+8]
    decoded = jpeg_roundtrip(region, quality)
    return decoded[:, u:u+patch, v:v+patch].contiguous(), context[:, 16:16+patch, 16:16+patch].contiguous()
