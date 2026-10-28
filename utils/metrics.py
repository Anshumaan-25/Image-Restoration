"""Project RGB float metrics: clamp once, CPU float64, per-image aggregation."""
from __future__ import annotations

from typing import Any
import math
import numpy as np
import torch
import torch.nn.functional as F


def _pair(pred: torch.Tensor, target: torch.Tensor, border: int) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred, target = pred[0], target[0]
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[0] != 3 or min(pred.shape) < 1:
        raise ValueError("Metric inputs must be matching RGB CHW (or single-image BCHW)")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise ValueError("Metric images must be floating point")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite metric input")
    if target.min() < 0 or target.max() > 1:
        raise ValueError("Metric target must be in [0,1]")
    if type(border) is not int or border < 0 or min(pred.shape[-2:]) <= 2 * border:
        raise ValueError("Invalid metric crop border")
    a, b = pred.detach().cpu().double().clamp(0, 1), target.detach().cpu().double()
    if border:
        a, b = a[:, border:-border, border:-border], b[:, border:-border, border:-border]
    return a, b


def psnr(pred: torch.Tensor, target: torch.Tensor, border: int = 0) -> float:
    """PSNR with peak=1; exact reconstruction gives positive infinity."""
    a, b = _pair(pred, target, border)
    mse = float(np.mean((a.numpy() - b.numpy()) ** 2))
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, border: int = 0) -> float:
    """11x11 Gaussian sigma=1.5, population covariance, valid windows, RGB mean."""
    a, b = _pair(pred, target, border)
    if min(a.shape[-2:]) < 11:
        raise ValueError("SSIM requires at least 11x11 pixels after border cropping")
    positions = torch.arange(11, dtype=torch.float64) - 5
    g = torch.exp(-positions.square() / (2 * 1.5**2))
    g /= g.sum()
    kernel = torch.outer(g, g)[None, None].expand(3, 1, 11, 11)
    def window(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x[None], kernel, groups=3)[0]
    ma, mb = window(a), window(b)
    va, vb, cov = window(a*a) - ma*ma, window(b*b) - mb*mb, window(a*b) - ma*mb
    value = ((2*ma*mb + .01**2) * (2*cov + .03**2)) / ((ma*ma + mb*mb + .01**2) * (va + vb + .03**2))
    return float(value.mean())


def evaluate_pair(pred: torch.Tensor, target: torch.Tensor, border: int = 0) -> dict[str, float]:
    """Score one image under project_rgb_float_v1/project_ssim_v1."""
    return {"psnr": psnr(pred, target, border), "ssim": ssim(pred, target, border)}


def aggregate(rows: list[dict[str, float]]) -> dict[str, float | int]:
    """Mean individual scores; do not drop infinity or average pooled MSE."""
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")
    if any(math.isnan(r[k]) for r in rows for k in ("psnr", "ssim")):
        raise ValueError("NaN metric encountered")
    if any(r["psnr"] == -float("inf") or not math.isfinite(r["ssim"]) for r in rows):
        raise ValueError("Invalid infinite metric encountered")
    return {"count": len(rows), "psnr": float(np.mean([r["psnr"] for r in rows])),
            "ssim": float(np.mean([r["ssim"] for r in rows]))}


def json_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Strict JSON representation of infinite PSNR with an explicit flag."""
    result = dict(row)
    if isinstance(result.get("psnr"), (int, float)) and math.isinf(result["psnr"]):
        if result["psnr"] < 0:
            raise ValueError("Negative infinite PSNR is invalid")
        result["psnr"] = None
        result["psnr_is_infinite"] = True
    else:
        result["psnr_is_infinite"] = False
    return result
