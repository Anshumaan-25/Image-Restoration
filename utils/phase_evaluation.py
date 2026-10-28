"""Fixed patch phase panels and image-level robustness endpoints."""
from __future__ import annotations
from typing import Any

import numpy as np
import torch
from torch import nn
from utils.dataset_loader import RestorationDataset
from utils.jpeg_phase import DEV_PHASES, clean_context, phase_view
from utils.metrics import aggregate, evaluate_pair


@torch.no_grad()
def evaluate_phases(model: nn.Module, dataset: RestorationDataset, device: torch.device,
                    phases: tuple[tuple[int, int], ...] = DEV_PHASES,
                    quality: int | None = None) -> dict[str, Any]:
    """Stream phases one at a time; aggregate phases within each image first."""
    if not phases or len(set(phases)) != len(phases):
        raise ValueError("Nonempty unique phase panel required")
    model.eval()
    cfg = dataset.cfg
    quality = cfg.degradation.jpeg_quality if quality is None else quality
    images = []
    for index in range(len(dataset)):
        source, image_id = dataset.clean_source(index)
        context = clean_context(source, cfg.dataset.patch_size)
        rows = []
        for phase in phases:
            degraded, target = phase_view(context, cfg.dataset.patch_size, phase, quality)
            prediction = model(degraded[None].to(device)).float().cpu()[0]
            scores = evaluate_pair(prediction, target)
            input_scores = evaluate_pair(degraded, target)
            # Edge fidelity diagnostic, distinct from PSNR/SSIM benchmark metrics.
            prediction = prediction.clamp(0, 1).double()
            target = target.double()
            error_y = (prediction.diff(dim=-2) - target.diff(dim=-2)).abs()
            error_x = (prediction.diff(dim=-1) - target.diff(dim=-1)).abs()
            u, v = phase
            boundary_y = ((torch.arange(target.shape[-2]-1) + 1 + u) % 8 == 0)
            boundary_x = ((torch.arange(target.shape[-1]-1) + 1 + v) % 8 == 0)
            def edge_group(mask_y: torch.Tensor, mask_x: torch.Tensor) -> float:
                parts = [error_y[:, mask_y, :].reshape(-1), error_x[:, :, mask_x].reshape(-1)]
                values = torch.cat(parts)
                return float(values.mean()) if values.numel() else 0.0
            rows.append({"phase": list(phase), **scores,
                         "input_psnr": input_scores["psnr"], "input_ssim": input_scores["ssim"],
                         "edge_error": .5*(float(error_y.mean())+float(error_x.mean())),
                         "boundary_edge_error": edge_group(boundary_y, boundary_x),
                         "interior_edge_error": edge_group(~boundary_y, ~boundary_x)})
        values = [r["psnr"] for r in rows]
        images.append({"id": image_id, **aggregate(rows), "worst_phase_psnr": min(values),
                       "input_worst_phase_psnr": min(r["input_psnr"] for r in rows),
                       "input_mean_psnr": float(np.mean([r["input_psnr"] for r in rows])),
                       "input_mean_ssim": float(np.mean([r["input_ssim"] for r in rows])),
                       "phase_spread_psnr": 0.0 if all(np.isposinf(v) for v in values) else max(values)-min(values),
                       "phases": rows})
    metrics = aggregate(images)
    return {**metrics, "worst_phase_psnr": float(np.mean([i["worst_phase_psnr"] for i in images])),
            "input_worst_phase_psnr": float(np.mean([i["input_worst_phase_psnr"] for i in images])),
            "input_mean_psnr": float(np.mean([i["input_mean_psnr"] for i in images])),
            "input_mean_ssim": float(np.mean([i["input_mean_ssim"] for i in images])),
            "phase_spread_psnr": float(np.mean([i["phase_spread_psnr"] for i in images])),
            "phase_count": len(phases), "phase_panel": [list(p) for p in phases],
            "quality": quality, "patch_size": cfg.dataset.patch_size,
            "endpoint": "mean_per_image_worst_phase_psnr", "images": images}
