"""Pinned SwinIR-style uint8/Y-channel SR metric replay, separate from RGB float scores."""
from __future__ import annotations
import numpy as np
import torch
from utils.metrics import _pair,evaluate_pair


def _official_y(image: torch.Tensor) -> torch.Tensor:
    """Match official RGB-export -> BGR uint8 -> float32 BT.601 Y conversion.

    SwinIR official utils/util_calculate_psnr_ssim.py @ 6545850fbf8df298df73d81f3e8cba638787c8bd
    supplies the reference; license/attribution lives in docs/third_party/SwinIR.LICENSE.
    Coefficients are applied in explicitly reversed BGR order, not RGB by accident.
    """
    rgb=image.detach().cpu().float().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()
    bgr=rgb[...,::-1].astype(np.float32)/255.
    raw=np.dot(bgr,[24.966,128.553,65.481])+16.
    y255=(raw/255.).astype(np.float32)*255.
    y=torch.from_numpy(y255.astype(np.float64)/255.)[None]
    return y.expand(3,-1,-1)


def evaluate_sr_replay(pred: torch.Tensor,target: torch.Tensor,scale: int) -> dict[str,float]:
    """Quantize, use MATLAB-convention Y, crop scale pixels and score valid-window SSIM."""
    if scale not in (2,4):raise ValueError('Replay supports SR x2/x4 only')
    _pair(pred,target,scale)
    if pred.ndim==4:pred,target=pred[0],target[0]
    return evaluate_pair(_official_y(pred),_official_y(target),scale)


def swinir_reference_forward(model: torch.nn.Module,x: torch.Tensor,scale: int) -> torch.Tensor:
    """Match pinned main_test_swinir's symmetric flip padding, including one full window.

    This evaluation-only outer padding differs from the general model's reflection
    padding contract. Reject undersized replay inputs rather than label a changed
    boundary convention an exact reference replay.
    """
    h,w=x.shape[-2:]
    if min(h,w)<8:raise ValueError('Official SwinIR replay requires LR dimensions at least window8')
    ph=(h//8+1)*8-h;pw=(w//8+1)*8-w
    padded=torch.cat([x,torch.flip(x,[2])],2)[:,:,:h+ph,:]
    padded=torch.cat([padded,torch.flip(padded,[3])],3)[:,:,:,:w+pw]
    return model(padded)[:,:,:h*scale,:w*scale]
