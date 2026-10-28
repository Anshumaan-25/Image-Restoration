"""Equivalent CPU float64 separable implementation of project RGB SSIM.

Evaluation helper only; the frozen training/source snapshot is not modified.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.ndimage import convolve1d
from utils.metrics import _pair


def evaluate_pair_fast(pred: torch.Tensor, target: torch.Tensor, border: int = 0) -> dict[str, float]:
    """Match project_rgb_float_v1 with separable valid Gaussian11 SSIM.

    The 11x11 Gaussian is an outer product, so two float64 one-dimensional
    convolutions are mathematically equivalent. Summation rounding is checked
    against the frozen reference before cloud scoring; this is not a new metric.
    """
    ta,tb=_pair(pred,target,border)
    if min(ta.shape[-2:])<11:raise ValueError('SSIM requires at least 11x11 pixels after border cropping')
    a,b=ta.numpy(),tb.numpy()
    mse=float(np.mean((a-b)**2))
    g=np.exp(-np.square(np.arange(11,dtype=np.float64)-5)/(2*1.5**2));g/=g.sum()
    def window(x: np.ndarray) -> np.ndarray:
        return convolve1d(convolve1d(x,g,axis=-1,mode='constant'),g,axis=-2,mode='constant')[:,5:-5,5:-5]
    ma,mb=window(a),window(b)
    va,vb,cov=window(a*a)-ma*ma,window(b*b)-mb*mb,window(a*b)-ma*mb
    value=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return {'psnr':float('inf') if mse==0 else -10*math.log10(mse),'ssim':float(value.mean())}
