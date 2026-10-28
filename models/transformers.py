"""Approved supervised transformer variants with finite RGB and dynamic-size contracts."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from models.common import validate_batch
from models.vendor.restormer import Restormer
from models.vendor.swinir import SwinIR


def pad_to_multiple(x: torch.Tensor,multiple: int) -> torch.Tensor:
    """Bottom/right reflect-pad when legal, replicate-pad undersized dimensions."""
    if type(multiple) is not int or multiple<1:raise ValueError('Positive integer padding multiple required')
    h,w=x.shape[-2:];ph=(-h)%multiple;pw=(-w)%multiple
    if pw:x=F.pad(x,(0,pw,0,0),mode='reflect' if w>pw else 'replicate')
    if ph:x=F.pad(x,(0,0,0,ph),mode='reflect' if h>ph else 'replicate')
    return x


class RestormerDenoising(Restormer):
    """Canonical RGB denoising model; tiny overrides exist solely for procedural tests."""
    def __init__(self,dim: int=48,num_blocks: tuple[int,...]=(4,6,6,8),
                 refinement_blocks: int=4) -> None:
        if type(dim) is not int or dim<2 or dim%2 or len(num_blocks)!=4 or any(type(v) is not int or v<1 for v in num_blocks) or refinement_blocks<1:
            raise ValueError('Positive even width, four positive block counts and refinement required')
        super().__init__(inp_channels=3,out_channels=3,dim=dim,num_blocks=list(num_blocks),
                         num_refinement_blocks=refinement_blocks,heads=[1,2,4,8],
                         ffn_expansion_factor=2.66,bias=False,LayerNorm_type='BiasFree',dual_pixel_task=False)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """Restore arbitrary positive RGB H/W; retain unrestricted prediction values."""
        validate_batch(x);h,w=x.shape[-2:]
        return super().forward(pad_to_multiple(x,8))[:,:,:h,:w]


class SwinIRLightSR(SwinIR):
    """Official lightweight classical SR configuration with separate x2/x4 weights."""
    def __init__(self,scale: int,embed_dim: int=60,depths: tuple[int,...]=(6,6,6,6),
                 num_heads: tuple[int,...]=(6,6,6,6)) -> None:
        if scale not in (2,4) or not depths or len(depths)!=len(num_heads) or any(d<1 for d in depths) or any(h<1 or embed_dim%h for h in num_heads):
            raise ValueError('SwinIR requires scale2/4 and compatible positive depths/heads')
        super().__init__(upscale=scale,in_chans=3,img_size=64,window_size=8,img_range=1.,
                         depths=list(depths),embed_dim=embed_dim,num_heads=list(num_heads),mlp_ratio=2,
                         upsampler='pixelshuffledirect',resi_connection='1conv')

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """Pad tiny/odd LR inputs dynamically and unpad to exact scaled dimensions."""
        validate_batch(x);h,w=x.shape[-2:]
        return super().forward(pad_to_multiple(x,8))[:,:,:h*self.upscale,:w*self.upscale]
