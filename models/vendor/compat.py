"""Small native-torch replacements for the exact helpers used by pinned model ports."""
from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn

trunc_normal_ = nn.init.trunc_normal_


def to_2tuple(value: int | Sequence[int]) -> tuple[int, int]:
    """Normalize the upstream spatial size convention without importing timm."""
    if isinstance(value,int):return (value,value)
    if len(value)!=2:raise ValueError('Exactly two spatial dimensions required')
    return (value[0],value[1])


class DropPath(nn.Module):
    """Per-sample stochastic residual dropping; identity during evaluation."""
    def __init__(self,drop_prob: float=0.) -> None:
        super().__init__()
        if not 0<=drop_prob<1:raise ValueError('DropPath probability must be in [0,1)')
        self.drop_prob=drop_prob

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """Scale retained residual branches by inverse keep probability."""
        if not self.training or self.drop_prob==0:return x
        keep=1-self.drop_prob
        shape=(x.shape[0],)+(1,)*(x.ndim-1)
        mask=torch.empty(shape,dtype=x.dtype,device=x.device).bernoulli_(keep)
        return x*mask/keep


def rearrange(x: torch.Tensor,pattern: str,**axes: int) -> torch.Tensor:
    """Implement only Restormer's four explicitly pinned reshape/permutation patterns."""
    if pattern=='b c h w -> b (h w) c':return x.flatten(2).transpose(1,2)
    if pattern=='b (h w) c -> b c h w':return x.transpose(1,2).reshape(x.shape[0],x.shape[2],axes['h'],axes['w'])
    if pattern=='b (head c) h w -> b head c (h w)':
        b,c,h,w=x.shape
        if c%axes['head']:raise ValueError('Head count must divide channels')
        return x.reshape(b,axes['head'],c//axes['head'],h*w)
    if pattern=='b head c (h w) -> b (head c) h w':
        return x.reshape(x.shape[0],x.shape[1]*x.shape[2],axes['h'],axes['w'])
    raise ValueError(f'Unsupported pinned tensor permutation: {pattern}')
