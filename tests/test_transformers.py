"""Approved transformer contracts and FP32 parity against pinned upstream definitions."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch
from torch import nn
from models.transformers import RestormerDenoising,SwinIRLightSR,pad_to_multiple
from utils.config import config_from_dict

ROOT=Path(__file__).resolve().parents[1]


def reference_rearrange(x: torch.Tensor,pattern: str,**axes: int) -> torch.Tensor:
    """Independent einsum/permutation reference for upstream einops calls."""
    if pattern=='b c h w -> b (h w) c':return torch.einsum('bchw->bhwc',x).reshape(x.shape[0],-1,x.shape[1])
    if pattern=='b (h w) c -> b c h w':return torch.einsum('bhwc->bchw',x.reshape(x.shape[0],axes['h'],axes['w'],x.shape[-1]))
    if pattern=='b (head c) h w -> b head c (h w)':
        b,c,h,w=x.shape
        return torch.stack([x[:,i*(c//axes['head']):(i+1)*(c//axes['head'])].flatten(2) for i in range(axes['head'])],dim=1)
    if pattern=='b head c (h w) -> b (head c) h w':
        return torch.cat([x[:,i].reshape(x.shape[0],x.shape[2],axes['h'],axes['w']) for i in range(x.shape[1])],dim=1)
    raise ValueError(pattern)


def upstream(name: str):
    """Load exact retained upstream source with dependency shims used only by tests."""
    module=types.ModuleType('pinned_reference_'+name)
    if name=='restormer':
        shim=types.ModuleType('einops');shim.rearrange=reference_rearrange
        modules={'einops':shim}
    else:
        from torchvision.ops import StochasticDepth
        class ReferenceDropPath(StochasticDepth):
            def __init__(self,p):super().__init__(p,'row')
        shim=types.ModuleType('timm.models.layers')
        shim.DropPath=ReferenceDropPath;shim.to_2tuple=lambda x:(x,x) if isinstance(x,int) else tuple(x)
        shim.trunc_normal_=nn.init.trunc_normal_
        modules={'timm':types.ModuleType('timm'),'timm.models':types.ModuleType('timm.models'),'timm.models.layers':shim}
    source=ROOT/f'tests/fixtures/upstream/{name}_reference.txt'
    with patch.dict(sys.modules,modules):exec(compile(source.read_text(),str(source),'exec'),module.__dict__)
    return module


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_restormer_state_dict_forward_and_gradient_reference_parity(self):
        model=RestormerDenoising(4,(1,1,1,1),1)
        ref=upstream('restormer').Restormer(dim=4,num_blocks=[1]*4,num_refinement_blocks=1,LayerNorm_type='BiasFree')
        ref.load_state_dict(model.state_dict(),strict=True)
        model.eval();ref.eval()
        x=torch.rand(1,3,16,24,requires_grad=True);other=x.detach().clone().requires_grad_(True)
        a=model(x);b=ref(other)
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
        a.square().mean().backward();b.square().mean().backward()
        torch.testing.assert_close(x.grad,other.grad,atol=1e-5,rtol=1e-5)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_swinir_reference_parity_dynamic_masks_and_scale_specific_heads(self):
        for scale in (2,4):
            model=SwinIRLightSR(scale,12,(2,2),(3,3)).eval()
            ref=upstream('swinir').SwinIR(upscale=scale,in_chans=3,img_size=64,window_size=8,img_range=1.,depths=[2,2],embed_dim=12,num_heads=[3,3],mlp_ratio=2,upsampler='pixelshuffledirect',resi_connection='1conv').eval()
            ref.load_state_dict(model.state_dict(),strict=True)
            for h,w in [(8,16),(16,24)]:
                x=torch.rand(1,3,h,w)
                with torch.no_grad():torch.testing.assert_close(model(x),ref(x),atol=1e-5,rtol=1e-5)
            x=torch.rand(1,3,9,11,requires_grad=True)
            model(x).mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.abs().sum()),0)

    def test_tiny_odd_shapes_unclamped_outputs_and_padding(self):
        for model,scale in [(RestormerDenoising(4,(1,1,1,1),1),1),(SwinIRLightSR(2,12,(2,),(3,)),2),(SwinIRLightSR(4,12,(2,),(3,)),4)]:
            model.eval()
            for h,w in [(1,1),(1,9),(3,5),(9,11)]:
                with torch.no_grad():result=model(torch.rand(1,3,h,w))
                self.assertEqual(result.shape,(1,3,h*scale,w*scale))
                self.assertTrue(torch.isfinite(result).all())
            with self.assertRaises(ValueError):model(torch.ones(1,1,8,8))
        x=torch.arange(9.).reshape(1,1,3,3)
        self.assertTrue(torch.equal(pad_to_multiple(x,4)[:,:,:3,:3],x))
        model=RestormerDenoising(4,(1,1,1,1),1).eval()
        for parameter in model.parameters():parameter.data.zero_()
        with torch.no_grad():self.assertGreater(float(model(torch.full((1,3,8,8),2.)).max()),1)

    def test_task_configurations_and_invalid_combinations(self):
        config_from_dict({'model':{'name':'restormer_denoising'},'degradation':{'profile':'awgn_unclipped_v1'}})
        config_from_dict({'model':{'name':'swinir_light_sr','scale':2},'degradation':{'profile':'torch_bicubic_fixture_v1'}})
        for raw in [{'model':{'name':'restormer_denoising'}},{'model':{'name':'swinir_light_sr'}}, {'model':{'name':'restormer_denoising','scale':2},'degradation':{'profile':'torch_bicubic_fixture_v1'}}]:
            with self.assertRaises(ValueError):config_from_dict(raw)

    def test_canonical_factory_models_have_finite_backward(self):
        from models import build_model
        for name,scale,profile in [('restormer_denoising',1,'awgn_unclipped_v1'),('swinir_light_sr',2,'torch_bicubic_fixture_v1'),('swinir_light_sr',4,'torch_bicubic_fixture_v1')]:
            cfg=config_from_dict({'model':{'name':name,'scale':scale},'degradation':{'profile':profile}})
            model=build_model(cfg.model)
            x=torch.rand(1,3,8,8,requires_grad=True)
            result=model(x);result.abs().mean().backward()
            self.assertEqual(result.shape,(1,3,8*scale,8*scale))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_actual_mps_small_transformer_forward_backward(self):
        for model,scale in [(RestormerDenoising(4,(1,1,1,1),1),1),(SwinIRLightSR(2,12,(2,),(3,)),2)]:
            model=model.to('mps');x=torch.rand(1,3,9,11,device='mps',requires_grad=True)
            result=model(x);result.square().mean().backward()
            self.assertEqual(result.shape,(1,3,9*scale,11*scale))
            self.assertTrue(torch.isfinite(x.grad).all().item())


if __name__=='__main__':unittest.main(verbosity=2)
