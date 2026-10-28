"""Learned scale-specific SR must share trainer, checkpoints and strict contracts."""
from pathlib import Path
import tempfile
import unittest
import torch
from models import build_model
from models.edsr import EDSRBaseline
from train import run_training
from utils.config import load_config, config_from_dict
from utils.checkpoint import load_checkpoint


class SRTests(unittest.TestCase):
    """Learned SR tests are fixtures, never official benchmark results."""
    def test_shapes_backward_and_no_batchnorm(self) -> None:
        torch.set_num_threads(2)
        for scale in (2,4):
            model=EDSRBaseline(scale,width=8,blocks=2)
            self.assertFalse(any(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()))
            for h,w in ((1,1),(3,5),(13,17)):
                x=torch.rand(1,3,h,w)
                y=model(x)
                self.assertEqual(y.shape,(1,3,h*scale,w*scale))
                y.abs().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_train_checkpoint_round_trip(self) -> None:
        cfg=load_config(Path(__file__).resolve().parents[1]/'configs/smoke.yaml')
        cfg.model.name='edsr_baseline'
        cfg.model.scale=2
        cfg.model.blocks=2
        cfg.training.device='cpu'
        cfg.degradation.profile='torch_bicubic_fixture_v1'
        cfg=config_from_dict(cfg.to_dict())
        with tempfile.TemporaryDirectory() as tmp:
            result=run_training(cfg,Path(tmp)/'sr')
            state=load_checkpoint(Path(result['run_dir'])/'best_model.pth')
            model=build_model(config_from_dict(state['config']).model)
            model.load_state_dict(state['model'],strict=True)
            with torch.no_grad():
                self.assertEqual(model(torch.rand(1,3,7,9)).shape,(1,3,14,18))
            self.assertTrue(result['completed'])

    def test_mean_convention_at_zero_residual(self) -> None:
        model=EDSRBaseline(4,width=8,blocks=2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            output=model(torch.rand(1,3,3,5))
        expected=torch.tensor((.4488,.4371,.4040)).view(1,3,1,1).expand(1,3,12,20)
        torch.testing.assert_close(output,expected)
