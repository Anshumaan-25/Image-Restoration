"""Independent primary-source SR metric parity and RGB/BGR/border edge cases."""
from pathlib import Path
import importlib.util
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from utils.sr_metrics import evaluate_sr_replay


class SRReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_pinned_upstream_y_psnr_and_independent_numpy_ssim_parity(self):
        source=Path(__file__).parent/'fixtures/upstream/swinir_metrics_reference.txt'
        ref=types.ModuleType('upstream_metrics')
        # Only upstream NumPy color/PSNR functions are called. OpenCV is not
        # required: SSIM uses the independent valid-window NumPy reference.
        import sys
        with patch.dict(sys.modules,{'cv2':types.ModuleType('unused_cv2')}):
            exec(compile(source.read_text(),str(source),'exec'),ref.__dict__)
        from tests.test_metrics import reference_ssim
        generator=torch.Generator().manual_seed(19)
        for scale in (2,4):
            target=torch.rand(3,31,37,generator=generator)
            pred=target+torch.randn(3,31,37,generator=generator)*.08
            actual=evaluate_sr_replay(pred,target,scale)
            def bgr(x):return x.clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()[...,::-1]
            a=ref.to_y_channel(bgr(pred)[scale:-scale,scale:-scale].astype(np.float64))
            b=ref.to_y_channel(bgr(target)[scale:-scale,scale:-scale].astype(np.float64))
            expected={'psnr':ref.calculate_psnr(bgr(pred),bgr(target),scale,test_y_channel=True),
                      'ssim':reference_ssim(a.transpose(2,0,1).astype(np.float64)/255.,b.transpose(2,0,1).astype(np.float64)/255.)}
            for key in actual:self.assertAlmostEqual(actual[key],expected[key],delta=1e-6)

    def test_color_weights_borders_infinity_and_invalid_geometry(self):
        target=torch.zeros(3,25,25)
        red=target.clone();red[0]=.2
        blue=target.clone();blue[2]=.2
        self.assertLess(evaluate_sr_replay(red,target,2)['psnr'],evaluate_sr_replay(blue,target,2)['psnr'])
        self.assertEqual(evaluate_sr_replay(target,target,2)['psnr'],float('inf'))
        altered=target.clone();altered[:,:2]=1
        self.assertEqual(evaluate_sr_replay(altered,target,2)['psnr'],float('inf'))
        for scale in (1,3):
            with self.assertRaises(ValueError):evaluate_sr_replay(target,target,scale)
        with self.assertRaises(ValueError):evaluate_sr_replay(torch.zeros(3,12,12),torch.zeros(3,12,12),2)

    def test_replay_padding_retains_official_extra_window_and_exact_crop(self):
        from utils.sr_metrics import swinir_reference_forward
        class Recorder(torch.nn.Module):
            def forward(self,x):self.received=x.clone();return x.repeat_interleave(2,2).repeat_interleave(2,3)
        model=Recorder();x=torch.arange(3*8*9.).reshape(1,3,8,9)
        result=swinir_reference_forward(model,x,2)
        self.assertEqual(result.shape,(1,3,16,18))
        self.assertEqual(model.received.shape,(1,3,16,16))
        self.assertTrue(torch.equal(model.received[:,:,:8,:9],x))
        self.assertTrue(torch.equal(model.received[:,:,8:,:9],torch.flip(x,[2])))
        with self.assertRaises(ValueError):swinir_reference_forward(model,torch.ones(1,3,1,1),2)


if __name__=='__main__':unittest.main(verbosity=2)
