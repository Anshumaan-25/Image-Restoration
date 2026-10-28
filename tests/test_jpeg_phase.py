"""Scientific contracts for matched JPEG views, losses and deterministic replay."""
from pathlib import Path
import tempfile
import csv
import unittest
from unittest.mock import patch
import torch
from losses.consistency import phase_consistency
from losses import CompositeLoss
from train import run_training
from utils.checkpoint import load_checkpoint
from utils.config import load_config, config_from_dict
from utils.dataset_loader import RestorationDataset
from utils.jpeg_phase import ALL_PHASES, DEV_PHASES, clean_context, phase_pair, phase_view
from utils.phase_evaluation import evaluate_phases
from models.baselines import Identity


class JPEGPhaseTests(unittest.TestCase):
    """Test alignment independently of compression and both-branch gradients."""
    def setUp(self) -> None:
        torch.set_num_threads(2)
        self.cfg = load_config(Path(__file__).resolve().parents[1]/'configs/smoke.yaml')
        self.cfg.training.device = 'cpu'
        self.cfg.jpeg_study.mode = 'stratified'
        self.cfg.degradation.profile = 'jpeg_rgb_444_v1'
        self.cfg.dataset.patch_size = 16
        self.cfg.dataset.val_samples = 1

    def test_all_phase_geometry_without_codec(self) -> None:
        context = torch.arange(3*48*48).reshape(3,48,48).float()
        target = context[:,16:32,16:32]
        with patch('utils.jpeg_phase.jpeg_roundtrip', side_effect=lambda image, quality: image):
            for phase in ALL_PHASES:
                noisy, clean = phase_view(context,16,phase,20)
                self.assertTrue(torch.equal(noisy,target), phase)
                self.assertTrue(torch.equal(clean,target), phase)
        with self.assertRaises(ValueError):
            phase_view(context,16,(8,0),20)
        with self.assertRaises(ValueError):
            clean_context(torch.zeros(3,47,48),16)

    def test_cycle_and_deterministic_aligned_controls(self) -> None:
        for cycle in (0,64):
            pairs=[phase_pair(42,cycle+i,'stratified') for i in range(64)]
            self.assertEqual({a for a,b in pairs},set(ALL_PHASES))
            self.assertEqual({b for a,b in pairs},set(ALL_PHASES))
            self.assertTrue(all(a != b for a,b in pairs))
        target=None
        for mode in ('random','stratified'):
            self.cfg.jpeg_study.mode=mode
            dataset=RestorationDataset(self.cfg,'train',True)
            a,b=dataset[5],dataset[5]
            self.assertTrue(torch.equal(a['input'],b['input']))
            self.assertEqual(a['input'].shape,(3,16,16))
            if target is not None:
                self.assertTrue(torch.equal(target,a['target']))
            target=a['target']

    def test_consistency_both_gradients_and_zero_weight(self) -> None:
        a=torch.ones(2,3,12,12,requires_grad=True)
        b=torch.zeros_like(a,requires_grad=True)
        value=phase_consistency(a,b)
        self.assertEqual(float(value.detach()),1.0)
        value.backward()
        self.assertTrue((a.grad > 0).all() and (b.grad < 0).all())
        pixel=CompositeLoss()
        target=torch.rand_like(a)
        combined,_=pixel(torch.cat((a,b)),torch.cat((target,target)))
        first,_=pixel(a,target)
        second,_=pixel(b,target)
        torch.testing.assert_close(combined,.5*(first+second))
        torch.testing.assert_close(combined+0*phase_consistency(a,b),combined)
        with self.assertRaises(ValueError):
            config_from_dict({'loss':{'consistency_weight':.05}})

    def test_fixed_panel_and_constant_perfect_reconstruction(self) -> None:
        dataset=RestorationDataset(self.cfg,'val')
        with patch.object(dataset,'clean_source',return_value=(torch.zeros(3,48,48),'constant')):
            result=evaluate_phases(Identity(),dataset,torch.device('cpu'))
        self.assertEqual(result['phase_count'],16)
        self.assertEqual(result['count'],1)
        self.assertEqual(result['worst_phase_psnr'],float('inf'))
        self.assertEqual(result['phase_spread_psnr'],0)
        self.assertEqual(result['images'][0]['phases'][0]['edge_error'],0)
        self.assertEqual(result['phase_panel'],[list(p) for p in DEV_PHASES])

    def test_candidate_exact_resume_and_profile_no_validation(self) -> None:
        cfg=self.cfg
        cfg.training.max_steps=3
        cfg.loss.consistency_weight=.05
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run_training(cfg,root/'full')
            run_training(cfg,root/'split',stop_after_updates=1)
            run_training(cfg,resume=root/'split/latest.pth')
            a,b=(load_checkpoint(root/p/'latest.pth') for p in ('full','split'))
            with (root/'full/metrics.csv').open() as handle:
                rows=list(csv.DictReader(handle))
            self.assertTrue(rows[-1]['worst_phase_psnr'])
            self.assertEqual(a['cursor'],b['cursor'])
            self.assertTrue(torch.equal(a['rng']['torch'],b['rng']['torch']))
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)
            self.assertEqual(a['scheduler'],b['scheduler'])
            with patch('train.validate',side_effect=AssertionError('profile must not validate')):
                result=run_training(cfg,root/'profile',profile_only=True)
            self.assertTrue(result['profile_only'])
            self.assertFalse((root/'profile/best_model.pth').exists())
