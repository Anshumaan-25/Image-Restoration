"""Approved mixed photon-level training versus fixed reproducible validation."""
import unittest
import torch
import tempfile,json,subprocess,sys
from pathlib import Path
from utils.config import config_from_dict
from utils.dataset_loader import RestorationDataset


class PoissonSamplingTests(unittest.TestCase):
    def test_training_levels_are_explicit_repeatable_and_validation_is_fixed(self):
        cfg=config_from_dict({'dataset':{'kind':'procedural','patch_size':16,'val_samples':2},'degradation':{'profile':'poisson_linear_rgb_v1','peak_photons':60}})
        train=RestorationDataset(cfg,'train',True);val=RestorationDataset(cfg,'val')
        levels={train[index]['metadata']['peak_photons'] for index in range(64)}
        self.assertEqual(levels,{30.,60.,120.})
        self.assertTrue(torch.equal(train[7]['input'],train[7]['input']))
        self.assertEqual(val[0]['metadata']['peak_photons'],60)
        self.assertTrue(torch.equal(val[0]['input'],val[0]['input']))
        cfg.degradation.poisson_training_peaks=[30.]
        self.assertEqual(RestorationDataset(cfg,'train',True)[0]['metadata']['peak_photons'],30.)

    def test_invalid_training_peak_lists_fail_before_sampling(self):
        for value in ([],[0],[float('nan')],[30,30],None,[True]):
            with self.subTest(value=value),self.assertRaises(ValueError):
                config_from_dict({'degradation':{'poisson_training_peaks':value}})

    def test_trained_checkpoint_cli_evaluation_records_explicit_photon_level(self):
        from train import run_training
        root=Path(__file__).resolve().parents[1]
        cfg=config_from_dict({'model':{'width':4,'depth':3},'dataset':{'kind':'procedural','patch_size':16,'train_samples':2,'val_samples':2},'degradation':{'profile':'poisson_linear_rgb_v1'},'training':{'device':'cpu','batch_size':1,'accumulation_steps':1,'max_steps':2,'validation_interval':1,'checkpoint_interval':1}})
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/'run';run_training(cfg,run)
            output=Path(folder)/'eval.json'
            result=subprocess.run([sys.executable,'eval.py','--checkpoint',str(run/'best_model.pth'),'--device','cpu','--peak-photons','30','--output',str(output)],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(output.read_text())
            self.assertEqual(report['degradation']['peak_photons'],30)
            self.assertEqual(report['count'],2)
            invalid=subprocess.run([sys.executable,'eval.py','--checkpoint',str(run/'best_model.pth'),'--device','cpu','--peak-photons','nan','--output',str(Path(folder)/'invalid.json')],cwd=root,capture_output=True,text=True)
            self.assertNotEqual(invalid.returncode,0)
            self.assertFalse((Path(folder)/'invalid.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
