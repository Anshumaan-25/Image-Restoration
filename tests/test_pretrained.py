"""Offline pretrained integrity/provenance guards; authored tensor fixtures only."""
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from models import DnCNNRGB
from utils.pretrained import load_pretrained,extract_parameters,PRETRAINED

ROOT=Path(__file__).resolve().parents[1]


class PretrainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_tensor_mapping_is_explicit_finite_and_no_ema_fallback(self):
        params={'weight':torch.ones(2)}
        self.assertIs(extract_parameters({'params':params},'params'),params)
        self.assertIs(extract_parameters(params,'params',True),params)
        for payload in [{'params_ema':params},{'params':{}},{'params':{'w':torch.tensor(float('nan'))}},{'params':{'w':'not tensor'}}]:
            with self.assertRaises(ValueError):extract_parameters(payload,'params')

    def test_strict_import_receipt_and_integrity_before_tensor_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'authored_fixture.pth'
            model=DnCNNRGB();torch.save({'params':model.state_dict()},path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            entry={**PRETRAINED['restormer_gaussian'],'model':'dncnn_rgb_l1','bytes':path.stat().st_size}
            with patch.dict(PRETRAINED,{'fixture':entry}):
                loaded,cfg,receipt=load_pretrained('fixture',path,digest)
                self.assertFalse(loaded.training)
                self.assertIsNone(receipt['training_seed'])
                self.assertFalse(receipt['project_training_performed'])
                self.assertEqual(receipt['sha256'],digest)
                with patch('utils.pretrained.torch.load') as load:
                    with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):load_pretrained('fixture',path,'0'*64)
                    load.assert_not_called()
                with self.assertRaisesRegex(ValueError,'Explicit lowercase'):load_pretrained('fixture',path,None)
                entry['bytes']+=1
                with self.assertRaisesRegex(ValueError,'byte size'):load_pretrained('fixture',path,digest)
            with self.assertRaisesRegex(ValueError,'Unapproved'):load_pretrained('GAN_variant',path,digest)

    def test_acquisition_and_evaluation_require_explicit_authorization(self):
        commands=[['scripts/acquire_pretrained.py','--ids','restormer_gaussian','--output','unused'],
                  ['scripts/evaluate_pretrained.py','--id','restormer_gaussian','--weights','unused','--weights-sha256','0'*64,'--data-dir','.','--manifest','unused','--device','cpu','--output','unused']]
        for command in commands:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Specific',result.stderr)
            self.assertIn('approval',result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
