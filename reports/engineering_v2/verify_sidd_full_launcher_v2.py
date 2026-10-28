"""Offline completion/timeout/failure contracts for the full-validation launcher."""
from pathlib import Path
import hashlib,importlib.util,json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
p=Path(__file__).with_name('colab_sidd_full_validation_v2.py');spec=importlib.util.spec_from_file_location('sidd_launcher',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ScoringContractTests(unittest.TestCase):
    """Use authored scalar rows, never pretend these are actual CUDA measurements."""
    def row(self,index: int) -> dict:
        """Produce one finite synthetic scoring record for loop tests."""
        return {'id':str(index),'psnr':20.+index,'ssim':.5,'reference':{'psnr':10.,'ssim':.2}}
    def test_complete_exact_count(self) -> None:
        """All requested rows and their per-image means must be retained."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);module.score_all(self.row,3,root,10)
            r=json.loads((root/'results.json').read_text());self.assertEqual((r['status'],r['count'],r['psnr']),('complete',3,21.))
    def test_midrun_failure_keeps_partial(self) -> None:
        """A failed image cannot be dropped into a smaller successful report."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def fail(index: int) -> dict:
                if index==1:raise ValueError('authored failure')
                return self.row(index)
            with self.assertRaises(ValueError):module.score_all(fail,3,root,10)
            r=json.loads((root/'progress.json').read_text());self.assertEqual((r['status'],r['count']),('incomplete',1));self.assertFalse((root/'results.json').exists())
    def test_timeout_never_marks_complete(self) -> None:
        """An exhausted budget must fail before scoring another sample."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaises(TimeoutError):module.score_all(self.row,3,root,0)
            r=json.loads((root/'progress.json').read_text());self.assertEqual((r['status'],r['count']),('incomplete',0));self.assertFalse((root/'results.json').exists())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScoringContractTests))
    evidence={'status':'passed' if result.wasSuccessful() else 'failed','test_count':result.testsRun,'scope':'Offline authored loop tests only, not actual CUDA/SIDD full validation','launcher_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    p.with_name('sidd_full_launcher_v2_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
