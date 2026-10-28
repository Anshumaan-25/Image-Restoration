"""Study matching, leakage, protocol guards and image-level bootstrap contracts."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
import yaml
from utils.study import study_plan, compare_phase_reports
from utils.logger import write_json
from utils.config import load_config
from utils.dataset_loader import validate_splits
from utils.checkpoint import load_checkpoint
from train import run_training


class StudyTests(unittest.TestCase):
    """Reject misleading comparisons before a scientific conclusion is emitted."""
    def test_plan_matching_and_no_execution(self) -> None:
        plan=study_plan('configs/jpeg/study.yaml','mps',2,4,'fp32')
        self.assertEqual(len(plan['runs']),3)
        self.assertEqual(plan['total_optimizer_updates'],3000)
        self.assertFalse(plan['protected_test_used'])
        a,b,c=[r['config'] for r in plan['runs']]
        self.assertEqual(a['training'],b['training'])
        self.assertEqual(b['training'],c['training'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            altered=yaml.safe_load(Path('configs/jpeg/stratified_phase.yaml').read_text())
            altered['training']['batch_size']=3
            (root/'bad.yaml').write_text(yaml.safe_dump(altered))
            study=yaml.safe_load(Path('configs/jpeg/study.yaml').read_text())
            study['arms']['B']=str(root/'bad.yaml')
            (root/'study.yaml').write_text(yaml.safe_dump(study))
            with self.assertRaisesRegex(ValueError,'differ'):
                study_plan(root/'study.yaml','cpu',precision='fp32')

    def test_grouped_bootstrap_and_protocol_checks(self) -> None:
        def report(seed,offset):
            return {'training_seed':seed,'dataset_hash':'same','split':'val','protocol':'project_rgb_float_v1',
                    'endpoint':'mean_per_image_worst_phase_psnr','phase_panel':[[0,0],[2,2]],
                    'quality':20,'patch_size':128,'count':2,'codec':{'jpeg':'fixed'},
                    'model':{'name':'fixture'},'procedural_test_only':True,'frozen_study_sha256':None,
                    'phase_constructor':'jpeg_aligned_phase_v1','reference_crop':'center_P_plus_32_then_center_P',
                    'training_config':{'max_steps':1000},'selection_horizon':1000,
                    'training_source_hash':'fixture_source',
                    'images':[{'id':'a','worst_phase_psnr':20+offset},{'id':'b','worst_phase_psnr':25+offset}]}
        baseline=[report(42,0),report(43,1)]
        candidate=[report(42,2),report(43,3)]
        result=compare_phase_reports(baseline,candidate,repetitions=100)
        self.assertEqual(result['delta_db'],2)
        self.assertEqual(result['image_bootstrap_ci95_db'],[2,2])
        self.assertEqual(result['image_count'],2)
        candidate[0]['quality']=10
        with self.assertRaisesRegex(ValueError,'quality'):
            compare_phase_reports(baseline,candidate)
        candidate=deepcopy(baseline)
        candidate[0]['images'][0]['id']='wrong'
        with self.assertRaisesRegex(ValueError,'identit'):
            compare_phase_reports(baseline,candidate)

    def test_nested_infinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json'
            write_json(path,{'psnr':float('inf'),'worst_phase_psnr':float('inf'),
                             'images':[{'psnr':float('inf'),'ssim':1.0}]})
            record=json.loads(path.read_text())
            self.assertIsNone(record['images'][0]['psnr'])
            self.assertTrue(record['worst_phase_psnr_is_infinite'])
            with self.assertRaises(ValueError):
                write_json(path,{'images':[{'ssim':float('nan')}]})

    def test_profile_incomplete_report_and_no_model_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp)/'profile.json'
            result=subprocess.run([sys.executable,'scripts/profile.py','--config','configs/smoke.yaml',
                 '--device','cpu','--warmup','100','--steps','1','--max-seconds','0.00000001','--output',str(report)],
                 capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,2,result.stderr)
            value=json.loads(report.read_text())
            self.assertFalse(value['completed'])
            self.assertEqual(value['measured_updates'],0)
            self.assertIsNone(value['median_update_seconds'])
            self.assertFalse(value['validation_executed'])

    def test_external_evaluation_source_leakage(self) -> None:
        identity={'dataset':'test','id':'one','source_id':'one','sha256':'aaa','split':'train'}
        with self.assertRaisesRegex(ValueError,'leakage'):
            validate_splits([identity,{**identity,'split':'test'}])
        with self.assertRaisesRegex(ValueError,'leakage'):
            validate_splits([identity,{**identity,'id':'renamed','source_id':'other','split':'test'}])

    def test_latency_cli_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'latency.json'
            result=subprocess.run([sys.executable,'scripts/benchmark.py','--config','configs/smoke.yaml',
                 '--device','cpu','--sizes','12','--warmup','1','--iterations','2','--output',str(output)],
                 capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(output.read_text())
            self.assertEqual(report['results'][0]['measured_passes'],2)
            self.assertTrue(report['random_weights'])
            self.assertFalse(report['restoration_accuracy_evaluated'])
