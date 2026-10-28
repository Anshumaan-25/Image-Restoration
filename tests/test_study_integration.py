"""Full phase-study train -> freeze -> held-out eval -> paired report on authored fixtures."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import torch
import yaml
from scripts.freeze_study import freeze
from train import run_training
from utils.config import load_config, config_from_dict
from utils.io import save_rgb, file_sha256
from utils.study import study_plan, compare_phase_reports


class StudyIntegrationTests(unittest.TestCase):
    """Tiny authored images verify mechanics; they are not natural-image experiments."""
    def test_study_freeze_evaluation_leakage_and_comparison(self) -> None:
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for index,split in enumerate(('train','val','test')):
                path=root/f'{split}.png'
                generator=torch.Generator().manual_seed(100+index)
                save_rgb(torch.rand(3,48,48,generator=generator),path)
                row={'id':split,'dataset':'authored_fixture','version':'1','split':split,
                     'source_id':split,'scene_id':None,'input_path':None,'target_path':path.name,
                     'sha256':file_sha256(path),'height':48,'width':48,'bit_depth':8,'color_space':'sRGB','scale':1}
                (root/f'{split}.jsonl').write_text(json.dumps(row)+'\n')
            paths={}
            for arm,mode,eta in [('A','random',0),('B','stratified',0),('C','stratified',.05)]:
                cfg=load_config('configs/smoke.yaml')
                cfg.training.device='cpu'
                cfg.training.batch_size=1
                cfg.training.accumulation_steps=1
                cfg.training.max_steps=1
                cfg.training.warmup_steps=0
                cfg.dataset.kind='manifest'
                cfg.dataset.root=str(root)
                cfg.dataset.train_manifest='train.jsonl'
                cfg.dataset.val_manifest='val.jsonl'
                cfg.dataset.patch_size=16
                cfg.degradation.profile='jpeg_rgb_444_v1'
                cfg.jpeg_study.mode=mode
                cfg.loss.consistency_weight=eta
                path=root/f'{arm}.yaml'
                path.write_text(yaml.safe_dump(cfg.to_dict()))
                paths[arm]=str(path)
            (root/'study.yaml').write_text(yaml.safe_dump({'name':'authored_fixture','arms':paths,'seeds':[42],'max_seconds_per_run':30}))
            plan=study_plan(root/'study.yaml','cpu')
            (root/'plan.json').write_text(json.dumps(plan))
            checkpoints=[]
            for run in plan['runs']:
                run_training(config_from_dict(run['config']),root/run['id'])
                checkpoints.append(str(root/run['id']/'best_model.pth'))
            frozen=root/'frozen.json'
            record=freeze(str(root/'plan.json'),checkpoints,'test.jsonl',str(root),str(frozen))
            self.assertEqual(len(record['selected_checkpoints']),3)
            self.assertFalse(record['test_scored'])
            def evaluate(checkpoint,manifest,split,output,*extra):
                return subprocess.run([sys.executable,'eval.py','--checkpoint',checkpoint,'--data-dir',str(root),
                    '--manifest',manifest,'--split',split,'--device','cpu','--output',str(output),*extra],
                    capture_output=True,text=True,timeout=60)
            with self.assertRaises(FileExistsError):
                freeze(str(root/'plan.json'),checkpoints,'test.jsonl',str(root),str(frozen))
            # A train image relabeled as test must fail even with a separately named manifest.
            leaked=json.loads((root/'train.jsonl').read_text())
            leaked['split']='test'
            (root/'leak.jsonl').write_text(json.dumps(leaked)+'\n')
            result=evaluate(checkpoints[0],'leak.jsonl','test',root/'bad.json','--phase-panel','all','--frozen-study',str(frozen))
            self.assertNotEqual(result.returncode,0)
            self.assertIn('leakage',result.stderr)
            result=evaluate(checkpoints[0],'test.jsonl','test',root/'unfrozen.json','--phase-panel','all')
            self.assertNotEqual(result.returncode,0)
            reports=[]
            for index in (0,2):
                output=root/f'eval{index}.json'
                result=evaluate(checkpoints[index],'test.jsonl','test',output,'--phase-panel','all','--frozen-study',str(frozen))
                self.assertEqual(result.returncode,0,result.stderr)
                report=json.loads(output.read_text())
                self.assertEqual((report['count'],report['phase_count']),(1,64))
                self.assertEqual(report['reference_crop'],'center_P_plus_32_then_center_P')
                reports.append(report)
            comparison=compare_phase_reports([reports[0]],[reports[1]],repetitions=100)
            self.assertEqual(comparison['image_count'],1)
            self.assertEqual(comparison['seeds'],[42])
            result=evaluate(checkpoints[2],'test.jsonl','test',root/'full_image.json','--full-image','--frozen-study',str(frozen))
            self.assertEqual(result.returncode,0,result.stderr)
            full=json.loads((root/'full_image.json').read_text())
            self.assertEqual(full['reference_crop'],'full_image')
            self.assertNotIn('phase_count',full)
            result=evaluate(checkpoints[2],'val.jsonl','val',root/'q10.json','--jpeg-quality','10')
            self.assertEqual(result.returncode,0,result.stderr)
            mismatch=json.loads((root/'q10.json').read_text())
            self.assertEqual(mismatch['quality'],10)
            self.assertEqual(mismatch['degradation']['jpeg_quality'],10)
