"""Offline completion-workflow checks; procedural CPU fixtures, no protected scoring."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import ast
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
from models import DnCNNRGB
from train import validate
from utils.metrics import aggregate,evaluate_pair
from PIL import Image

SCRIPT=Path(__file__).with_name('colab_completion_checks.py')
spec=importlib.util.spec_from_file_location('completion_checks',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Fixtures:
    """Deterministic odd-sized image pairs, compatible with ordinary validation."""
    def __init__(self):
        from utils.config import Config
        self.cfg=Config()
        generator=torch.Generator().manual_seed(8)
        self.items=[]
        for index,(h,w) in enumerate([(17,19),(21,23)]):
            target=torch.rand(3,h,w,generator=generator)
            self.items.append({'id':f'{801+index:04d}','input':target*.8,'target':target})
    def __len__(self):return len(self.items)
    def __getitem__(self,index):return self.items[index]


class CompletionChecksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_full_image_metrics_match_existing_validation_and_save_progress(self):
        torch.manual_seed(3)
        model=DnCNNRGB(4,3).eval();dataset=Fixtures()
        expected=validate(model,dataset,torch.device('cpu'))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'A'
            result=module.evaluate_full(model,dataset,torch.device('cpu'),output,{'0801'})
            self.assertEqual(result['count'],2)
            self.assertEqual(result['reference_crop'],'full_image')
            for metric in ('psnr','ssim'):self.assertAlmostEqual(expected[metric],result[metric],places=12)
            self.assertEqual(json.loads((output/'progress.json').read_text())['count'],2)
            self.assertEqual(len(list(output.glob('*.png'))),6)
            with Image.open(output/'0801_restored.png') as image:self.assertEqual(image.size,(19,17))
            baseline=aggregate([evaluate_pair(item['input'],item['target']) for item in dataset.items])
            self.assertAlmostEqual(result['input_psnr'],baseline['psnr'],places=12)
            with self.assertRaises(FileExistsError):
                module.evaluate_full(model,dataset,torch.device('cpu'),output,set())
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_nonfinite_and_shape_failure_never_produces_success_report(self):
        class Bad(torch.nn.Module):
            def forward(self,x):return torch.full_like(x,float('nan'))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'Invalid full-image'):
                module.evaluate_full(Bad(),Fixtures(),torch.device('cpu'),Path(directory)/'bad',set())
            self.assertFalse((Path(directory)/'bad/progress.json').exists())

    def test_fixed_panels_use_identical_coordinates_and_preserve_source_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for arm in 'ABC':
                (root/arm).mkdir()
                for image_id in ('0801','0802','0803'):
                    for kind in ('clean','compressed','restored'):
                        Image.new('RGB',(16,16),{'clean':'white','compressed':'gray','restored':'blue'}[kind]).save(root/arm/f'{image_id}_{kind}_center.png')
            module.make_panels(root)
            with Image.open(root/'0801_comparison.png') as image:
                self.assertEqual(image.size,(80,48))
                self.assertEqual(image.getpixel((1,33)),(255,255,255))
                self.assertEqual(image.getpixel((33,33)),(0,0,255))

    def test_authorization_precedes_drive_or_cuda_access(self):
        result=subprocess.run([sys.executable,str(SCRIPT)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Explicit completion-check authorization required',result.stderr)

    def test_perfect_psnr_is_strict_json_and_other_nonfinite_values_fail(self):
        value=module.strict_metrics({'psnr':float('inf'),'images':[{'input_psnr':float('inf')} ]})
        self.assertIsNone(value['psnr'])
        self.assertTrue(value['psnr_is_infinite'])
        self.assertTrue(value['images'][0]['input_psnr_is_infinite'])
        json.dumps(value,allow_nan=False)
        for bad in ({'ssim':float('nan')},{'psnr':-float('inf')}):
            with self.assertRaises(ValueError):module.strict_metrics(bad)

    def test_actual_worker_allocator_call_uses_explicit_cuda_index(self):
        tree=ast.parse(SCRIPT.read_text())
        worker=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='worker')
        call=next(node for node in ast.walk(worker) if isinstance(node,ast.Expr)
                  and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                  and node.value.func.attr=='set_per_process_memory_fraction')
        code=compile(ast.Module(body=[call],type_ignores=[]),str(SCRIPT),'exec')
        for label,expected in [('cuda',0),('cuda:1',1)]:
            with self.subTest(device=label),patch.object(torch.cuda,'current_device',return_value=0) as current,patch.object(torch.cuda,'set_per_process_memory_fraction') as allocator:
                exec(code,{'torch':torch,'device':torch.device(label),'total':15*1024**3})
                allocator.assert_called_once_with(.8,expected)
                self.assertIs(type(allocator.call_args.args[1]),int)
                self.assertEqual(current.call_count,1 if label=='cuda' else 0)


if __name__=='__main__':unittest.main(verbosity=2)
