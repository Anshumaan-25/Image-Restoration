"""Offline parity and boundary checks for the bounded SIDD evaluation helper."""
from pathlib import Path
import sys,time,json,unittest
sys.path[:0]=[str(Path.cwd()),str(Path(__file__).parent)]
import torch
from utils.metrics import evaluate_pair
from sidd_fast_metrics import evaluate_pair_fast

class FastMetricTests(unittest.TestCase):
    def test_reference_parity(self):
        torch.manual_seed(82)
        cases=[]
        for h,w in [(11,11),(13,17),(64,64),(256,256)]:
            x=torch.rand(3,h,w);cases.extend([(x,x,0),(x+.07*torch.randn_like(x),x,0),(1-x,x,0)])
        for v in [0.,.5,1.]:
            x=torch.full((3,31,33),v);cases.extend([(x,x,0),(x+1e-5,x,0)])
        x=torch.rand(3,41,43);cases.append((x+.1,x,3))
        largest=0.
        for pred,target,border in cases:
            a=evaluate_pair(pred,target,border);b=evaluate_pair_fast(pred,target,border)
            self.assertEqual(a['psnr'],b['psnr']);error=abs(a['ssim']-b['ssim']);largest=max(largest,error);self.assertLess(error,1e-10)
        print('Maximum SSIM absolute difference:',largest)
    def test_invalid_contracts(self):
        x=torch.zeros(3,11,11)
        for a,b,border in [(x[:,:10],x[:,:10],0),(x.float()+float('nan'),x,0),(x,x+2,0),(x,x,-1),(x[0],x[0],0)]:
            with self.assertRaises(ValueError):evaluate_pair_fast(a,b,border)

if __name__=='__main__':
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FastMetricTests))
    x=torch.rand(3,256,256);y=torch.rand_like(x);timings={}
    for name,fn in [('reference',evaluate_pair),('separable',evaluate_pair_fast)]:
        start=time.perf_counter()
        for _ in range(10):fn(x,y)
        timings[name]=(time.perf_counter()-start)/10
    Path('reports/engineering_v2/sidd_fast_metrics_verification.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'timing_seconds_256_local_cpu':timings,'tolerance':1e-10,'cloud_profile':'not yet performed'},indent=2)+'\n')
    print(timings)
    raise SystemExit(0 if result.wasSuccessful() else 1)
