"""Explicit paired MAT layout/range/import correctness with authored fixtures only."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
import numpy as np
from utils.sidd_blocks import prepare_sidd_blocks
from utils.io import load_rgb
from utils.config import config_from_dict
from utils.dataset_loader import RestorationDataset


class SiddBlocksTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('scipy'),'Optional scipy unavailable')
    def test_mat_v5_pairs_preserve_pixels_parent_identity_and_bypass_synthesis(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rng=np.random.default_rng(9)
            clean=rng.integers(0,256,(2,2,13,15,3),dtype=np.uint8)
            noisy=255-clean
            savemat(root/'noisy.mat',{'ValidationNoisyBlocksSrgb':noisy})
            savemat(root/'clean.mat',{'ValidationGtBlocksSrgb':clean})
            out=root/'pairs';result=prepare_sidd_blocks(root/'noisy.mat',root/'clean.mat',out,'nkhwc')
            self.assertEqual(result['count'],4)
            cfg=config_from_dict({'dataset':{'root':str(out),'val_manifest':'val.jsonl'},'degradation':{'profile':'sidd_srgb_pairs_v1'}})
            data=RestorationDataset(cfg,'val')
            self.assertEqual(data.records[0]['source_id'],data.records[1]['source_id'])
            self.assertNotEqual(data.records[0]['source_id'],data.records[2]['source_id'])
            sample=data[0]
            np.testing.assert_allclose(sample['input'].permute(1,2,0).numpy(),noisy[0,0]/255,atol=1e-7)
            np.testing.assert_allclose(sample['target'].permute(1,2,0).numpy(),clean[0,0]/255,atol=1e-7)
            with self.assertRaises(FileExistsError):prepare_sidd_blocks(root/'noisy.mat',root/'clean.mat',out,'nkhwc')

    @unittest.skipUnless(importlib.util.find_spec('h5py'),'Optional h5py unavailable')
    def test_hdf5_reverse_layout_matches_declared_rgb_pixels(self):
        import h5py
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pixels=np.arange(1*2*11*13*3,dtype=np.uint8).reshape(1,2,11,13,3)
            for name,key,value in [('noisy','ValidationNoisyBlocksSrgb',pixels),('clean','ValidationGtBlocksSrgb',255-pixels)]:
                with h5py.File(root/(name+'.mat'),'w') as handle:handle[key]=value.transpose(4,3,2,1,0)
            out=root/'pairs';prepare_sidd_blocks(root/'noisy.mat',root/'clean.mat',out,'matlab_hdf5_reversed')
            actual=load_rgb(out/'noisy/sidd_val_000_001.png').permute(1,2,0).numpy()
            np.testing.assert_allclose(actual,pixels[0,1]/255,atol=1e-7)

    @unittest.skipUnless(importlib.util.find_spec('scipy'),'Optional scipy unavailable')
    def test_rejects_ambiguous_range_missing_key_bad_layout_and_geometry(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for dtype in (np.float32,np.uint16):
                savemat(root/'n.mat',{'ValidationNoisyBlocksSrgb':np.ones((1,1,11,11,3),dtype=dtype)})
                savemat(root/'c.mat',{'ValidationGtBlocksSrgb':np.ones((1,1,11,11,3),dtype=dtype)})
                with self.assertRaisesRegex(ValueError,'uint8'):prepare_sidd_blocks(root/'n.mat',root/'c.mat',root/'out','nkhwc')
            savemat(root/'n.mat',{'wrong':np.ones((1,1,11,11,3),dtype=np.uint8)})
            with self.assertRaisesRegex(ValueError,'Missing named'):prepare_sidd_blocks(root/'n.mat',root/'c.mat',root/'out','nkhwc')
            with self.assertRaisesRegex(ValueError,'Layout'):prepare_sidd_blocks(root/'n.mat',root/'c.mat',root/'out','guess')
            savemat(root/'n.mat',{'ValidationNoisyBlocksSrgb':np.ones((1,1,11,11,3),dtype=np.uint8)})
            savemat(root/'c.mat',{'ValidationGtBlocksSrgb':np.ones((1,2,11,11,3),dtype=np.uint8)})
            with self.assertRaisesRegex(ValueError,'geometry differs'):prepare_sidd_blocks(root/'n.mat',root/'c.mat',root/'out','nkhwc')


if __name__=='__main__':unittest.main(verbosity=2)
