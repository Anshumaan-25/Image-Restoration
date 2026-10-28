"""Bounded decoded caching must preserve samples, RNG and caller ownership."""
from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from utils.config import config_from_dict
from utils.dataset_loader import RestorationDataset
from utils.io import file_sha256,load_rgb


class DecodedCacheTests(unittest.TestCase):
    """Cache policy is an I/O optimization, not a different data experiment."""

    def fixture(self, root: Path, size: int = 24) -> Path:
        """Write three independent clean images and an explicit train manifest."""
        records=[]
        for index in range(3):
            path=root/f'{index}.png'
            Image.fromarray(np.random.default_rng(index).integers(0,256,(size,size,3),dtype=np.uint8)).save(path)
            records.append({'id':str(index),'dataset':'cache_fixture','version':'1','source_id':str(index),'split':'train','target_path':path.name,'input_path':None,'sha256':file_sha256(path),'height':size,'width':size,'bit_depth':8,'color_space':'sRGB','scale':1})
        manifest=root/'train.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
        return manifest

    def dataset(self, root: Path, cache: int, profile: str = 'poisson_linear_rgb_v1') -> RestorationDataset:
        """Create a small manifest-backed training dataset with one cache policy."""
        cfg=config_from_dict({'dataset':{'root':str(root),'train_manifest':'train.jsonl','patch_size':16,'cache_max_mib':cache},'degradation':{'profile':profile},'training':{'num_workers':0}})
        return RestorationDataset(cfg,'train',training=True)

    def test_bitwise_samples_rng_and_cache_hits(self) -> None:
        """Exact inputs/targets/metadata survive cold/hit paths for all synthetic tasks."""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            for profile in ['poisson_linear_rgb_v1','awgn_unclipped_v1','jpeg_rgb_444_v1']:
                baseline=self.dataset(root,0,profile);cached=self.dataset(root,1,profile)
                rng=torch.get_rng_state().clone()
                with patch('utils.dataset_loader.load_rgb',wraps=load_rgb) as decoder:
                    for ordinal in range(12):
                        actual=cached[ordinal];expected=baseline[ordinal]
                        self.assertTrue(torch.equal(actual['input'],expected['input']))
                        self.assertTrue(torch.equal(actual['target'],expected['target']))
                        self.assertEqual(actual['metadata'],expected['metadata'])
                    self.assertEqual(decoder.call_count,15) # 12 uncached + 3 cold reads
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertLessEqual(cached._cache_bytes,1024**2)

    def test_byte_bound_eviction_oversized_bypass_and_ownership(self) -> None:
        """LRU is bounded; a mutated return cannot poison future training samples."""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root,256);dataset=self.dataset(root,1)
            expected=load_rgb(root/'0.png');returned,_=dataset.clean_source(0);returned.zero_()
            actual,_=dataset.clean_source(0);self.assertTrue(torch.equal(expected,actual))
            dataset.clean_source(1)
            self.assertEqual(list(dataset._clean_cache),[1]);self.assertLessEqual(dataset._cache_bytes,1024**2)
            self.fixture(root,400);large=self.dataset(root,1);large.clean_source(0)
            self.assertEqual(large._cache_bytes,0);self.assertFalse(large._clean_cache)

    def test_invalid_or_multiplied_memory_budget_rejected(self) -> None:
        """Reject ambiguous byte budgets and worker copies before starting a run."""
        for value in [-1,True,1.5,float('nan')]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                config_from_dict({'dataset':{'cache_max_mib':value}})
        with self.assertRaises(ValueError):
            config_from_dict({'dataset':{'cache_max_mib':1},'training':{'num_workers':1}})


if __name__=='__main__':unittest.main()
