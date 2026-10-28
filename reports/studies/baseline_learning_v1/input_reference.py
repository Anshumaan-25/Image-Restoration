"""Recompute identified full-image JPEG input scores, development sources only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import os

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import torch
from utils.checkpoint import load_checkpoint
from utils.config import config_from_dict
from utils.dataset_loader import RestorationDataset
from utils.degradation import codec_metadata
from utils.logger import write_json
from utils.metrics import aggregate,evaluate_pair
from utils.reproducibility import source_identity


def main() -> None:
    """Retain raw per-image scores and immutable dataset/operator/source identities."""
    torch.set_num_threads(2)
    here=Path(__file__).resolve().parent
    output=here/'jpeg_input_full_image_development.json'
    if output.exists():raise FileExistsError(output)
    state=load_checkpoint(ROOT/'experiments/studies/baseline_learning_v1/B_seed42/best_model.pth')
    cfg=config_from_dict(state['config']);cfg.jpeg_study.mode='off'
    assert cfg.degradation.jpeg_quality==20 and cfg.loss.consistency_weight==0
    data=RestorationDataset(cfg,'val')
    rows=[]
    for index in range(len(data)):
        sample=data[index]
        rows.append({'id':sample['id'],**evaluate_pair(sample['input'],sample['target']),
                     'height':sample['target'].shape[-2],'width':sample['target'].shape[-1],
                     'target_sha256':data.records[index]['sha256']})
    report={**aggregate(rows),'images':rows,'split':'val','endpoint':'full_image_jpeg_input',
            'dataset_hash':data.fingerprint,'degradation':cfg.degradation.__dict__,
            'protocol':'project_rgb_float_v1','ssim_protocol':'project_ssim_v1',
            'codec':codec_metadata(),'trained':False,'protected_test_scored':False,
            'source_files':source_identity()['files'],
            'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('images','source_files')},indent=2))


if __name__=='__main__':main()
