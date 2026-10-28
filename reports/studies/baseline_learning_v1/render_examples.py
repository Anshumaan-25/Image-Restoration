"""Render fixed first-two-development examples with float inference for both budgets."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/restoration_study_matplotlib")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import config_from_dict
from utils.dataset_loader import RestorationDataset
from utils.io import save_rgb
from utils.jpeg_phase import clean_context,phase_view
from utils.logger import write_json
from utils.metrics import evaluate_pair
from utils.runtime import select_device


@torch.inference_mode()
def main() -> None:
    """Save unsearched IDs/phase, common scales and explicit budget/schedule caveat."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    args=parser.parse_args();device=select_device(args.device);torch.set_num_threads(2)
    here=Path(__file__).resolve().parent
    checkpoints={'B_1000':ROOT/'experiments/studies/mps_screening_v1/B_seed42/best_model.pth',
                 'B_5000':ROOT/'experiments/studies/baseline_learning_v1/B_seed42/best_model.pth'}
    models={};identities={};data=None
    for name,path in checkpoints.items():
        state=load_checkpoint(path);cfg=config_from_dict(state['config'])
        model=build_model(cfg.model).to(device).eval();model.load_state_dict(state['model']);models[name]=model
        identities[name]={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected_step':state['step'],'training_horizon':state['total_steps']}
        data=RestorationDataset(cfg,'val')
    assert data is not None
    fig,axes=plt.subplots(4,4,figsize=(11,10),layout='constrained');records=[]
    for index in range(2):
        source,image_id=data.clean_source(index)
        context=clean_context(source,data.cfg.dataset.patch_size)
        image,target=phase_view(context,data.cfg.dataset.patch_size,(0,0),20)
        images={'target':target,'input':image,**{key:model(image[None].to(device)).float().cpu()[0].clamp(0,1) for key,model in models.items()}}
        for column,(name,tensor) in enumerate(images.items()):
            scores=evaluate_pair(tensor,target)
            axes[2*index,column].imshow(tensor.permute(1,2,0).numpy())
            axes[2*index,column].set_title(f'{image_id} {name}'+('' if name=='target' else f" / {scores['psnr']:.3f} dB"))
            axes[2*index+1,column].imshow((tensor-target).abs().mean(0).numpy(),cmap='inferno',vmin=0,vmax=.10)
            axes[2*index+1,column].set_title('Absolute RGB error; shared scale 0–0.10',fontsize=8)
            axes[2*index,column].axis('off');axes[2*index+1,column].axis('off')
            save_rgb(tensor,here/f'{image_id}_{name}.png')
            if name!='target':records.append({'id':image_id,'phase':[0,0],'model':name,**scores})
    fig.suptitle('First two development IDs, fixed center/phase (0,0); different budgets/cosine schedules, not a matched method test')
    fig.savefig(here/'qualitative_development.png',dpi=150);plt.close(fig)
    write_json(here/'qualitative_metadata.json',{'device':str(device),'split':'val','phase':[0,0],'selection':'first two development IDs without quality search','checkpoints':identities,'records':records,'display':'8-bit PNG; metrics and error maps from float RGB','error_scale':[0,.1],'protected_test_scored':False,'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__=='__main__':main()
