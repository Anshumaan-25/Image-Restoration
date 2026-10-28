"""Reproduce this study's plots and diagnostics without modifying frozen pipeline code."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
os.environ['MPLCONFIGDIR']='/private/tmp/restoration_study_matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import config_from_dict
from utils.dataset_loader import RestorationDataset
from utils.io import save_rgb
from utils.jpeg_phase import DEV_PHASES,clean_context,phase_view
from utils.metrics import psnr
from utils.runtime import select_device

HERE=Path(__file__).resolve().parent
STUDY=ROOT/'experiments/studies/mps_screening_v1'
COLORS={'A':'#4477AA','B':'#228833','C':'#CC6677'}


def curves() -> None:
    """Plot only observed fixed-development measurements and JPEG references."""
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for arm in 'ABC':
        with (STUDY/f'{arm}_seed42/metrics.csv').open() as handle:
            rows=[r for r in csv.DictReader(handle) if r['worst_phase_psnr']]
        steps=[int(r['step']) for r in rows]
        for axis,metric in zip(axes,('worst_phase_psnr','ssim')):
            axis.plot(steps,[float(r[metric]) for r in rows],marker='o',markersize=3,color=COLORS[arm],label=arm)
        best=json.loads((HERE/f'{arm}_dev.json').read_text())
        axes[0].scatter([best['step']],[best['worst_phase_psnr']],marker='X',s=80,color=COLORS[arm],zorder=4)
    base=json.loads((HERE/'A_dev.json').read_text())
    for axis,value in zip(axes,(base['input_worst_phase_psnr'],base['input_mean_ssim'])):
        axis.axhline(value,color='#777777',linestyle='--',label='JPEG input')
        axis.set_xlabel('Successful optimizer updates')
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel('Mean per-image worst-phase PSNR (dB)')
    axes[1].set_ylabel('Mean RGB SSIM')
    fig.suptitle('Development screening: 8 images, 16 phases, one seed; X = selected checkpoint')
    fig.savefig(HERE/'learning_curves.png',dpi=180)
    fig.savefig(HERE/'learning_curves.svg')
    plt.close(fig)


@torch.inference_mode()
def diagnostics(device_name: str) -> None:
    """Inspect residuals and a separately labeled 20-pixel interior diagnostic."""
    torch.set_num_threads(2)
    device=select_device(device_name)
    record={'device':str(device),'split':'development','phase_count':16,'image_count':8,
            'primary_border':0,'additional_diagnostic_border':20,
            'diagnostic_note':'20-pixel interior crop checks the radius-20 inference receptive field of 20 stride-one 3x3 convolutions; it does not replace the primary endpoint.',
            'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'models':{}}
    examples={}
    for arm in 'ABC':
        state=load_checkpoint(STUDY/f'{arm}_seed42/best_model.pth')
        cfg=config_from_dict(state['config'])
        dataset=RestorationDataset(cfg,'val')
        model=build_model(cfg.model).to(device).eval()
        model.load_state_dict(state['model'],strict=True)
        torch.manual_seed(cfg.training.seed)
        initial=build_model(cfg.model)
        squared_delta=sum(float((state['model'][name]-parameter).double().square().sum()) for name,parameter in initial.named_parameters())
        rows=[]
        for index in range(len(dataset)):
            source,image_id=dataset.clean_source(index)
            context=clean_context(source,cfg.dataset.patch_size)
            input_interior=[];output_interior=[]
            for phase in DEV_PHASES:
                degraded,target=phase_view(context,cfg.dataset.patch_size,phase,cfg.degradation.jpeg_quality)
                prediction=model(degraded[None].to(device)).float().cpu()[0].clamp(0,1)
                input_interior.append(psnr(degraded,target,border=20))
                output_interior.append(psnr(prediction,target,border=20))
                rows.append({'input_mae':float((degraded-target).abs().mean()),
                             'restored_mae':float((prediction-target).abs().mean()),
                             'predicted_residual_rms':float((degraded-prediction).square().mean().sqrt()),
                             'target_residual_rms':float((degraded-target).square().mean().sqrt())})
                if index < 2 and phase==(0,0):
                    entry=examples.setdefault(image_id,{'target':target,'input':degraded})
                    entry[arm]=prediction
            rows.append({'input_interior_worst_psnr':min(input_interior),'output_interior_worst_psnr':min(output_interior)})
        fields=('input_mae','restored_mae','predicted_residual_rms','target_residual_rms',
                'input_interior_worst_psnr','output_interior_worst_psnr')
        record['models'][arm]={key:float(np.mean([r[key] for r in rows if key in r])) for key in fields}
        record['models'][arm]['parameter_l2_change_from_seed_initialization']=squared_delta**.5
        del model
    # First two validation IDs, phase (0,0), selected before viewing outputs.
    fig,axes=plt.subplots(4,5,figsize=(13,10),layout='constrained')
    labels=('target','input','A','B','C')
    for row,(image_id,images) in enumerate(examples.items()):
        target=images['target']
        for column,key in enumerate(labels):
            image=images[key]
            axes[2*row,column].imshow(image.permute(1,2,0).numpy())
            title='Clean' if key=='target' else ('JPEG' if key=='input' else key)
            score=psnr(image,target)
            axes[2*row,column].set_title(f'{image_id} {title}'+('' if key=='target' else f' / {score:.3f} dB'))
            error=(image-target).abs().mean(dim=0)
            axes[2*row+1,column].imshow(error.numpy(),cmap='inferno',vmin=0,vmax=.10)
            axes[2*row+1,column].set_title('Mean absolute RGB error; fixed scale 0–0.10',fontsize=8)
            axes[2*row,column].axis('off');axes[2*row+1,column].axis('off')
            save_rgb(image,HERE/f'{image_id}_{key}.png')
    fig.suptitle('Development examples: first two IDs, fixed center crop and phase (0,0); display is 8-bit, scores use float RGB')
    fig.savefig(HERE/'qualitative_dev.png',dpi=150)
    plt.close(fig)
    (HERE/'diagnostics.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


def main() -> None:
    """Analysis-only outputs live with this immutable study's evidence."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['curves','diagnostics','all'],default='all')
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    args=parser.parse_args()
    if args.mode in ('curves','all'):curves()
    if args.mode in ('diagnostics','all'):diagnostics(args.device)


if __name__=='__main__':main()
