"""Bounded explicit-device pretrained evaluation, separate from project-trained results."""
from pathlib import Path
import argparse
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from utils.pretrained import load_pretrained
from utils.config import config_from_dict
from utils.dataset_loader import RestorationDataset
from utils.metrics import evaluate_pair,aggregate
from utils.logger import write_json
from utils.runtime import select_device,environment
from utils.reproducibility import source_identity


def main() -> None:
    """Score every supplied pair/source with one frozen operator; fail on incomplete runs."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved',action='store_true')
    parser.add_argument('--id',required=True)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--weights-sha256',required=True)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--device',required=True,choices=['cpu','mps','cuda'])
    parser.add_argument('--sigma',type=float)
    parser.add_argument('--crop-size',type=int,help='Optional fixed central target-size crop after the declared input operator; separately labeled')
    parser.add_argument('--max-seconds',type=float,default=300.)
    parser.add_argument('--output',required=True)
    parser.add_argument('--protocol',choices=['project_rgb_float_v1','swinir_sr_replay_v1'],default='project_rgb_float_v1')
    args=parser.parse_args()
    if not args.approved:raise PermissionError('Specific bounded pretrained evaluation approval required')
    import math
    if not math.isfinite(args.max_seconds) or args.max_seconds<=0:parser.error('Positive finite wall-time cap required')
    if args.sigma is not None and (not math.isfinite(args.sigma) or not 10<=args.sigma<=50):parser.error('AWGN sigma must be in [10,50] (8-bit units)')
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2)
    model,cfg,receipt=load_pretrained(args.id,args.weights,args.weights_sha256)
    if args.protocol=='swinir_sr_replay_v1' and cfg.model.name!='swinir_light_sr':parser.error('SwinIR SR replay requires the scale-specific lightweight model')
    if args.sigma is not None and cfg.degradation.profile!='awgn_unclipped_v1':parser.error('--sigma applies only to the Gaussian denoiser')
    if args.crop_size is not None and (args.crop_size<11 or args.crop_size%cfg.model.scale):parser.error('Crop target size must be >=11 and divisible by scale')
    cfg.dataset.kind='manifest';cfg.dataset.root=args.data_dir;cfg.dataset.val_manifest=args.manifest
    if args.sigma is not None:cfg.degradation.sigma=args.sigma
    cfg=config_from_dict(cfg.to_dict())
    dataset=RestorationDataset(cfg,args.split)
    if args.id in {'restormer_gaussian','swinir_x2','swinir_x4'}:
        # Known DIV2K official train membership must not be passed off as held out.
        for row in dataset.records:
            if row['dataset'].upper().startswith('DIV2K') and row['id'].isdigit() and 1<=int(row['id'])<=800:
                raise ValueError('Known official-pretraining DIV2K train source cannot be labeled held-out evaluation')
    device=select_device(args.device);model=model.to(device).eval()
    from models.baselines import Bicubic
    reference=Bicubic(cfg.model.scale) if cfg.model.scale>1 else None
    scores=[];baselines=[];rows=[];started=time.monotonic()
    with torch.no_grad():
        for index in range(len(dataset)):
            if time.monotonic()-started>=args.max_seconds:
                raise TimeoutError('Evaluation incomplete; no smaller-count success report written')
            item=dataset[index]
            if args.crop_size is not None:
                scale=cfg.model.scale;size=args.crop_size//scale
                h,w=item['input'].shape[-2:]
                if min(h,w)<size:raise ValueError('Source smaller than the explicitly requested crop')
                top,left=(h-size)//2,(w-size)//2
                item={**item,'input':item['input'][:,top:top+size,left:left+size],
                      'target':item['target'][:,top*scale:(top+size)*scale,left*scale:(left+size)*scale]}
            x=item['input'][None].to(device)
            if args.protocol=='swinir_sr_replay_v1':
                from utils.sr_metrics import swinir_reference_forward
                prediction=swinir_reference_forward(model,x,cfg.model.scale).float().cpu()[0]
            else:prediction=model(x).float().cpu()[0]
            border=cfg.model.scale if cfg.model.scale>1 else 0
            if args.protocol=='swinir_sr_replay_v1':
                from utils.sr_metrics import evaluate_sr_replay
                score=evaluate_sr_replay(prediction,item['target'],cfg.model.scale)
            else:score=evaluate_pair(prediction,item['target'],border)
            baseline=reference(item['input'][None])[0] if reference else item['input']
            reference_score=(evaluate_sr_replay(baseline,item['target'],cfg.model.scale) if args.protocol=='swinir_sr_replay_v1' else evaluate_pair(baseline,item['target'],border))
            scores.append(score);baselines.append(reference_score)
            rows.append({'id':item['id'],**score,'reference_psnr':reference_score['psnr'],'reference_ssim':reference_score['ssim']})
    if time.monotonic()-started>args.max_seconds:raise TimeoutError('Evaluation exceeded bounded compute budget')
    report={**aggregate(scores),'images':rows,'reference':aggregate(baselines),
            'origin':'official_supervised_pretrained','weights':receipt,'task_profile':cfg.degradation.__dict__,
            'model':cfg.model.__dict__,'dataset_hash':dataset.fingerprint,'split':args.split,
            'protocol':args.protocol,'ssim_protocol':'valid_gaussian11_sigma1.5', 'border':border,
            'metric_channel':'Y_BT601' if args.protocol=='swinir_sr_replay_v1' else 'RGB',
            'metric_quantization':args.protocol=='swinir_sr_replay_v1',
            'outer_padding':'official_concat_flip_plus_window' if args.protocol=='swinir_sr_replay_v1' else 'project_model_pad',
            'precision':'fp32','tile_mode':'full','environment':environment(device),'source':source_identity(),
            'reference_crop':'full_image' if args.crop_size is None else f'center_target_{args.crop_size}_after_input_operator',
            'elapsed_seconds':time.monotonic()-started,'training_performed':False,
            'pretrained_overlap_limitation':receipt['training_overlap'],
            'reference_operator':'torch_bicubic_reference_no_benchmark_claim' if reference else 'identity',
            'literature_benchmark_reproduction':False}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,report)
    print(f'Pretrained evaluation complete: {len(rows)} sources, report={output}')


if __name__=='__main__':main()
