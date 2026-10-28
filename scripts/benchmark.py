"""Synchronized model-only latency on synthetic inputs, separate from restoration accuracy."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
_script_directory=Path(__file__).resolve().parent
sys.path=[str(_script_directory.parent)]+[p for p in sys.path if Path(p).resolve()!=_script_directory]
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config, config_from_dict
from utils.logger import write_json
from utils.runtime import select_device, precision_context, synchronize, environment


def main() -> None:
    """Benchmark fixed batch-one shapes; no task degradation, training or test scoring."""
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config')
    source.add_argument('--weights')
    parser.add_argument('--device',required=True,choices=['cpu','mps','cuda'])
    parser.add_argument('--precision',default='fp32',choices=['fp32','fp16','bf16'])
    parser.add_argument('--sizes',nargs='+',type=int,default=[256,512])
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--max-seconds',type=float,default=120)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or any(v < 1 for v in args.sizes) or args.max_seconds <= 0 or not np.isfinite(args.max_seconds):
        parser.error('Positive sizes/iterations/time cap and nonnegative warmup required')
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    state=load_checkpoint(args.weights) if args.weights else None
    cfg=config_from_dict(state['config']) if state else load_config(args.config)
    device=select_device(args.device)
    model=build_model(cfg.model).to(device).eval()
    if state:
        model.load_state_dict(state['model'],strict=True)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    results=[]
    started=time.monotonic()
    stopped=False
    with torch.inference_mode(),precision_context(device,args.precision):
        for size in args.sizes:
            x=torch.rand(1,3,size,size,device=device)
            times=[]
            for iteration in range(args.warmup+args.iterations):
                synchronize(device)
                before=time.monotonic()
                y=model(x)
                synchronize(device)
                elapsed=time.monotonic()-before
                if iteration >= args.warmup:
                    times.append(elapsed)
                if y.shape != (1,3,size*cfg.model.scale,size*cfg.model.scale) or not torch.isfinite(y).all():
                    raise RuntimeError('Invalid inference shape or nonfinite output')
                if time.monotonic()-started >= args.max_seconds:
                    stopped=True
                    break
            results.append({'input_shape':[1,3,size,size],'output_scale':cfg.model.scale,
                            'measured_passes':len(times),'median_seconds':float(np.median(times)) if times else None,
                            'p95_seconds':float(np.quantile(times,.95)) if times else None})
            if stopped:
                break
    report={'model':cfg.model.__dict__,'model_parameters':sum(p.numel() for p in model.parameters()),
            'environment':environment(device),'precision':args.precision,'weights':args.weights,
            'random_weights':state is None,'warmup':args.warmup,'requested_measured_passes':args.iterations,
            'completed':not stopped,'batch_size':1,'self_ensemble':False,'tile_mode':'full',
            'timing_scope':'synchronized model forward only; synthetic input preallocated; excludes I/O/degradation/metrics',
            'results':results,'restoration_accuracy_evaluated':False,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
            'mps_driver_allocated_bytes':torch.mps.driver_allocated_memory() if device.type=='mps' else None}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,report)
    print(f'latency_report={output} completed={not stopped}')
    if stopped:
        raise SystemExit(2)


if __name__=='__main__':
    main()
