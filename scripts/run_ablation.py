"""Plan matched JPEG experiments by default; --execute explicitly starts bounded runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import uuid
_script_directory=Path(__file__).resolve().parent
sys.path=[str(_script_directory.parent)]+[p for p in sys.path if Path(p).resolve()!=_script_directory]
from train import run_training
from utils.config import config_from_dict
from utils.logger import write_json
from utils.study import study_plan


def main() -> None:
    """Persist the exact resolved plan before executing any selected work."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',default='configs/jpeg/study.yaml')
    parser.add_argument('--device',required=True,choices=['cpu','mps','cuda'])
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--accumulation-steps',type=int)
    parser.add_argument('--precision',choices=['fp32','fp16','bf16'])
    parser.add_argument('--output',default=None)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    plan=study_plan(args.study,args.device,args.batch_size,args.accumulation_steps,args.precision)
    directory=Path(args.output or f'experiments/studies/{uuid.uuid4().hex[:10]}')
    directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'plan.json',plan)
    print(f"plan={directory/'plan.json'} runs={len(plan['runs'])} updates={plan['total_optimizer_updates']} execute={args.execute}",flush=True)
    if not args.execute:
        return
    results=[]
    for run in plan['runs']:
        result=run_training(config_from_dict(run['config']),directory/run['id'])
        results.append({'id':run['id'],**result})
        write_json(directory/'progress.json',{'results':results})
        if not result['completed']:
            print('Study stopped at incomplete run; resume that run explicitly before proceeding.',flush=True)
            raise SystemExit(2)
    write_json(directory/'completed.json',{'status':'training_completed','results':results,'protected_test_used':False})


if __name__=='__main__':
    main()
