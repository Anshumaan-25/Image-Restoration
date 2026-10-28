"""Freeze development-selected checkpoints and a protected test manifest before scoring."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys
_script_directory=Path(__file__).resolve().parent
sys.path=[str(_script_directory.parent)]+[p for p in sys.path if Path(p).resolve()!=_script_directory]
from utils.checkpoint import load_checkpoint, compatibility_hash
from utils.config import config_from_dict
from utils.dataset_loader import read_manifest, validate_splits, manifest_fingerprint
from utils.io import file_sha256
from utils.logger import write_json


def freeze(plan_path: str, checkpoints: list[str], manifest: str, data_root: str, output: str) -> dict[str, Any]:
    """Match every planned arm/seed and require completed runs plus selected best files."""
    destination=Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    plan=json.loads(Path(plan_path).read_text())
    if plan.get('schema_version') != 1 or plan.get('status') != 'planned':
        raise ValueError('A resolved study plan is required')
    runs=plan['runs']
    if len(checkpoints) != len(runs):
        raise ValueError('Freeze requires exactly one selected checkpoint per planned arm/seed')
    test=read_manifest(Path(data_root)/manifest,data_root)
    if {r['split'] for r in test} != {'test'}:
        raise ValueError('Frozen evaluation requires a protected test manifest')
    selected=[]
    pending=list(runs)
    codec=None
    source=None
    for filename in checkpoints:
        path=Path(filename)
        state=load_checkpoint(path)
        if state.get('profile_only') or not state.get('data_identity'):
            raise ValueError('Research checkpoint with source identities required')
        latest=load_checkpoint(path.parent/'latest.pth')
        if latest['step'] != latest['total_steps'] or state['step'] != latest['best_step'] or state['best_metric'] != latest['best_metric'] or state['compatibility'] != latest['compatibility']:
            raise ValueError('Selected checkpoint must match a completed run development best')
        matches=[r for r in pending if compatibility_hash(config_from_dict(r['config']),state['datasets']) == state['compatibility']]
        if len(matches) != 1:
            raise ValueError('Checkpoint does not uniquely match a remaining planned arm/seed')
        run=matches[0]
        pending.remove(run)
        validate_splits(state['data_identity']+test)
        current_codec=state['environment']['codec']
        current_source=state['source']['files']
        if codec is not None and (codec != current_codec or source != current_source):
            raise ValueError('Study source or codec changed across arms')
        codec,source=current_codec,current_source
        selected.append({'run_id':run['id'],'path':str(path.resolve()),'sha256':file_sha256(path),
                         'development_best_step':state['step'],'development_best_worst_phase_psnr':state['best_metric']})
    result={'schema_version':1,'status':'frozen','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'plan_sha256':file_sha256(plan_path),'selected_checkpoints':[s['sha256'] for s in selected],
            'selections':selected,'test_manifest_hashes':[manifest_fingerprint(test)],
            'test_manifest_file_sha256':file_sha256(Path(data_root)/manifest),'test_count':len(test),
            'codec':codec,'source_files':source,'qualities':[20,10,40],'phase_count':64,
            'test_scored':False,'novelty_established':False}
    destination.parent.mkdir(parents=True,exist_ok=True)
    write_json(destination,result)
    return result


def main() -> None:
    """Write an immutable freeze record; this command never evaluates the test set."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--checkpoints',nargs='+',required=True)
    parser.add_argument('--test-manifest',required=True)
    parser.add_argument('--data-root',default='.')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=freeze(args.plan,args.checkpoints,args.test_manifest,args.data_root,args.output)
    print(f"frozen={args.output} selected={len(result['selections'])} test_images={result['test_count']} test_scored=False")


if __name__=='__main__':
    main()
