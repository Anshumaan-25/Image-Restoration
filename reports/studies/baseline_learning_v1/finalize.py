"""Run the approved baseline's bounded development-only verification after training."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np
from PIL import Image
from utils.checkpoint import run_lock
from utils.logger import write_json
from utils.reproducibility import source_identity

HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(ROOT)
RUN=Path('experiments/studies/baseline_learning_v1/B_seed42')
WEIGHTS=str(RUN/'best_model.pth')


def main() -> None:
    """Retain every command/result, fail explicitly, never score protected data."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.wait_seconds<=1800:raise ValueError('Wait must be bounded to 0..1800 seconds')
    output=HERE/'postprocess.json'
    if output.exists():raise FileExistsError(output)
    wait_deadline=time.monotonic()+args.wait_seconds
    while True:
        ready=False
        try:
            with run_lock(RUN):
                ready=(RUN/'summary.json').exists()
        except RuntimeError as error:
            if 'locked' not in str(error).lower():raise
        if ready:break
        if time.monotonic()>=wait_deadline:raise RuntimeError('Training has not finished/saved its summary within the bounded wait')
        time.sleep(2)
    proposal=json.loads(Path('reports/proposals/baseline_learning_v1/proposal.json').read_text())
    assert source_identity()['files']==proposal['source_files']
    supplement=json.loads((HERE/'supplementary_plan.json').read_text())
    assert hashlib.sha256(Path(supplement['manifest']).read_bytes()).hexdigest()==supplement['manifest_file_sha256']
    record={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'device':'mps',
            'protected_test_scored':False,'commands':[],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output,record)
    started=time.monotonic()

    def execute(arguments: list[str], timeout: int=180) -> subprocess.CompletedProcess:
        """Use shared CLIs with bounded runtime and retain real outputs, including failures."""
        remaining=900-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Overall verification budget exhausted')
        before=time.monotonic()
        result=subprocess.run([sys.executable,*arguments],capture_output=True,text=True,timeout=min(timeout,remaining))
        record['commands'].append({'command':[sys.executable,*arguments],'exit_code':result.returncode,
                                   'seconds':time.monotonic()-before,'stdout':result.stdout,'stderr':result.stderr})
        write_json(output,record)
        print(f"check={arguments[0]} exit={result.returncode}",flush=True)
        return result

    try:
        with run_lock(RUN):
            commands=[
                ['eval.py','--checkpoint',WEIGHTS,'--phase-panel','dev','--device','mps','--output',str(REL/'development.json')],
                ['eval.py','--checkpoint',WEIGHTS,'--full-image','--device','mps','--output',str(REL/'full_image_development.json')],
                [str(REL/'input_reference.py')],
                supplement['command'][1:],
                [str(REL/'render_examples.py'),'--device','mps'],
                ['inference.py','--input',str(REL/'0721_input.png'),'--output',str(REL/'0721_B_5000_cli.png'),'--weights',WEIGHTS,'--device','mps'],
                [str(REL/'analyze.py')]
            ]
            for arguments in commands:
                result=execute(arguments)
                if result.returncode:raise RuntimeError(f'Verification command failed: {arguments}')
            direct=np.asarray(Image.open(HERE/'0721_B_5000.png'))
            exported=np.asarray(Image.open(HERE/'0721_B_5000_cli.png'))
            cli={'status':'passed' if np.array_equal(direct,exported) else 'failed','same_as_direct_export':bool(np.array_equal(direct,exported)),
                 'max_absolute_uint8_difference':int(np.abs(direct.astype('int16')-exported.astype('int16')).max()),
                 'output_sha256':hashlib.sha256((HERE/'0721_B_5000_cli.png').read_bytes()).hexdigest(),'protected_test_scored':False}
            write_json(HERE/'trained_cli_verification.json',cli)
            assert cli['status']=='passed',cli
            secondary=json.loads((HERE/'supplementary_development72.json').read_text())
            expected_ids={r['id'] for r in [json.loads(line) for line in Path(supplement['manifest']).read_text().splitlines()]}
            assert secondary['count']==72 and {i['id'] for i in secondary['images']}==expected_ids
            independent=float(np.mean([min(p['psnr'] for p in i['phases']) for i in secondary['images']]))
            assert abs(independent-secondary['worst_phase_psnr'])<1e-12
            assert secondary['split']=='val' and secondary['phase_count']==16
            secondary_summary={'count':72,'selected_step':secondary['step'],'checkpoint_sha256':secondary['checkpoint_sha256'],
                'worst_phase_psnr':secondary['worst_phase_psnr'],'input_worst_phase_psnr':secondary['input_worst_phase_psnr'],
                'worst_phase_gain_over_input_db':secondary['worst_phase_psnr']-secondary['input_worst_phase_psnr'],
                'mean_phase_psnr':secondary['psnr'],'mean_phase_ssim':secondary['ssim'],'input_mean_ssim':secondary['input_mean_ssim'],
                'independent_aggregation_verified':True,'protected_test_scored':False,
                'role':'Supplementary development sources unused for checkpoint selection; prepared after observing interim primary improvement, not confirmatory test inference.'}
            write_json(HERE/'supplementary_summary.json',secondary_summary)
            rejected=execute(['scripts/compare_study.py','--baseline','reports/studies/mps_screening_v1/B_dev.json',
                              '--candidate',str(REL/'development.json'),'--output',str(REL/'INVALID_unmatched_budget_comparison.json')])
            assert rejected.returncode!=0 and not (HERE/'INVALID_unmatched_budget_comparison.json').exists()
            assert 'incomparable evaluation protocol' in rejected.stderr.lower() and ('training_config' in rejected.stderr or 'selection_horizon' in rejected.stderr),rejected.stderr
            write_json(HERE/'budget_guard_verification.json',{'status':'passed','unmatched_budget_rejected':True,
                'exit_code':rejected.returncode,'reason':rejected.stderr,'protected_test_scored':False})
            assert source_identity()['files']==proposal['source_files']
        record.update(status='passed',elapsed_seconds=time.monotonic()-started,source_unchanged=True,
                      expected_budget_guard_rejection=True,completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output,record)
        execution=json.loads((HERE/'execution.json').read_text());execution['status']='training_and_development_verification_finished'
        execution['shutdown_request']['verification_finished']=True
        execution['shutdown_request']['shutdown_not_started']=True
        write_json(HERE/'execution.json',execution)
        print(f'Postprocessing passed: {output}',flush=True)
    except Exception as error:
        record.update(status='failed',error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(output,record)
        raise


if __name__=='__main__':main()
