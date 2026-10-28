"""Run the required local acceptance commands and retain exact evidence in new JSON."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import time
_script_directory=Path(__file__).resolve().parent
sys.path=[str(_script_directory.parent)]+[p for p in sys.path if Path(p).resolve()!=_script_directory]
from utils.logger import write_json
from utils.reproducibility import source_identity
from utils.runtime import environment, select_device


def main() -> None:
    """Fail on any required check; no dependency installation or network activity."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    parser.add_argument('--data-check',action='store_true')
    args=parser.parse_args()
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    root=_script_directory.parent
    identity=source_identity()
    commands=[['-m','unittest','discover','-s','tests','-p','test_*.py','-v'],['tests/smoke_test.py']]
    sources=['train.py','eval.py','inference.py']+[str(p.relative_to(root)) for name in ('models','losses','utils','scripts') for p in sorted((root/name).glob('*.py'))]
    commands.extend([['-m','py_compile',*sources],['train.py','--dry-run','--epochs','1','--batch-size','2','--device',args.device],['-m','pip','check']])
    if args.data_check:
        commands.append(['scripts/prepare_manifest.py','--config','configs/jpeg/div2k_screening.yaml','--validate-only'])
    checks=[]
    for command in commands:
        started=time.monotonic()
        try:
            result=subprocess.run([sys.executable,*command],cwd=root,capture_output=True,text=True,timeout=180)
            record={'command':[sys.executable,*command],'exit_code':result.returncode,
                    'seconds':time.monotonic()-started,'stdout':result.stdout,'stderr':result.stderr}
        except subprocess.TimeoutExpired as error:
            record={'command':[sys.executable,*command],'exit_code':None,'seconds':time.monotonic()-started,'error':str(error)}
        checks.append(record)
        print(f"check={' '.join(command[:4])} exit={record['exit_code']}",flush=True)
        if record['exit_code'] != 0:
            break
    match=re.search(r'Ran (\d+) tests? in ([\d.]+)s',checks[0].get('stderr',''))
    skips=re.search(r'skipped=(\d+)',checks[0].get('stderr',''))
    final_source=source_identity()
    passed=all(r['exit_code']==0 for r in checks) and len(checks)==len(commands) and identity['files']==final_source['files']
    report={'status':'passed' if passed else 'failed','verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'test_count':int(match.group(1)) if match else None,'test_seconds':float(match.group(2)) if match else None,
            'test_skips':int(skips.group(1)) if skips else 0,'checks':checks,'environment':environment(select_device(args.device)),
            'source':identity,'source_changed_during_verification':identity['files']!=final_source['files'],
            'research_study_completed':False,'novelty_established':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,report)
    print(f"verification={output} status={report['status']} tests={report['test_count']}")
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
