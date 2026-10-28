"""Approved-only CPU phase evaluation; no training, downloads or test retuning."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, shutil, stat, subprocess, sys, time, uuid, zipfile


def digest(path: Path) -> str:
    """Stream an artifact SHA256."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Reject unsafe members before extracting any payload."""
    seen=set();items=archive.infolist()
    for e in items:
        p=PurePosixPath(e.filename)
        if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in e.filename or e.filename in seen or stat.S_ISLNK(e.external_attr>>16):
            raise ValueError('Unsafe ZIP member')
        seen.add(e.filename)
    if archive.testzip() is not None:raise ValueError('ZIP CRC failure')
    return items


def main() -> None:
    """Score fixed selections once, with 600-second per-job caps and Drive saves."""
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--approved',action='store_true')
    args=ap.parse_args()
    if not args.approved:raise PermissionError('Specific protected-test and CPU budget approval is required')
    drive=Path('/content/drive/MyDrive/RestorationProject')
    study=drive/'studies/cuda_jpeg_5000_v1'
    destination=study/'final_test_v1'
    if destination.exists():raise FileExistsError('Final evaluation already initiated; inspect saved progress before any authorized recovery')
    project=Path('/content/Image_Degradatation_Restoration')
    code=drive/'colab_restoration_v1.zip'
    if digest(code)!='37066216dcce9ae8f53fa449b22dcdef20a7a83e098001bb39bd62baaf87cb3f':raise ValueError('Original project bundle hash mismatch')
    if not project.exists():
        with zipfile.ZipFile(code) as z:
            entries=safe(z)
            if any(PurePosixPath(e.filename).parts[0]!=project.name for e in entries):raise ValueError('Unexpected project root')
            z.extractall('/content')
    os.chdir(project)
    sys.path[:0]=[str(project),str(project/'notebooks')]
    from cloud_workflow import verify_bundle,restore_snapshot
    from utils.reproducibility import source_identity
    from utils.degradation import codec_metadata
    from utils.runtime import environment,select_device
    from utils.study import compare_phase_reports
    verify_bundle(project)
    package=drive/'colab_frozen_test16_v1.zip'
    if digest(package)!='3c570cda7806d96535264373652fa85634a4d450a118937130cc44729e982519':raise ValueError('Frozen data bundle hash mismatch')
    with zipfile.ZipFile(package) as z:
        entries=safe(z);inv=json.loads(z.read('FINAL_EVAL_BUNDLE.json'))
        if {e.filename for e in entries}!=set(inv['files'])|{'FINAL_EVAL_BUNDLE.json'}:raise ValueError('Unexpected evaluation payload')
        for name,record in inv['files'].items():
            data=z.read(name)
            if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Evaluation payload hash mismatch')
            path=project/name
            if not path.resolve().is_relative_to(project.resolve()):raise ValueError('Unsafe destination')
            if path.exists():
                if digest(path)!=record['sha256']:raise FileExistsError(path)
            else:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    frozen=project/'reports/studies/cuda_jpeg_5000_v1/frozen_study.json'
    f=json.loads(frozen.read_text())
    if source_identity()['files']!=f['source_files']:raise ValueError('Frozen model source changed')
    if codec_metadata()!=f['codec']:raise ValueError('Colab codec changed; do not score mismatched JPEG inputs')
    work=project/'reports'/f'final_test_v1_{uuid.uuid4().hex[:8]}'
    work.mkdir(parents=True,exist_ok=False)
    checkpoints={}
    for arm,folder in {'A':'A_seed42','B':'B_seed42','C':'C_seed42_resume_45934e2e'}.items():
        restored=work/arm
        restore_snapshot(study/folder/'step_005000_micro_00020000.zip',restored)
        checkpoints[arm]=restored/'best_model.pth'
        if digest(checkpoints[arm])!=next(s['sha256'] for s in f['selections'] if s['run_id']==f'{arm}_seed42'):raise ValueError('Frozen checkpoint mismatch')
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copy2(frozen,destination/'frozen_study.json')
    env=environment(select_device('cpu'))
    (destination/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    status={'status':'running','budget_seconds_per_job':600,'jobs':9,'device':'cpu','spending_usd':0,'training':False,'frozen_study_sha256':digest(frozen),'completed':[]}
    statusfile=destination/'execution.json'
    statusfile.write_text(json.dumps(status,indent=2)+'\n')
    child_env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    try:
        for quality in (20,10,40):
            reports={}
            for arm,checkpoint in checkpoints.items():
                label=f'{arm}_Q{quality}';out=work/f'{label}.json'
                print(f'Evaluating {label}: CPU, 16 held-out images, all64 phases; no training',flush=True)
                command=[sys.executable,'eval.py','--checkpoint',str(checkpoint),'--data-dir',str(project),'--manifest','data/manifests/div2k_jpeg_v1/screening/test.jsonl','--split','test','--task','jpeg','--device','cpu','--phase-panel','all','--jpeg-quality',str(quality),'--frozen-study',str(frozen),'--output',str(out)]
                start=time.monotonic()
                result=subprocess.run(command,cwd=project,env=child_env,capture_output=True,text=True,timeout=600)
                (destination/f'{label}.log').write_text(result.stdout+'\n'+result.stderr)
                if result.returncode:raise RuntimeError(f'{label} failed: {result.stderr}')
                report=json.loads(out.read_text())
                if report['count']!=16 or report['phase_count']!=64:raise ValueError('Incomplete test panel')
                shutil.copy2(out,destination/out.name);reports[arm]=report
                status['completed'].append({'job':label,'seconds':time.monotonic()-start,'report_sha256':digest(out)})
                statusfile.write_text(json.dumps(status,indent=2)+'\n')
                print(f"Saved {label}; worst-phase PSNR={report['worst_phase_psnr']:.6f}; SSIM={report['ssim']:.6f}",flush=True)
            for a,b in [('A','B'),('B','C'),('A','C')]:
                comparison=compare_phase_reports([reports[a]],[reports[b]])
                (destination/f'{b}_minus_{a}_Q{quality}.json').write_text(json.dumps(comparison,indent=2)+'\n')
                print(f"Q{quality} {b}-{a}: {comparison['delta_db']:+.6f} dB; interval={comparison['image_bootstrap_ci95_db']}",flush=True)
    except BaseException as error:
        status['status']='incomplete';status['error']=repr(error);statusfile.write_text(json.dumps(status,indent=2)+'\n');raise
    status['status']='complete';statusfile.write_text(json.dumps(status,indent=2)+'\n')
    print('FINAL PHASE EVALUATION COMPLETE:',destination,flush=True)


if __name__=='__main__':main()
