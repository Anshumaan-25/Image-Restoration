"""Run only the specifically approved recovery, development export and SIDD validation."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time,uuid,zipfile
HELPERS={
 'colab_poisson_resume_900.py':'27c0176505797e00a91bc24231cfd057505eb86fcd5f51081d82c84e3bb68562',
 'colab_poisson_finalize_v4.py':'9206337afa94ac72a132761927e8fa1e59329c5d59f24777239c2ce9a30fd8c8',
 'colab_sidd_full_validation_v2.py':'acf8a0ca2388b142c82736af475658c142e4a2d162b9a67424bd9a37f427cc51'}


def sha(path: Path) -> str:
    """Stream supplied artifact integrity checks without downloads."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Fail closed; never repeat training or incomplete SIDD automatically."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--approved-stages',action='store_true');args=parser.parse_args()
    if not args.approved_stages:raise PermissionError('Specific 900-update recovery and full SIDD validation approvals required')
    if not Path('/content/drive').is_mount():raise RuntimeError('Use the browser session with a real Drive mount')
    base=Path('/content/drive/MyDrive/RestorationProject');studies=base/'studies'
    project=Path('/content/Image_Restoration_Extension_v3')
    preflight=json.loads((project/'reports/poisson_recovery_preflight.json').read_text())
    if preflight['step']!=4100 or preflight['remaining_updates']!=900 or preflight['environment_differences']:raise ValueError('Matched recovery preflight required')
    if list((studies/'poisson_rgb_5000_v3_cache').glob('*/step_005000_micro_*.zip')):raise RuntimeError('Final Poisson snapshot already exists; inspect rather than repeat training')
    if list(studies.glob('sidd_public_validation_full_v1_*')):raise RuntimeError('A previous full SIDD attempt exists; review its results/budget before another attempt')
    for name,digest in HELPERS.items():
        path=base/name
        if sha(path)!=digest:raise ValueError(f'Helper changed: {name}')
    for name,size,digest in [('test.zip',461780640,'1139c32896697e960159ae845070751e58894ddf607ebfe8f6d83ef012e2fbff'),('real_denoising.pth',104611957,'4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387')]:
        path=base/name
        if path.stat().st_size!=size or sha(path)!=digest:raise ValueError(f'Approved SIDD payload mismatch: {name}')
    destination=studies/('approved_completion_chain_'+uuid.uuid4().hex[:8]);destination.mkdir(parents=True,exist_ok=False)
    journal={'status':'running','training_recovery_updates':900,'training_recovery_cap_seconds':900,'full_sidd_cap_seconds':1200,'spending_usd':0,'protected_test_scoring':False,'completed_stages':[],'launcher_sha256':sha(Path(__file__))}
    receipt=destination/'execution.json';receipt.write_text(json.dumps(journal,indent=2)+'\n')
    jobs=[('poisson_recovery','colab_poisson_resume_900.py',['--approved-resume']),('development_replay_export','colab_poisson_finalize_v4.py',[]),('full_sidd_public_validation','colab_sidd_full_validation_v2.py',['--approved-full-validation'])]
    try:
        for stage,name,flags in jobs:
            local=Path('/content')/name;shutil.copyfile(base/name,local)
            print('Starting approved stage:',stage,flush=True);started=time.monotonic()
            with (destination/(stage+'.log')).open('x') as log:
                with subprocess.Popen([sys.executable,str(local),*flags],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                    for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    code=process.wait()
            if code:raise RuntimeError(f'{stage} failed; inspect retained log. No retry, fallback or further stage started.')
            journal['completed_stages'].append({'stage':stage,'elapsed_seconds':time.monotonic()-started});receipt.write_text(json.dumps(journal,indent=2)+'\n')
        candidates=list(studies.glob('sidd_public_validation_full_v1_*'))
        if len(candidates)!=1:raise ValueError('Expected one unambiguous full SIDD study')
        sidd=candidates[0];result=json.loads((sidd/'results.json').read_text())
        if result['status']!='complete' or result['count']!=1280 or len({r['id'] for r in result['images']})!=1280:raise ValueError('SIDD completion count/identity mismatch')
        payload=[sidd/name for name in ['frozen_plan.json','cuda_profile.json','results.json','progress.json','validation_manifest.jsonl']]
        inventory={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in payload}
        export=sidd/('sidd_completed_'+uuid.uuid4().hex[:8]+'.zip')
        with zipfile.ZipFile(export,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in payload:archive.write(p,p.name)
            archive.writestr('export_inventory.json',json.dumps(inventory,indent=2)+'\n')
        with zipfile.ZipFile(export) as archive:
            if archive.testzip() is not None:raise ValueError('SIDD export CRC failure')
            for name,record in inventory.items():
                data=archive.read(name)
                if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('SIDD export inventory mismatch')
        journal.update({'status':'complete','sidd_export':str(export),'sidd_export_bytes':export.stat().st_size,'sidd_export_sha256':sha(export),'sidd_count':1280,'sidd_psnr':result['psnr'],'sidd_ssim':result['ssim']})
        receipt.write_text(json.dumps(journal,indent=2)+'\n');print(json.dumps(journal,indent=2),flush=True)
    except BaseException as error:
        journal.update({'status':'incomplete','error':repr(error)});receipt.write_text(json.dumps(journal,indent=2)+'\n');raise


if __name__=='__main__':main()
