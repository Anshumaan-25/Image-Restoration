"""Profile first, then separately authorized bounded Poisson training on Colab CUDA."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,os,subprocess,sys,zipfile

EXPECTED_BUNDLE_SHA256='db81ac9de44a48699de33c00d5dba16aa476f11f376f797a8f0bee38ee8c2240'


def sha(path: Path) -> str:
    """Hash supplied artifacts without network access."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Keep profiling and long-run approval distinct; snapshots go to mounted Drive."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['profile','train'],default='profile')
    parser.add_argument('--approved-training',action='store_true')
    args=parser.parse_args()
    if args.mode=='train' and not args.approved_training:raise PermissionError('Specific 5000-update/5400-second Poisson run approval required')
    if not Path('/content/drive').is_mount():raise RuntimeError('Google Drive is not mounted; no profile or training starts without durable backups')
    drive=Path('/content/drive/MyDrive/RestorationProject')
    archive_path=Path('/content/colab_extension_v3.zip')
    if not archive_path.is_file():archive_path=drive/archive_path.name
    if sha(archive_path)!=EXPECTED_BUNDLE_SHA256:raise ValueError('Extension bundle SHA256 mismatch')
    project=Path('/content/Image_Restoration_Extension_v3')
    with zipfile.ZipFile(archive_path) as archive:
        # No external dependency is needed before extraction.
        import stat
        seen=set()
        for entry in archive.infolist():
            p=PurePosixPath(entry.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in entry.filename or entry.filename in seen or stat.S_ISLNK(entry.external_attr>>16):raise ValueError('Unsafe extension ZIP member')
            if not p.parts or p.parts[0]!=project.name:raise ValueError('Unexpected extension root')
            seen.add(entry.filename)
        if archive.testzip() is not None:raise ValueError('Extension archive CRC failure')
        inventory=json.loads(archive.read(project.name+'/EXTENSION_INVENTORY.json'))
        if seen!={project.name+'/'+name for name in inventory}|{project.name+'/EXTENSION_INVENTORY.json'}:raise ValueError('Unexpected bundle members')
        for name,record in inventory.items():
            data=archive.read(project.name+'/'+name)
            if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Extension file hash mismatch')
            target=project/name
            if target.exists():
                if sha(target)!=record['sha256']:raise FileExistsError('Retain changed runtime source; use a fresh session')
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    os.chdir(project);sys.path[:0]=[str(project),str(project/'notebooks')]
    from utils.runtime import select_device,environment
    from utils.config import load_config
    from utils.degradation import codec_metadata
    from utils.reproducibility import source_identity
    device=select_device('cuda');cfg=load_config('configs/denoising/poisson_research_v2.yaml')
    location=project/'reports/poisson_cloud_v3';location.mkdir(parents=True,exist_ok=True)
    profile=location/'cuda_profile.json';planfile=location/'reviewed_plan.json'
    backup=drive/'studies'
    if args.mode=='profile':
        if profile.exists() or planfile.exists():raise FileExistsError('Prior profile retained; review before rerunning')
        subprocess.run([sys.executable,'scripts/profile.py','--config','configs/denoising/poisson_research_v2.yaml','--device','cuda','--warmup','10','--steps','50','--max-seconds','300','--output',str(profile)],check=True,timeout=360)
        raw=json.loads(profile.read_text())
        if not raw['completed'] or raw['measured_updates']!=50 or raw['peak_cuda_allocated_bytes']>12*1024**3:raise ValueError('Poisson profile failed count/memory gate')
        projection=raw['mean_update_seconds']*5000
        if projection>=4320:raise ValueError('Projected updates leave insufficient validation/save headroom in 90-minute cap')
        from utils.checkpoint import load_checkpoint
        state=load_checkpoint(Path(raw['run_dir'])/'latest.pth')
        if state['skipped_updates']:raise ValueError('Profile has AMP skips; review precision before research training')
        manifests={}
        for path in (cfg.dataset.train_manifest,cfg.dataset.val_manifest):manifests[path]={'path':path,'sha256':sha(project/path)}
        plan={'name':'poisson_rgb_5000_v3_cache','runs':[{'id':'poisson_seed42','config':cfg.to_dict()}],
              'cloud_launcher_sha256':sha(project/'notebooks/cloud_workflow.py'),
              'cuda_profile':{'path':str(profile.relative_to(project)),'sha256':sha(profile),'projected_update_seconds':projection},
              'cuda_environment':{**environment(device),'codec':codec_metadata()},
              'training_source_files':source_identity()['files'],'manifests':manifests,
              'status':'profiled_not_training_authorized','total_optimizer_updates':5000,'max_total_training_seconds':5400,
              'spending_usd':0,'protected_test_scoring':False}
        planfile.write_text(json.dumps(plan,indent=2)+'\n')
        destination=backup/plan['name'];destination.mkdir(parents=True,exist_ok=True)
        import shutil
        for path in (profile,planfile):
            target=destination/path.name
            if target.exists():raise FileExistsError('Preserve previous Drive profile')
            shutil.copy2(path,target)
        print(json.dumps({'profile_passed':True,'mean_update_seconds':raw['mean_update_seconds'],'peak_allocated_GiB':raw['peak_cuda_allocated_bytes']/1024**3,'projected_update_minutes':projection/60,'training_started':False,'plan':str(destination/planfile.name)},indent=2),flush=True)
        return
    if not planfile.exists():raise FileNotFoundError('Run and review actual-device profile first')
    plan=json.loads(planfile.read_text())
    if plan.get('runs')!=[{'id':'poisson_seed42','config':cfg.to_dict()}]:raise ValueError('Prepared run differs from the fixed reviewed configuration')
    if plan.get('max_total_training_seconds')!=5400 or plan.get('total_optimizer_updates')!=5000:raise ValueError('Prepared budget changed')
    from cloud_workflow import run_arm
    result=run_arm(project,planfile,'poisson_seed42',backup,approved=True)
    print(json.dumps(result,indent=2),flush=True)
    print('POISSON TRAINING COMPLETE; protected test has not been scored.',flush=True)


if __name__=='__main__':main()
