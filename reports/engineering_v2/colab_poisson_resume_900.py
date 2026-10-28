"""One separately approved 900-update recovery; unchanged 5000-update schedule."""
from pathlib import Path
import argparse,hashlib,json,os,sys,uuid
SNAPSHOT_SHA='db13d20b37564c390e732453db0b1601ff4123feb1c6ac542c80ae01bf483127'


def sha(path: Path) -> str:
    """Stream an existing immutable artifact hash."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Reject mismatched state/environment and repeat recovery before any training."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--approved-resume',action='store_true');args=parser.parse_args()
    if not args.approved_resume:raise PermissionError('Specific 900-update / 900-second recovery approval required')
    if not Path('/content/drive').is_mount():raise RuntimeError('Durable Drive mount required')
    project=Path('/content/Image_Restoration_Extension_v3');os.chdir(project);sys.path[:0]=[str(project),str(project/'notebooks')]
    study=Path('/content/drive/MyDrive/RestorationProject/studies/poisson_rgb_5000_v3_cache')
    if list(study.glob('*/step_005000_micro_*.zip')):raise RuntimeError('A final snapshot already exists; inspect it instead of retraining')
    receipt=json.loads((project/'reports/poisson_recovery_preflight.json').read_text())
    if receipt['step']!=4100 or receipt['remaining_updates']!=900 or receipt['environment_differences']:raise ValueError('Recovery preflight is not matched')
    snapshot=Path(receipt['snapshot'])
    if sha(snapshot)!=SNAPSHOT_SHA:raise ValueError('Recovery snapshot changed')
    from utils.checkpoint import load_checkpoint,compatibility_hash
    from utils.config import config_from_dict
    from utils.runtime import select_device,environment
    from utils.degradation import codec_metadata
    from utils.reproducibility import source_identity
    from cloud_workflow import run_arm,inspect_snapshot
    state=load_checkpoint(receipt['prepared_checkpoint']);cfg=config_from_dict(state['config'])
    if state['step']!=4100 or state['total_steps']!=5000 or state['profile_only']:raise ValueError('Invalid research recovery state')
    current={**environment(select_device('cuda')),'codec':codec_metadata()}
    for key in ('python','torch','cuda','device_name','dependencies','codec'):
        if current.get(key)!=state['environment'].get(key):raise ValueError(f'Resume environment changed: {key}')
    if source_identity()['files']!=state['source']['files']:raise ValueError('Frozen training source changed')
    cfg.training.max_seconds=900
    if cfg.training.max_steps!=5000 or cfg.training.epochs is not None or compatibility_hash(cfg,state['datasets'])!=state['compatibility']:raise ValueError('Recovery changes scientific schedule/data')
    plan=json.loads((study/'reviewed_plan.json').read_text());plan['runs']=[{'id':'poisson_seed42','config':cfg.to_dict()}]
    plan['status']='explicitly_approved_single_recovery';plan['max_total_training_seconds']=900
    plan['recovery']={'from_step':4100,'remaining_updates':900,'max_additional_training_seconds':900,'snapshot_sha256':SNAPSHOT_SHA,'launcher_sha256':sha(Path(__file__)),'training_horizon':5000,'no_automatic_restart':True}
    uid=uuid.uuid4().hex[:8];planfile=project/'reports/poisson_cloud_v3'/('recovery_plan_'+uid+'.json');planfile.write_text(json.dumps(plan,indent=2)+'\n')
    (study/planfile.name).write_bytes(planfile.read_bytes())
    result=run_arm(project,planfile,'poisson_seed42',study.parent,approved=True,resume_archive=snapshot)
    run=Path(result['run_dir']);finals=list((study/run.name).glob('step_005000_micro_*.zip'))
    if not result['completed'] or result['step']!=5000 or len(finals)!=1:raise RuntimeError('Recovery incomplete; keep all backups for review')
    inspect_snapshot(finals[0])
    completed={'completed':True,'step':5000,'run_dir':str(run),'final_snapshot':str(finals[0]),'final_snapshot_sha256':sha(finals[0]),'recovered_from_step':4100,'recovery_max_seconds':900,'source_files':state['source']['files']}
    pointer=project/'reports/poisson_cloud_v3/completed_run.json'
    if pointer.exists():raise FileExistsError('Retain previous completed-run pointer')
    pointer.write_text(json.dumps(completed,indent=2)+'\n');(study/('recovery_completed_'+uid+'.json')).write_bytes(pointer.read_bytes())
    print(json.dumps(result,indent=2),flush=True);print('RECOVERY COMPLETE; final snapshot verified in Drive.',flush=True)


if __name__=='__main__':main()
