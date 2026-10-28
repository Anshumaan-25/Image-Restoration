"""After approved training, freeze selected weights, replay development data and export."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,uuid,zipfile


def sha(path: Path) -> str:
    """Hash a retained local artifact without network access."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Never train, score protected tests, change selection or overwrite old exports."""
    if not Path('/content/drive').is_mount():raise RuntimeError('Durable Drive mount required')
    project=Path('/content/Image_Restoration_Extension_v3');os.chdir(project);sys.path[:0]=[str(project),str(project/'notebooks')]
    study=Path('/content/drive/MyDrive/RestorationProject/studies/poisson_rgb_5000_v3_cache')
    parent=project/'experiments/cloud/poisson_rgb_5000_v3_cache'
    pointer=project/'reports/poisson_cloud_v3/completed_run.json'
    run=Path(json.loads(pointer.read_text())['run_dir']) if pointer.exists() else parent/'poisson_seed42'
    if run.resolve().parent!=parent.resolve() or not run.name.startswith('poisson_seed42'):raise ValueError('Unexpected completed-run path')
    summary=json.loads((run/'summary.json').read_text())
    if not summary.get('completed') or summary.get('step')!=5000:raise RuntimeError('Training incomplete; preserve snapshots and review before recovery')
    from cloud_workflow import inspect_snapshot
    from utils.checkpoint import load_checkpoint
    from utils.reproducibility import source_identity
    plan=json.loads((study/'reviewed_plan.json').read_text())
    if source_identity()['files']!=plan['training_source_files']:raise ValueError('Frozen source identity changed')
    snapshots=list((study/run.name).glob('step_005000_micro_*.zip'))
    if len(snapshots)!=1:raise ValueError('Expected one unambiguous final checkpoint snapshot')
    snapshot=snapshots[0];inventory=inspect_snapshot(snapshot)
    best=run/'best_model.pth'
    if sha(best)!=inventory['files']['best_model.pth']:raise ValueError('Selected checkpoint differs from final durable backup')
    checkpoint=load_checkpoint(best)
    destination=study/('development_replay_'+uuid.uuid4().hex[:8]);destination.mkdir(exist_ok=False)
    frozen={'role':'development validation only, not protected testing','selected_checkpoint_sha256':sha(best),'selected_step':checkpoint['step'],'selection':'Highest fixed peak60 development PSNR, earliest tie, no reselection','levels':[30,60,120],'expected_sources':8,'manifest_sha256':sha(project/'data/manifests/poisson_v2/val.jsonl'),'training_source_files':plan['training_source_files'],'precision':'fp32','device':'cpu','training':False,'protected_test_scoring':False,'spending_usd':0}
    (destination/'frozen_development_replay.json').write_text(json.dumps(frozen,indent=2)+'\n')
    reports=[]
    for peak in frozen['levels']:
        output=destination/f'peak_{peak}_development.json'
        command=[sys.executable,'eval.py','--checkpoint',str(best),'--device','cpu','--peak-photons',str(peak),'--output',str(output)]
        result=subprocess.run(command,check=True,timeout=120)
        report=json.loads(output.read_text())
        if report['count']!=8 or report['split']!='val' or report['step']!=checkpoint['step'] or report['checkpoint_sha256']!=frozen['selected_checkpoint_sha256'] or report['degradation']['peak_photons']!=peak:raise ValueError('Development replay contract mismatch')
        reports.append(output)
        print(f"Poisson development peak{peak}: PSNR={report['psnr']:.6f}; SSIM={report['ssim']:.6f}; count=8",flush=True)
    files=[snapshot,study/'cuda_profile.json',study/'reviewed_plan.json',study/run.name/'cloud_workflow.py',destination/'frozen_development_replay.json',*reports]
    if pointer.exists():
        shutil.copy2(pointer,destination/'completed_run.json');files.extend([destination/'completed_run.json',study/run.name/'plan.json'])
    export=study/('poisson_completed_'+uuid.uuid4().hex[:8]+'.zip')
    records={str(p.relative_to(study)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    with zipfile.ZipFile(export,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for p in files:archive.write(p,str(p.relative_to(study)))
        archive.writestr('export_inventory.json',json.dumps(records,indent=2)+'\n')
    with zipfile.ZipFile(export) as archive:
        if archive.testzip() is not None:raise ValueError('Export CRC failure')
        for name,record in records.items():
            data=archive.read(name)
            if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Export inventory failure')
    receipt={'status':'complete','selected_step':checkpoint['step'],'target_steps':5000,'development_levels':[30,60,120],'protected_test_scoring':False,'export':str(export),'bytes':export.stat().st_size,'sha256':sha(export)}
    (destination/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
