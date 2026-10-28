"""Offline finalizer contracts: durable selected weights, dev-only replay and export."""
from pathlib import Path
import contextlib,hashlib,io,json,os,runpy,sys,tempfile,unittest.mock as mock,zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'notebooks')]
launcher=Path(__file__).with_name('colab_poisson_finalize_v4.py');source=launcher.read_text();checks={}
with tempfile.TemporaryDirectory() as temporary:
    base=Path(temporary);project=base/'Image_Restoration_Extension_v3';study=base/'study';run=project/'experiments/cloud/poisson_rgb_5000_v3_cache/poisson_seed42_resume_fixture'
    run.mkdir(parents=True);(study/'poisson_seed42_resume_fixture').mkdir(parents=True)
    (run/'summary.json').write_text(json.dumps({'completed':True,'step':5000}));(run/'best_model.pth').write_bytes(b'authored selected-checkpoint fixture')
    best_sha=hashlib.sha256((run/'best_model.pth').read_bytes()).hexdigest()
    payload={'latest.pth':b'authored latest fixture','best_model.pth':(run/'best_model.pth').read_bytes()}
    archive=study/'poisson_seed42_resume_fixture/step_005000_micro_00010000.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name,data in payload.items():z.writestr(name,data)
        z.writestr('snapshot.json',json.dumps({'schema_version':1,'files':{name:hashlib.sha256(data).hexdigest() for name,data in payload.items()}}))
    (study/'cuda_profile.json').write_text('{}');(study/'reviewed_plan.json').write_text(json.dumps({'training_source_files':{'fixture.py':'authored source identity'}}));(study/'poisson_seed42_resume_fixture/cloud_workflow.py').write_text('# authored helper fixture')
    (study/'poisson_seed42_resume_fixture/plan.json').write_bytes((study/'reviewed_plan.json').read_bytes())
    pointer=project/'reports/poisson_cloud_v3/completed_run.json';pointer.parent.mkdir(parents=True);pointer.write_text(json.dumps({'run_dir':str(run)}))
    manifest=project/'data/manifests/poisson_v2/val.jsonl';manifest.parent.mkdir(parents=True);manifest.write_text('authored manifest fixture')
    text=source.replace("Path('/content/Image_Restoration_Extension_v3')",f'Path({str(project)!r})').replace("Path('/content/drive/MyDrive/RestorationProject/studies/poisson_rgb_5000_v3_cache')",f'Path({str(study)!r})')
    script=base/'finalizer.py';script.write_text(text)
    def fake_eval(command,**kwargs):
        peak=int(command[command.index('--peak-photons')+1]);out=Path(command[command.index('--output')+1]);out.write_text(json.dumps({'count':8,'split':'val','step':4700,'checkpoint_sha256':best_sha,'degradation':{'peak_photons':peak},'psnr':30.,'ssim':.9}))
    previous=Path.cwd();paths=list(sys.path)
    try:
        with mock.patch.object(Path,'is_mount',return_value=True), mock.patch('utils.reproducibility.source_identity',return_value={'files':{'fixture.py':'authored source identity'}}),mock.patch('utils.checkpoint.load_checkpoint',return_value={'step':4700}),mock.patch('subprocess.run',side_effect=fake_eval) as run_eval,contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(script),run_name='__main__');assert run_eval.call_count==3
        exports=list(study.glob('poisson_completed_*.zip'));assert len(exports)==1
        with zipfile.ZipFile(exports[0]) as z:
            assert z.testzip() is None
            inventory=json.loads(z.read('export_inventory.json'))
            assert len(inventory)==10
            for name,record in inventory.items():assert hashlib.sha256(z.read(name)).hexdigest()==record['sha256']
        checks['exact_three_dev_levels_and_verified_export']=True
        (run/'summary.json').write_text(json.dumps({'completed':False,'step':4900}))
        with mock.patch.object(Path,'is_mount',return_value=True):
            try:runpy.run_path(str(script),run_name='__main__')
            except RuntimeError as error:assert 'Training incomplete' in str(error);checks['incomplete_training_rejected_before_replay']=True
            else:raise AssertionError('Incomplete training accepted')
    finally:os.chdir(previous);sys.path[:]=paths
result={'status':'passed','scope':'Offline filesystem/inventory contracts with authored checkpoint/metric fixtures; not actual training, CUDA profiling or research scores','launcher_sha256':hashlib.sha256(launcher.read_bytes()).hexdigest(),'checks':checks}
launcher.with_name('poisson_finalize_v4_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
