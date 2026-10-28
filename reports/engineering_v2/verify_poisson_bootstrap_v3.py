"""Offline bootstrap integration: real ZIP extraction, mocked CUDA/profile/training."""
from pathlib import Path
import contextlib,hashlib,io,json,os,runpy,sys,tempfile,subprocess,unittest.mock as mock
real_run=subprocess.run

root=Path.cwd()
sys.path.insert(0,str(root))
launcher=root/'reports/engineering_v2/colab_poisson_launcher_v3.py'
text=launcher.read_text()
results={}
with tempfile.TemporaryDirectory() as temporary:
    location=Path(temporary)
    archive=root/'dist/colab_extension_v3.zip'
    drive=location/'drive'; drive.mkdir()
    project=location/'Image_Restoration_Extension_v3'
    altered=text.replace("Path('/content/drive/MyDrive/RestorationProject')",f'Path({str(drive)!r})').replace("Path('/content/colab_extension_v3.zip')",f'Path({str(archive)!r})').replace("Path('/content/Image_Restoration_Extension_v3')",f'Path({str(project)!r})')
    test_launcher=location/'launcher.py'; test_launcher.write_text(altered)
    def profile_stub(command,**kwargs):
        if "--output" not in command: return real_run(command,**kwargs)
        output=Path(command[command.index('--output')+1]); run=project/'experiments/profile_fixture';run.mkdir(parents=True)
        output.write_text(json.dumps({'completed':True,'measured_updates':50,'peak_cuda_allocated_bytes':1234,'mean_update_seconds':.5,'run_dir':str(run)}))
    import torch
    from utils.runtime import select_device
    cwd=Path.cwd(); previous=list(sys.path)
    try:
        with mock.patch.object(Path,'is_mount',return_value=True), mock.patch('utils.runtime.select_device',return_value=torch.device('cpu')), mock.patch('utils.checkpoint.load_checkpoint',return_value={'skipped_updates':0}), mock.patch('subprocess.run',side_effect=profile_stub), mock.patch.object(sys,'argv',[str(test_launcher),'--mode','profile']), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(test_launcher),run_name='__main__')
        results['fresh_parent_directory_extraction_profile_plan_backup']=True
        with mock.patch.object(Path,'is_mount',return_value=False), mock.patch.object(sys,'argv',[str(test_launcher),'--mode','profile']):
            try:runpy.run_path(str(test_launcher),run_name='__main__')
            except RuntimeError as error:
                assert 'Drive is not mounted' in str(error)
                results['unmounted_drive_rejected_before_profile_or_training']=True
            else:raise AssertionError('Missing durable mount guard')
        plan=project/'reports/poisson_cloud_v3/reviewed_plan.json'
        assert plan.is_file() and (drive/'studies/poisson_rgb_5000_v3_cache/reviewed_plan.json').read_bytes()==plan.read_bytes()
        import cloud_workflow
        with mock.patch.object(Path,'is_mount',return_value=True), mock.patch('utils.runtime.select_device',return_value=torch.device('cpu')), mock.patch.object(cloud_workflow,'run_arm',return_value={'completed':True}) as run, mock.patch.object(sys,'argv',[str(test_launcher),'--mode','train','--approved-training']), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(test_launcher),run_name='__main__')
            assert run.call_count==1 and run.call_args.kwargs['approved'] is True
        results['matching_plan_calls_authorized_training_once']=True
        with mock.patch.object(sys,'argv',[str(test_launcher),'--mode','train']):
            try:runpy.run_path(str(test_launcher),run_name='__main__')
            except PermissionError:results['unapproved_training_rejected']=True
            else:raise AssertionError('Missing permission guard')
    finally:
        os.chdir(cwd);sys.path[:]=previous
report={'status':'passed','scope':'Actual archive/bootstrap filesystem integration only; CUDA timing, memory, AMP and training mocked, NOT a GPU profile or scientific result','launcher_sha256':hashlib.sha256(launcher.read_bytes()).hexdigest(),'checks':results}
(root/'reports/engineering_v2/poisson_bootstrap_v3_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
