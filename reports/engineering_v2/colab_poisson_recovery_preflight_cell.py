# Restore verified files and inspect recovery state only. NO TRAINING.
from google.colab import drive
from pathlib import Path, PurePosixPath
import hashlib, json, os, stat, sys, uuid, zipfile, shutil

drive.mount('/content/drive')
assert Path('/content/drive').is_mount()
base = Path('/content/drive/MyDrive/RestorationProject')
archive = base / 'colab_extension_v3.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'db81ac9de44a48699de33c00d5dba16aa476f11f376f797a8f0bee38ee8c2240'
project = Path('/content/Image_Restoration_Extension_v3')
with zipfile.ZipFile(archive) as z:
    names=set()
    for entry in z.infolist():
        path=PurePosixPath(entry.filename)
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in entry.filename
        assert path.parts[0] == project.name and entry.filename not in names
        assert not stat.S_ISLNK(entry.external_attr >> 16)
        names.add(entry.filename)
    assert z.testzip() is None
    inventory=json.loads(z.read(project.name+'/EXTENSION_INVENTORY.json'))
    assert names == {project.name+'/'+name for name in inventory} | {project.name+'/EXTENSION_INVENTORY.json'}
    for name,record in inventory.items():
        data=z.read(project.name+'/'+name)
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
        target=project/name
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256'], f'Retain changed file: {name}'
        else:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
os.chdir(project)
sys.path[:0]=[str(project),str(project/'notebooks')]
from cloud_workflow import inspect_snapshot, restore_snapshot
from utils.checkpoint import load_checkpoint
from utils.runtime import select_device, environment
from utils.reproducibility import source_identity
from utils.degradation import codec_metadata
study=base/'studies/poisson_rgb_5000_v3_cache'
snapshot=study/'poisson_seed42/step_004100_micro_00008200.zip'
assert hashlib.sha256(snapshot.read_bytes()).hexdigest()=='db13d20b37564c390e732453db0b1601ff4123feb1c6ac542c80ae01bf483127'
inspect_snapshot(snapshot)
prepared=project/'experiments'/('recovery_preflight_'+uuid.uuid4().hex[:8])
state=load_checkpoint(restore_snapshot(snapshot,prepared))
device=select_device('cuda')
current={**environment(device),'codec':codec_metadata()}
keys=('python','torch','cuda','device_name','dependencies','codec')
differences={key:{'saved':state['environment'].get(key),'current':current.get(key)} for key in keys if state['environment'].get(key)!=current.get(key)}
assert state['source']['files']==source_identity()['files'], 'Frozen source changed'
assert state['step']==4100 and state['total_steps']==5000 and not state['profile_only']
plan=json.loads((study/'reviewed_plan.json').read_text())
location=project/'reports/poisson_cloud_v3';location.mkdir(parents=True,exist_ok=True)
for name in ['reviewed_plan.json','cuda_profile.json']:
    target=location/name;data=(study/name).read_bytes()
    if target.exists():assert target.read_bytes()==data
    else:target.write_bytes(data)
receipt={'step':state['step'],'best_step':state['best_step'],'remaining_updates':5000-state['step'],'skipped_updates':state['skipped_updates'],'environment_differences':differences,'environment':current,'snapshot':str(snapshot),'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'prepared_checkpoint':str(prepared/'latest.pth'),'training_started':False}
(project/'reports/poisson_recovery_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
print('RECOVERY INSPECTED. NO TRAINING STARTED.')
