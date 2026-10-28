"""Verify imported cloud artifacts and recompute development statistics offline."""
from pathlib import Path
import sys,hashlib,json,zipfile,stat,csv,io,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'notebooks'))
from cloud_workflow import safe_members,restore_snapshot,sha256
from utils.checkpoint import load_checkpoint,compatibility_hash
from utils.config import config_from_dict
from utils.reproducibility import source_identity
from utils.dataset_loader import read_manifest,validate_splits
from utils.study import compare_phase_reports
from utils.degradation import codec_metadata

source=Path('/Users/anshu/Downloads/restoration_completed_study_28aebcdf.zip')
expected='55795ddf1a1b3dc44903db8c137fe29cf93ba2914e5886d7c0b90ee031aab834'
assert sha256(source)==expected
out=ROOT/'experiments/studies/cuda_jpeg_5000_v1_import_20261025'
assert not out.exists(),out
out.mkdir(parents=True)
with zipfile.ZipFile(source) as z:
 safe_members(z);assert z.testzip() is None;z.extractall(out/'transfer')
t=out/'transfer';plan=json.loads((t/'reviewed_plan.json').read_text())
assert sha256(t/'cuda_profile.json')==plan['cuda_profile']['sha256']
assert source_identity()['files']==plan['training_source_files']
assert sha256(ROOT/'notebooks/cloud_workflow.py')==plan['cloud_launcher_sha256']
for record in plan['manifests'].values():assert sha256(ROOT/record['path'])==record['sha256']
folders={'A':'A_seed42','B':'B_seed42','C':'C_seed42_resume_45934e2e'}
reports={};checks={};states={}
for arm,folder in folders.items():
 archive=t/folder/'step_005000_micro_00020000.zip'
 run=out/f'{arm}_seed42';restore_snapshot(archive,run)
 latest=load_checkpoint(run/'latest.pth');best=load_checkpoint(run/'best_model.pth')
 assert latest['step']==latest['total_steps']==5000
 assert best['step']==latest['best_step']=={'A':4700,'B':4400,'C':4900}[arm]
 assert best['best_metric']==latest['best_metric']
 assert best['compatibility']==latest['compatibility']
 expected_run=next(r for r in plan['runs'] if r['id']==f'{arm}_seed42')
 assert compatibility_hash(config_from_dict(expected_run['config']),best['datasets'])==best['compatibility']
 assert best['source']['files']==latest['source']['files']==plan['training_source_files']
 assert best['environment']['codec']==plan['cuda_environment']['codec']
 report=json.loads((t/'dev_replay_cpu_016f7784'/f'{arm}_dev.json').read_text())
 assert report['checkpoint_sha256']==sha256(run/'best_model.pth')
 assert report['step']==best['step'] and report['selection_horizon']==5000
 assert report['training_seed']==42 and report['split']=='val'
 assert report['count']==len(report['images'])==8 and report['phase_count']==16
 assert report['codec']==best['environment']['codec']
 assert all(len(row['phases'])==16 for row in report['images'])
 assert abs(sum(row['worst_phase_psnr'] for row in report['images'])/8-report['worst_phase_psnr'])<1e-12
 for row in report['images']:
  assert abs(min(v['psnr'] for v in row['phases'])-row['worst_phase_psnr'])<1e-12
  assert abs(sum(v['psnr'] for v in row['phases'])/16-row['psnr'])<1e-12
  assert abs(sum(v['ssim'] for v in row['phases'])/16-row['ssim'])<1e-12
  assert all(math.isfinite(v[k]) for v in row['phases'] for k in ('psnr','ssim'))
 logs=list(csv.DictReader((run/'metrics.csv').open()))
 selected=next(r for r in logs if int(r['step'])==best['step'] and r.get('worst_phase_psnr'))
 log_delta=report['worst_phase_psnr']-float(selected['worst_phase_psnr'])
 assert abs(log_delta)<1e-5,(arm,log_delta)
 reports[arm]=report;states[arm]=best
 checks[arm]={'completed_steps':latest['step'],'selected_step':best['step'],'selected_sha256':report['checkpoint_sha256'],'archive_sha256':sha256(archive),'logged_vs_cpu_replay_worst_phase_delta_db':log_delta,'count':8,'phase_count':16,'worst_phase_psnr':report['worst_phase_psnr'],'psnr':report['psnr'],'ssim':report['ssim'],'state_keys':list(latest.keys())}
 print(arm,json.dumps(checks[arm]),flush=True)
comparisons={}
for a,b in [('A','B'),('B','C'),('A','C')]:
 result=compare_phase_reports([reports[a]],[reports[b]])
 saved=json.loads((t/'dev_replay_cpu_016f7784'/f'{b}_minus_{a}.json').read_text())
 assert abs(result['delta_db']-saved['delta_db'])<1e-12
 assert all(abs(a-b)<1e-12 for a,b in zip(result['image_bootstrap_ci95_db'],saved['image_bootstrap_ci95_db']))
 comparisons[f'{b}_minus_{a}']=result
 print('COMPARE',b,a,result['delta_db'],result['image_bootstrap_ci95_db'])
test=read_manifest(ROOT/'data/manifests/div2k_jpeg_v1/screening/test.jsonl',ROOT)
assert len(test)==16
for state in states.values():validate_splits(state['data_identity']+test)
result={'status':'passed','import_source':str(source),'export_sha256':expected,'import_directory':str(out),'checks':checks,'comparisons':comparisons,'frozen_source_files_unchanged':True,'cloud_codec':plan['cuda_environment']['codec'],'local_codec':codec_metadata(),'test_source_integrity_and_disjointness':'16 protected images checked; no scoring','research_training_executed':False,'protected_test_scored':False,'evaluation_origin':'Imported Colab CPU inference reports independently reaggregated locally; local model inference not executed in this audit'}
(ROOT/'reports/studies/cuda_jpeg_5000_v1/import_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('AUDIT PASSED; no training or protected scoring.')
