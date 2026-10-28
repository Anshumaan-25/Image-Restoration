"""Package acquired test references and frozen selections without scoring them."""
from pathlib import Path
import sys,json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from utils.io import file_sha256
from utils.dataset_loader import read_manifest
base=ROOT/'reports/studies/cuda_jpeg_5000_v1'
frozen=base/'frozen_study.json'
manifest=ROOT/'data/manifests/div2k_jpeg_v1/screening/test.jsonl'
rows=read_manifest(manifest,ROOT)
assert [r['source_id'] for r in rows]==[f'{i:04d}' for i in range(801,817)]
files=[manifest,frozen]+[ROOT/r['target_path'] for r in rows]
output=ROOT/'dist/colab_frozen_test16_v1.zip'
assert not output.exists()
inv={'schema_version':1,'purpose':'protected_test16_images_and_frozen_selections_no_scoring','test_count':16,'source_ids':[r['source_id'] for r in rows],'frozen_study_sha256':file_sha256(frozen),'files':{str(p.relative_to(ROOT)):{'sha256':file_sha256(p),'bytes':p.stat().st_size} for p in files}}
with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_STORED) as z:
 for p in files:z.write(p,str(p.relative_to(ROOT)))
 z.writestr('FINAL_EVAL_BUNDLE.json',json.dumps(inv,indent=2)+'\n')
with zipfile.ZipFile(output) as z:
 assert z.testzip() is None
 for name,record in inv['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==record['sha256']
proposal={'status':'prepared_not_authorized_not_executed','device':'cpu','model_precision':'fp32','spending_usd':0,'new_downloads':False,'new_training':False,'dataset':'DIV2K project held-out IDs0801–0816;16 source-disjoint images','selection':'A4700 B4400 C4900; fixed by verified development best','qualities_in_order':[20,10,40],'phase_panel':'all64','primary':'paired mean per-image worst-phase PSNR; C-A combined effect and C-B added consistency; B-A sampling control','secondary':['mean RGB PSNR/SSIM','input references','phase spread','boundary and interior edge error'],'runs':9,'model_predictions':16*64*9,'seconds_per_arm_quality':600,'max_evaluation_seconds_total':5400,'limits':'90 minutes upper execution allowance plus setup/checkpoint loading/saving; not a measured estimate. Stop on any error/timeout; no automatic retry, model selection or tuning from test outcomes.','environment':'Colab CPU with Pillow11.3.0 and JPEG6.2 matching frozen codec; CPU environment recorded separately','bundle':str(output.relative_to(ROOT)),'bundle_bytes':output.stat().st_size,'bundle_sha256':file_sha256(output),'frozen_study_sha256':file_sha256(frozen),'full_image_test':'not included in this phase-panel invocation; separate remaining ordinary-quality check','no_significance_guarantee':True}
(base/'final_test_proposal.json').write_text(json.dumps(proposal,indent=2)+'\n')
output.with_suffix('.json').write_text(json.dumps({'bundle':output.name,'bytes':output.stat().st_size,'sha256':file_sha256(output),'test_count':16,'scored':False},indent=2)+'\n')
print(json.dumps({'bundle':str(output),'MiB':output.stat().st_size/1024**2,'sha256':file_sha256(output),'scored':False},indent=2))
