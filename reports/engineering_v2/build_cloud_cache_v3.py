"""Preserve failed v2 profile; seal identical Poisson samples with bounded CPU caching."""
from pathlib import Path
import hashlib,json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from utils.reproducibility import source_identity
from utils.io import file_sha256
stage=ROOT/'dist/Image_Restoration_Extension_v3';archive=ROOT/'dist/colab_extension_v3.zip'
if stage.exists() or archive.exists():raise FileExistsError('Preserve existing cache release')
stage.mkdir(parents=True);sources=source_identity()['files'];paths=set(sources)|{'notebooks/cloud_workflow.py'}
paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'tests/fixtures/upstream').glob('*.txt'))
for name in ['data/manifests/div2k_jpeg_v1/screening/train.jsonl','data/manifests/poisson_v2/val.jsonl']:
 paths.add(name)
 for line in (ROOT/name).read_text().splitlines():
  row=json.loads(line)
  for key in ['input_path','target_path']:
   if row.get(key):paths.add(row[key])
inventory={}
for name in sorted(paths):
 source=ROOT/name;target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 inventory[name]={'bytes':target.stat().st_size,'sha256':file_sha256(target)}
(stage/'EXTENSION_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(stage.rglob('*')):
  if p.is_file():z.write(p,str(Path(stage.name)/p.relative_to(stage)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
assert sources==source_identity()['files'];digest=file_sha256(archive)
base=ROOT/'reports/engineering_v2';launcher=base/'colab_poisson_launcher_v3.py'
text=(base/'colab_poisson_launcher.py').read_text().replace('f3665a079fb7c028244a8df6cbf4b6f9bb1808834bb1ad6e15fd9e944550024b',digest).replace('colab_extension_v2.zip','colab_extension_v3.zip').replace('Image_Restoration_Extension_v2','Image_Restoration_Extension_v3').replace('poisson_cloud_v2','poisson_cloud_v3').replace('poisson_rgb_5000_v2','poisson_rgb_5000_v3_cache')
launcher.write_text(text)
result={'status':'sealed_pending_cuda_profile','bundle':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'launcher':str(launcher),'launcher_sha256':file_sha256(launcher),'source_files':sources,'study':'poisson_rgb_5000_v3_cache','change':'Only optional 1536 MiB per-dataset CPU decoded-reference LRU; exact tensor/metadata/RNG parity tested. Same 32 sources, crops, photon levels, model, L1, seed, microbatch/accumulation, 5000 updates, 5400-second cap and profile gates. Defaults remain uncached. Old source/bundle/profile retained.','protected_test_payload':False,'spending_usd':0}
(base/'cloud_cache_v3_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source_files'},indent=2))
