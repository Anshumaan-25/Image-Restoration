"""Package verified pretrained inference without claiming pending research complete."""
from pathlib import Path
import hashlib,json,zipfile
root=Path.cwd();destination=root/'dist/restoration_pretrained_v2.zip'
if destination.exists():raise FileExistsError(destination)
manifest=json.loads((root/'reports/engineering_v2/cloud_extension_manifest.json').read_text())
files=set(manifest['source_files'])
files.update(str(p.relative_to(root)) for p in (root/'tests/fixtures/upstream').glob('*.txt'))
files.update(str(p.relative_to(root)) for p in (root/'docs/third_party').glob('*') if p.is_file())
files.update(['README.md','AGENTS.md','docs/task_runbook_v2.md','docs/architecture/approved_design.md','docs/architecture/approval_record.md','docs/architecture/implementation_handoff.md','notebooks/cloud_workflow.py'])
files.update(str(p.relative_to(root)) for p in (root/'experiments/pretrained/approved_v2').glob('*') if p.suffix in {'.pth','.json'})
files.update(str(p.relative_to(root)) for p in (root/'reports/engineering_v2/final_evaluation').glob('*.json'))
files.update(['reports/engineering_v2/final_pretrained_replay_verification.json','reports/engineering_v2/actual_remaining_pretrained_verification.json','reports/engineering_v2/actual_restormer_gaussian_verification.json','reports/engineering_v2/acceptance_extension_v2.json','reports/engineering_v2/research_completion_decision.md','data/raw/Set5_official_swinir/LR_bicubic/X2/birdx2.png'])
inventory={name:{'bytes':(root/name).stat().st_size,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in sorted(files)}
readme='''# Verified supervised pretrained inference v2\n\nThis kit contains four explicitly approved official supervised checkpoints: Restormer Gaussian and real denoising, SwinIR lightweight SR x2/x4. They are pretrained comparisons, not new project-trained methods. No GAN weights are included. SIDD real-data scoring and natural-image Poisson training remain pending; scientific novelty is not established. The completed project-trained JPEG pilot remains in its separately frozen v1 kit.\n\nUse an existing compatible Python/PyTorch environment; this package never downloads weights automatically. Run from the extracted folder:\n\n```bash\npython inference.py --input data/raw/Set5_official_swinir/LR_bicubic/X2/birdx2.png --output example_sr.png --weights experiments/pretrained/approved_v2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --pretrained-id swinir_x2 --weights-sha256 193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7 --device cpu\n```\n\nThe included sample is from the pinned official SwinIR repository. Other input images/datasets are not bundled. Every payload hash is in RELEASE_INVENTORY.json. Retain the included license files and acquisition receipts. The full project README refers to datasets/experiments outside this lightweight inference kit; use docs/task_runbook_v2.md for supported-model/protocol details. Official releases supplied no publisher SHA256; local receipt digests verify retained file integrity.\n'''
with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for name in sorted(files):z.write(root/name,'restoration_pretrained_v2/'+name)
 z.writestr('restoration_pretrained_v2/START_HERE.md',readme)
 z.writestr('restoration_pretrained_v2/RELEASE_INVENTORY.json',json.dumps(inventory,indent=2)+'\n')
with zipfile.ZipFile(destination) as z:
 assert z.testzip() is None
 for name,record in inventory.items():
  data=z.read('restoration_pretrained_v2/'+name)
  assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
result={'status':'packaged_integrity_verified','artifact':str(destination),'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'payload_files':len(inventory),'scope':'Pretrained inference kit, not full scientific project completion','inventory':inventory}
(root/'reports/engineering_v2/pretrained_release.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='inventory'},indent=2))
