"""Package current verified inference with retained official and project provenance."""
from pathlib import Path
import hashlib,json,zipfile,io
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]

def sha(p: Path) -> str:
    """Hash one retained payload."""
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> None:
    """Build a new private local package, never overwrite historical releases."""
    acceptance=json.loads((ROOT/'reports/engineering_v2/acceptance_final_delivery_20261027.json').read_text())
    assert acceptance['status']=='passed' and acceptance['test_count']==68 and acceptance['test_skips']==0
    sidd=json.loads((ROOT/'reports/engineering_v2/sidd_full_results_audit_20261027.json').read_text());assert sidd['status']=='passed' and sidd['count']==1280
    files={name:ROOT/name for name in acceptance['source']['files']}
    for folder,pattern in [('tests/fixtures/upstream','*.txt'),('docs/third_party','*'),('requirements/locks','*.txt')]:
        for p in (ROOT/folder).glob(pattern):
            if p.is_file():files[str(p.relative_to(ROOT))]=p
    for name in ['requirements/optional.txt','README.md','AGENTS.md','docs/delivery.md','docs/task_runbook_v2.md','docs/architecture/approved_design.md','docs/architecture/approval_record.md','docs/architecture/implementation_handoff.md','docs/architecture/adr/0001-approved-restoration-design.md','notebooks/cloud_workflow.py']:
        files[name]=ROOT/name
    for p in (ROOT/'experiments/pretrained/approved_v2').glob('*'):
        if p.suffix in {'.pth','.json'}:files[('weights/' if p.suffix=='.pth' else 'provenance/')+p.name]=p
    files['weights/poisson_best.pth']=ROOT/'experiments/studies/poisson_completed_v3_import_20261027/run/best_model.pth'
    pilot=ROOT/'experiments/studies/cuda_jpeg_5000_v1_import_20261025'
    for arm in 'ABC':files[f'weights/jpeg_{arm}.pth']=pilot/f'{arm}_seed42/best_model.pth'
    evidence=['acceptance_final_delivery_20261027.json','poisson_completed_import_audit_20261027.json','poisson_actual_inference_verification.json','sidd_full_results_audit_20261027.json','sidd_fast_metrics_verification.json','sidd_full_optimized_v3_loop_verification.json','actual_remaining_pretrained_verification.json','actual_restormer_gaussian_verification.json','final_pretrained_replay_verification.json','sidd_subset40_results_audit.json','sidd_optimized_validation_approval_20261027.json','colab_sidd_full_validation_optimized_v3.py']
    for name in evidence:files['reports/engineering_v2/'+name]=ROOT/'reports/engineering_v2'/name
    for p in (ROOT/'reports/engineering_v2/final_evaluation').glob('*.json'):files[str(p.relative_to(ROOT))]=p
    for name in ['final_results_audit.json','completion_results_audit.json']:
        files['reports/jpeg/'+name]=ROOT/'reports/studies/cuda_jpeg_5000_v1'/name
    for p in (ROOT/'experiments/studies/sidd_full_v3_import_20261027').glob('*.json'):files['reports/sidd/'+p.name]=p
    for p in (ROOT/'experiments/studies/poisson_completed_v3_import_20261027/development_replay_91c0ef67').glob('*.json'):files['reports/poisson/'+p.name]=p
    files['samples/poisson_peak60.png']=ROOT/'outputs/examples_poisson_v3/poisson_peak60_input.png'
    files['samples/clean.png']=ROOT/'outputs/examples_poisson_v3/clean.png'
    for scale in [2,4]:files[f'samples/bird_x{scale}.png']=ROOT/f'data/raw/Set5_official_swinir/LR_bicubic/X{scale}/birdx{scale}.png'
    files['provenance/Set5_acquisition.json']=ROOT/'reports/engineering_v2/set5_acquisition.json'
    samples=ROOT/'outputs/examples_release_v3';samples.mkdir(exist_ok=False)
    clean=Image.open(files['samples/clean.png']).convert('RGB')
    buf=io.BytesIO();clean.save(buf,format='JPEG',quality=20,subsampling=0,optimize=False,progressive=False);buf.seek(0)
    Image.open(buf).convert('RGB').save(samples/'jpeg_q20.png');files['samples/jpeg_q20.png']=samples/'jpeg_q20.png'
    # Small authentic real-noise fixture for inference verification; no new quality scoring.
    pair=json.loads((ROOT/'data/processed/SIDD_validation_v2/val.jsonl').read_text().splitlines()[0])
    Image.open(ROOT/'data/processed/SIDD_validation_v2'/pair['input_path']).crop((0,0,33,35)).save(samples/'real_noise_crop.png')
    files['samples/real_noise_crop.png']=samples/'real_noise_crop.png'
    readme='''# Restoration engineering v3 — start here

Verified approved engineering scope; no established novelty/SOTA claim. No training or downloads are necessary for the commands below. Use an existing compatible Python/PyTorch environment (see requirements/locks); CPU works on all platforms. Replace cpu with mps on Apple Silicon or cuda on an allocated NVIDIA runtime. Choose a fresh output path each time.

From this extracted folder:

```bash
python train.py --dry-run --epochs 1 --batch-size 2 --device cpu
python inference.py --input samples/jpeg_q20.png --output outputs/jpeg.png --weights weights/jpeg_B.pth --device cpu
python inference.py --input samples/poisson_peak60.png --output outputs/poisson.png --weights weights/poisson_best.pth --device cpu
python inference.py --input samples/real_noise_crop.png --output outputs/real.png --weights weights/real_denoising.pth --pretrained-id restormer_sidd --weights-sha256 4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387 --device cpu
python inference.py --input samples/bird_x2.png --output outputs/sr_x2.png --weights weights/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --pretrained-id swinir_x2 --weights-sha256 193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7 --device cpu
python inference.py --input samples/bird_x4.png --output outputs/sr_x4.png --weights weights/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth --pretrained-id swinir_x4 --weights-sha256 09fad24e32ae62722e1a055efde9921328f4137981bab0a42a4a3a806306c58e --device cpu
```

Gaussian restoration uses weights/gaussian_color_denoising_blind.pth with --pretrained-id restormer_gaussian and --weights-sha256 4e2314dae77bb907184da6e2ac625b921491d6b33de26a52512eff1c6bb37fbd. Supply an RGB noisy image. JPEG A and C selections are included for study inspection; B is the example command, not a newly reselected checkpoint. Poisson selection is4500/5000; its included results are development-only. Real-noise crop is only a tiny inference fixture, not the full SIDD score panel. Set5 images come from the pinned official SwinIR repository; source provenance and model licenses are included. Full1,280-block SIDD scoring uses the declared project RGB protocol and official supervised pretrained weights.

RELEASE_INVENTORY.json records all payloads. Retain licenses and provenance. Full datasets are intentionally absent; original acquisition and long-run approvals are not permissions to restart training. Source tests/configs are included; evaluation reports retain their original paths/environments as historical provenance. See docs/delivery.md for results and limitations. No submitted report is included or modified. The existing checkpoint configs may refer to the original dataset location; inference does not require that dataset. Historical exact JPEG research replay requires its separately preserved61-file source/codec, not this current84-file source.
'''
    notes=ROOT/'reports/engineering_v2/RELEASE_START_HERE.md';notes.write_text(readme);files['START_HERE.md']=notes
    inventory={name:{'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in sorted(files.items())}
    for name,digest in acceptance['source']['files'].items():assert inventory[name]['sha256']==digest
    archive=ROOT/'dist/restoration_engineering_v3.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,p in sorted(files.items()):z.write(p,'restoration_engineering_v3/'+name)
        z.writestr('restoration_engineering_v3/RELEASE_INVENTORY.json',json.dumps(inventory,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,row in inventory.items():
            data=z.read('restoration_engineering_v3/'+name);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    result={'status':'packaged_integrity_verified_standalone_pending','artifact':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'payload_files':len(inventory),'source_files':84,'weights':{name:row for name,row in inventory.items() if name.startswith('weights/')},'inventory':inventory,'scope':'Approved engineering delivery and bounded measured experiments; novelty not established'}
    (ROOT/'reports/engineering_v2/engineering_release_v3.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in {'inventory','weights'}},indent=2))

if __name__=='__main__':main()
