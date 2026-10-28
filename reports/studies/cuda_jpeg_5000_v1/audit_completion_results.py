"""Audit completed GPU full-image, visual and latency artifacts without inference."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'notebooks'))
from cloud_workflow import safe_members
from utils.reproducibility import source_identity
from PIL import Image
import numpy as np


def sha(data: bytes) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(data).hexdigest()


def near(a: float,b: float) -> None:
    """Fail on nonfinite or mismatched metric aggregates."""
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-11,(a,b)


def pixels(path: Path) -> np.ndarray:
    """Decode verified RGB PNG pixels without modifying source files."""
    with Image.open(path) as image:
        assert image.mode=='RGB' and image.format=='PNG'
        return np.array(image)


def main() -> None:
    """Validate archive inventory, frozen provenance, metrics and exact visual crops."""
    base=Path(__file__).parent
    source=Path('/Users/anshu/Downloads/restoration_completion_checks_v1_cuda_index_fix.zip')
    archive_sha=sha(source.read_bytes())
    assert archive_sha=='40ed8add2b269bccab07884629b9d52dcce457698076870ced79332275489967'
    destination=ROOT/'experiments/studies/cuda_jpeg_5000_v1_completion_import_20261026'
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        safe_members(archive)
        assert archive.testzip() is None
        inventory=json.loads(archive.read('INVENTORY.json'))
        assert set(archive.namelist())==set(inventory)|{'INVENTORY.json'}
        for name,record in inventory.items():
            data=archive.read(name)
            assert len(data)==record['bytes'] and sha(data)==record['sha256'],name
            path=destination/name
            path.parent.mkdir(exist_ok=True,parents=True)
            if path.exists():assert path.read_bytes()==data
            else:path.write_bytes(data)
    read=lambda name:json.loads((destination/name).read_text())
    frozen=read('frozen_study.json')
    assert (destination/'frozen_study.json').read_bytes()==(base/'frozen_study.json').read_bytes()
    assert frozen['source_files']==source_identity()['files']
    manifest=[json.loads(line) for line in (ROOT/'data/manifests/div2k_jpeg_v1/screening/test.jsonl').read_text().splitlines()]
    assert sha((ROOT/'data/manifests/div2k_jpeg_v1/screening/test.jsonl').read_bytes())==frozen['test_manifest_file_sha256']
    for row in manifest:
        assert sha((ROOT/row['target_path']).read_bytes())==row['sha256']
    records={row['id']:row for row in manifest}
    execution=read('execution.json')
    assert execution['status']=='complete' and execution['training'] is False and execution['spending_usd']==0
    assert execution['device']=='cuda' and execution['precision']=='fp32'
    assert execution['frozen_study_sha256']==sha((destination/'frozen_study.json').read_bytes())
    assert execution['launcher_sha256']==sha((destination/'completion_launcher.py').read_bytes())
    assert (base/'colab_completion_checks.py').read_bytes()==(destination/'completion_launcher.py').read_bytes()
    expected_jobs={(a,k) for a in 'ABC' for k in ('full','benchmark')}
    assert len(execution['completed'])==6
    assert {(job['arm'],job['kind']) for job in execution['completed']}==expected_jobs
    for job in execution['completed']:
        report=f"{job['arm']}/full_image.json" if job['kind']=='full' else f"{job['arm']}_latency.json"
        assert job['report_sha256']==sha((destination/report).read_bytes())
        assert 0<job['seconds']<=execution['full_image_cap_seconds_per_arm' if job['kind']=='full' else 'benchmark_cap_seconds_per_arm']
    summary=read('summary.json');results={};all_images={}
    for arm,selection in zip('ABC',frozen['selections']):
        report=read(f'{arm}/full_image.json');latency=read(f'{arm}_latency.json');progress=read(f'{arm}/progress.json')
        assert summary['full_image'][arm]==report and summary['latency'][arm]==latency
        assert report['checkpoint_sha256']==selection['sha256']
        assert sha(Path(selection['path']).read_bytes())==selection['sha256']
        assert report['step']==selection['development_best_step']
        assert report['count']==len(report['images'])==16
        assert {row['id'] for row in report['images']}==set(records)
        assert progress['status']=='complete' and progress['count']==progress['expected_count']==16
        assert progress['images']==report['images']
        assert report['codec']==frozen['codec'] and report['dataset_hash'] in frozen['test_manifest_hashes']
        assert report['reference_crop']=='full_image' and report['quality']==20 and report['border']==0
        assert report['precision']=='fp32' and report['tile_mode']=='full' and report['metric_quantization'] is False
        assert report['model_parameters']==669379 and report['training'] is False
        for metric in ('psnr','ssim','input_psnr','input_ssim'):
            near(sum(row[metric] for row in report['images'])/16,report[metric])
        all_images[arm]={row['id']:row for row in report['images']}
        for row in report['images']:
            reference=records[row['id']]
            assert row['shape']==[3,reference['height'],reference['width']]
            assert math.isfinite(row['forward_seconds']) and row['forward_seconds']>0
            if arm!='A':
                for key in ('input_psnr','input_ssim'):near(row[key],all_images['A'][row['id']][key])
        assert 0<report['peak_cuda_allocated_bytes']<=report['peak_cuda_reserved_bytes']<=12*1024**3
        assert latency['completed'] and not latency['random_weights'] and not latency['restoration_accuracy_evaluated']
        assert latency['precision']=='fp32' and latency['batch_size']==1 and latency['warmup']==10
        assert latency['environment']['device_name']=='Tesla T4'
        assert latency['requested_measured_passes']==30 and latency['model_parameters']==669379
        assert len(latency['results'])==2
        for size,row in zip((256,512),latency['results']):
            assert row['input_shape']==[1,3,size,size] and row['measured_passes']==30
            assert 0<row['median_seconds']<=row['p95_seconds']
        for image_id in ('0801','0802','0803'):
            for kind in ('clean','compressed','restored'):
                full=pixels(destination/arm/f'{image_id}_{kind}.png')
                h,w=full.shape[:2];size=min(256,h,w);top=(h-size)//2;left=(w-size)//2
                assert full.shape==tuple([records[image_id]['height'],records[image_id]['width'],3])
                assert np.array_equal(full[top:top+size,left:left+size],pixels(destination/arm/f'{image_id}_{kind}_center.png'))
                if kind=='clean':assert np.array_equal(full,pixels(ROOT/records[image_id]['target_path']))
                if arm!='A' and kind in ('clean','compressed'):
                    assert np.array_equal(full,pixels(destination/'A'/f'{image_id}_{kind}.png'))
        results[arm]={'psnr':report['psnr'],'ssim':report['ssim'],
                      'input_psnr':report['input_psnr'],'gain_over_input_db':report['psnr']-report['input_psnr'],
                      'peak_full_image_allocated_GiB':report['peak_cuda_allocated_bytes']/1024**3,
                      'peak_full_image_reserved_GiB':report['peak_cuda_reserved_bytes']/1024**3,
                      'latency_ms':{str(row['input_shape'][-1]):{'median':1000*row['median_seconds'],'p95':1000*row['p95_seconds']} for row in latency['results']}}
    for image_id in ('0801','0802','0803'):
        panel=pixels(destination/f'{image_id}_comparison.png')
        paths=[destination/'A'/f'{image_id}_{kind}_center.png' for kind in ('clean','compressed')]
        paths += [destination/arm/f'{image_id}_restored_center.png' for arm in 'ABC']
        for index,path in enumerate(paths):
            image=pixels(path);h,w=image.shape[:2]
            assert panel.shape==(h+32,w*5,3)
            assert np.array_equal(panel[32:,index*w:(index+1)*w],image)
    audit={'status':'passed','archive_sha256':archive_sha,'archive_bytes':source.stat().st_size,
           'verified_inventory_files':len(inventory),'import_directory':str(destination),
           'full_image_predictions':48,'fixed_visual_examples':3,'png_files_checked':57,
           'benchmark_passes_per_arm':60,'results':results,
           'child_wall_seconds':sum(job['seconds'] for job in execution['completed']),
           'frozen_source_files_unchanged':True,'new_inference_or_training_executed_by_audit':False,
           'visual_review':'Pixel alignment verified automatically; human-readable panel inspection recorded separately.',
           'limitations':['One seed and sixteen held-out images; no general benchmark or novelty claim.','Latency is synthetic model-only, not end-to-end application throughput.','Tiny speed differences between identical architectures are not algorithmic speedups.','SR and real/synthetic denoising research experiments and transformer comparators remain staged.']}
    (base/'completion_results_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    with (base/'full_image_results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['arm','count','psnr_db','ssim','input_psnr_db','gain_db','peak_allocated_GiB','peak_reserved_GiB'])
        for arm,row in results.items():writer.writerow([arm,16,row['psnr'],row['ssim'],row['input_psnr'],row['gain_over_input_db'],row['peak_full_image_allocated_GiB'],row['peak_full_image_reserved_GiB']])
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
