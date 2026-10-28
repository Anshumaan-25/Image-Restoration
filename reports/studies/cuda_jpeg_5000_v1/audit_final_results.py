"""Import and independently verify the completed frozen phase evaluation export."""
from pathlib import Path
import hashlib
import json
import math
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'notebooks'))
from cloud_workflow import safe_members
from utils.reproducibility import source_identity
from utils.study import compare_phase_reports


def digest(data: bytes) -> str:
    """Return the SHA256 of an artifact's exact bytes."""
    return hashlib.sha256(data).hexdigest()


def near(a: float, b: float) -> None:
    """Require independently aggregated finite metrics to match."""
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) < 1e-11, (a,b)


def main() -> None:
    """Audit provenance, every phase, image aggregation and paired statistics."""
    source = Path('/Users/anshu/Downloads/cuda_jpeg_5000_v1/restoration_final_results_ba8dd3e1.zip')
    archive_hash = digest(source.read_bytes())
    assert archive_hash == 'd823f66325a7cf3951992173cd2d99e649dc125c10a04771a60c359e07818e31'
    with zipfile.ZipFile(source) as archive:
        safe_members(archive)
        assert archive.testzip() is None
        payload = {n: archive.read(n) for n in archive.namelist()}
    frozen_bytes = (Path(__file__).parent / 'frozen_study.json').read_bytes()
    assert payload['frozen_study.json'] == frozen_bytes
    frozen = json.loads(frozen_bytes)
    assert frozen['source_files'] == source_identity()['files']
    execution = json.loads(payload['execution.json'])
    assert execution['status'] == 'complete' and execution['jobs'] == 9
    assert execution['training'] is False and execution['spending_usd'] == 0
    assert execution['evaluation_precision'] == 'fp32' and execution['device'] == 'cuda'
    assert execution['frozen_study_sha256'] == digest(frozen_bytes)
    expected_jobs = {f'{arm}_Q{q}' for arm in 'ABC' for q in (20,10,40)}
    assert len(execution['completed']) == 9
    assert {item['job'] for item in execution['completed']} == expected_jobs
    for item in execution['completed']:
        assert digest(payload[item['job']+'.json']) == item['report_sha256']
        assert 0 < item['seconds'] <= execution['budget_seconds_per_job']
    environment = json.loads(payload['environment.json'])
    assert environment['device_name'] == 'Tesla T4'
    assert environment['dependencies']['Pillow'] == frozen['codec']['pillow']
    expected_ids = {f'{i:04d}' for i in range(801,817)}
    panel = [[u,v] for u in range(8) for v in range(8)]
    reports = {}
    summary = {}
    for quality in (20,10,40):
        for arm, selection in zip('ABC', frozen['selections']):
            name = f'{arm}_Q{quality}'
            report = json.loads(payload[name+'.json'])
            assert report['checkpoint_sha256'] == selection['sha256']
            assert report['step'] == selection['development_best_step']
            assert report['dataset_hash'] in frozen['test_manifest_hashes']
            assert report['frozen_study_sha256'] == digest(frozen_bytes)
            assert report['split'] == 'test' and report['device'] == 'cuda'
            assert report['codec'] == frozen['codec'] and report['quality'] == quality
            assert report['training_seed'] == 42 and report['selection_horizon'] == 5000
            assert report['count'] == len(report['images']) == 16
            assert {row['id'] for row in report['images']} == expected_ids
            assert report['phase_panel'] == panel and report['phase_count'] == 64
            for row in report['images']:
                phases = row['phases']
                assert len(phases) == row['count'] == 64
                assert [p['phase'] for p in phases] == panel
                for phase in phases:
                    assert all(math.isfinite(phase[k]) for k in ('psnr','ssim','input_psnr','input_ssim','edge_error','boundary_edge_error','interior_edge_error'))
                near(min(p['psnr'] for p in phases), row['worst_phase_psnr'])
                near(min(p['input_psnr'] for p in phases), row['input_worst_phase_psnr'])
                for metric, field in [('psnr','psnr'),('ssim','ssim'),('input_psnr','input_mean_psnr'),('input_ssim','input_mean_ssim')]:
                    near(sum(p[metric] for p in phases)/64, row[field])
                near(max(p['psnr'] for p in phases)-min(p['psnr'] for p in phases), row['phase_spread_psnr'])
            for metric in ('psnr','ssim','worst_phase_psnr','input_worst_phase_psnr','input_mean_psnr','input_mean_ssim','phase_spread_psnr'):
                near(sum(row[metric] for row in report['images'])/16, report[metric])
            reports[name] = report
            summary[name] = {k:report[k] for k in ('step','count','phase_count','worst_phase_psnr','psnr','ssim','checkpoint_sha256')}
    comparisons = {}
    for quality in (20,10,40):
        for baseline, candidate in [('A','B'),('B','C'),('A','C')]:
            name = f'{candidate}_minus_{baseline}_Q{quality}'
            saved = json.loads(payload[name+'.json'])
            actual = compare_phase_reports([reports[f'{baseline}_Q{quality}']], [reports[f'{candidate}_Q{quality}']])
            assert actual.keys() == saved.keys()
            near(actual['delta_db'], saved['delta_db'])
            for left,right in zip(actual['image_bootstrap_ci95_db'], saved['image_bootstrap_ci95_db']):
                near(left,right)
            for left,right in zip(actual['per_image'], saved['per_image']):
                assert left['id'] == right['id']
                near(left['delta_db'],right['delta_db'])
            comparisons[name] = actual
    destination = ROOT / 'experiments/studies/cuda_jpeg_5000_v1_final_import_20261026'
    destination.mkdir(exist_ok=True)
    for name, data in payload.items():
        target = destination / name
        if target.exists():
            assert target.read_bytes() == data, target
        else:
            target.write_bytes(data)
    result = {'status':'passed','archive':str(source),'archive_sha256':archive_hash,
              'archive_bytes':source.stat().st_size,'import_directory':str(destination),
              'frozen_source_files_unchanged':True,'jobs_verified':9,'images_per_job':16,
              'phases_per_image':64,'predictions_verified':9216,'summary':summary,
              'comparisons':comparisons,'execution':execution,'environment':environment,
              'audit_scope':'Existing exported results only; no inference, training or test reselection performed.',
              'limitations':['One training seed and sixteen held-out source images.','Image-bootstrap intervals are conditional on the observed training seed.','Ordinary full-image evaluation, qualitative exports and inference latency remain pending.','Consistency benefit and publishable novelty are not established.']}
    (Path(__file__).parent / 'final_results_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASSED: nine jobs, 9216 predictions, phase/image aggregates and nine paired bootstrap comparisons.')


if __name__ == '__main__':
    main()
