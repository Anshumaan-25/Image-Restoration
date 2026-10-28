"""Explicitly approved, bounded SIDD public-validation scoring; no training/downloads."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,uuid,zipfile
from typing import Any,Callable

ARCHIVE_SHA='1139c32896697e960159ae845070751e58894ddf607ebfe8f6d83ef012e2fbff'
WEIGHT_SHA='4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387'
MAT_SHA={'ValidationNoisyBlocksSrgb.mat':'850e91ac1426fd953cfe50a12b8e43faca41081dc68d2e0701faa85d3fd62624','ValidationGtBlocksSrgb.mat':'fe17438ff6c25e4888260613c9dbaae485c2a90db46808471f33fde661f1434f'}


def sha(path: Path) -> str:
    """Stream a local approved artifact digest."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def score_all(score: Callable[[int],dict[str,Any]], count: int, destination: Path, max_seconds: float = 1200) -> None:
    """Save bounded per-image scoring atomically; partial runs never become complete."""
    from utils.metrics import aggregate
    from utils.logger import write_json
    rows=[];started=time.monotonic()
    def save(status: str,error: str | None = None) -> None:
        record={'status':status,'count':len(rows),'expected_count':count,'images':rows,'error':error,'elapsed_seconds':time.monotonic()-started,'training':False}
        if rows:record.update(aggregate(rows));record['reference']=aggregate([r['reference'] for r in rows])
        temporary=destination/('progress_'+uuid.uuid4().hex+'.partial');write_json(temporary,record);os.replace(temporary,destination/'progress.json')
    try:
        for index in range(count):
            if time.monotonic()-started>=max_seconds:raise TimeoutError('SIDD evaluation cap reached')
            rows.append(score(index))
            if len(rows)%32==0:save('incomplete');print(f'SIDD public validation: {len(rows)}/{count}; progress retained in Drive',flush=True)
        if time.monotonic()-started>max_seconds:raise TimeoutError('SIDD evaluation exceeded cap')
        save('complete')
        import shutil
        shutil.copy2(destination/'progress.json',destination/'results.json')
    except BaseException as error:
        save('incomplete',repr(error));raise


def main() -> None:
    """Profile before full FP32 scoring; retain partial per-image results on failure."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--approved-full-validation',action='store_true');args=parser.parse_args()
    if not args.approved_full_validation:raise PermissionError('Specific full SIDD1280 validation and data/weight transfer approval required')
    if not Path('/content/drive').is_mount():raise RuntimeError('Drive backup mount is required')
    project=Path('/content/Image_Restoration_Extension_v3');os.chdir(project);sys.path[:0]=[str(project),str(project/'notebooks')]
    drive=Path('/content/drive/MyDrive/RestorationProject');study=drive/'studies/poisson_rgb_5000_v3_cache'
    complete=json.loads((project/'experiments/cloud/poisson_rgb_5000_v3_cache/poisson_seed42/summary.json').read_text())
    if not complete.get('completed') or complete.get('step')!=5000:raise RuntimeError('Poisson run must finish before this separate evaluation')
    from utils.reproducibility import source_identity
    expected=json.loads((study/'reviewed_plan.json').read_text())['training_source_files']
    if source_identity()['files']!=expected:raise ValueError('Frozen runtime source changed')
    archive=drive/'test.zip';weights=drive/'real_denoising.pth'
    if archive.stat().st_size!=461780640 or sha(archive)!=ARCHIVE_SHA:raise ValueError('Approved SIDD archive integrity mismatch')
    if sha(weights)!=WEIGHT_SHA:raise ValueError('Approved official supervised weight integrity mismatch')
    from cloud_workflow import safe_members
    uid=uuid.uuid4().hex[:8];destination=drive/'studies'/('sidd_public_validation_full_v1_'+uid);destination.mkdir(parents=True,exist_ok=False)
    local=project/'data'/('sidd_full_'+uid);mats=local/'mat';mats.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        entries=safe_members(z)
        if z.testzip() is not None:raise ValueError('SIDD ZIP CRC failure')
        for name,digest in MAT_SHA.items():
            matches=[e for e in entries if Path(e.filename).name==name]
            if len(matches)!=1 or matches[0].file_size>512*1024**2:raise ValueError('Unexpected MAT inventory/size')
            import shutil
            with z.open(matches[0]) as src,(mats/name).open('xb') as out:shutil.copyfileobj(src,out)
            if sha(mats/name)!=digest:raise ValueError('MAT digest differs from acquired public-validation receipt')
    from utils.sidd_blocks import prepare_sidd_blocks
    pairs=local/'pairs';receipt=prepare_sidd_blocks(mats/'ValidationNoisyBlocksSrgb.mat',mats/'ValidationGtBlocksSrgb.mat',pairs,'nkhwc')
    if receipt['count']!=1280:raise ValueError('Expected all 1280 SIDD public-validation pairs')
    import torch
    from utils.pretrained import load_pretrained
    from utils.dataset_loader import RestorationDataset
    from utils.config import config_from_dict
    from utils.runtime import environment
    from utils.metrics import evaluate_pair,aggregate
    from utils.logger import write_json
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('Explicit CUDA required; no fallback')
    device=torch.device('cuda:0');total=torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,12*1024**3/total),0)
    model,cfg,weight_receipt=load_pretrained('restormer_sidd',weights,WEIGHT_SHA);model=model.to(device).eval()
    cfg.dataset.root=str(pairs);cfg.dataset.val_manifest='val.jsonl';cfg=config_from_dict(cfg.to_dict());dataset=RestorationDataset(cfg,'val')
    manifest=pairs/'val.jsonl'
    import shutil
    shutil.copy2(manifest,destination/'validation_manifest.jsonl')
    plan={'status':'fixed_before_full_scoring','expected_count':1280,'split':'public_validation','model_origin':'official_supervised_pretrained','weights':weight_receipt,'archive_sha256':ARCHIVE_SHA,'manifest_sha256':sha(manifest),'dataset_fingerprint':dataset.fingerprint,'source':source_identity(),'launcher_sha256':sha(Path(__file__)),'environment':environment(device),'precision':'fp32','batch_size':1,'tile_mode':'full_256_block','protocol':'project_rgb_float_v1','ssim_protocol':'valid_gaussian11_sigma1.5','metric_channel':'RGB','metric_border':0,'metric_quantization':False,'max_evaluation_seconds':1200,'max_allocated_GiB':12,'spending_usd':0,'training':False,'official_hidden_benchmark':False,'checkpoint_selection':'Fixed official real-denoising asset; no tuning or reselection'}
    write_json(destination/'frozen_plan.json',plan)
    def score(index: int) -> dict:
        item=dataset[index]
        with torch.inference_mode():prediction=model(item['input'][None].to(device)).float().cpu()[0]
        return {'id':item['id'],**evaluate_pair(prediction,item['target']),'reference':evaluate_pair(item['input'],item['target'])}
    torch.cuda.reset_peak_memory_stats(device);profile_start=time.monotonic()
    with torch.inference_mode():
        for _ in range(3):model(dataset[0]['input'][None].to(device))
    torch.cuda.synchronize(device);timings=[]
    for index in range(10):
        began=time.monotonic();score(index);timings.append(time.monotonic()-began)
        if time.monotonic()-profile_start>60:raise TimeoutError('SIDD preflight exceeded 60 seconds; no full scoring')
    measured={'measured_blocks':10,'warmup_forwards':3,'mean_block_seconds':sum(timings)/len(timings),'projected_full_seconds':1280*sum(timings)/len(timings),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'timing_scope':'paired PNG decode, FP32 forward, transfer, restoration and input CPU float64 metrics'}
    write_json(destination/'cuda_profile.json',measured)
    if measured['projected_full_seconds']>=960 or measured['peak_allocated_bytes']>12*1024**3:raise RuntimeError('SIDD profile failed the fixed timing/memory gate; no full scoring')
    score_all(score,1280,destination,1200)
    print('SIDD FULL PUBLIC VALIDATION COMPLETE:',destination,flush=True)


if __name__=='__main__':main()
