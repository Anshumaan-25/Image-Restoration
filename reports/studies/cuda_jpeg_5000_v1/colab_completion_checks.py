"""Bounded CUDA full-image checks, fixed visual examples and model-only benchmarks."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, shutil, stat, subprocess, sys, time, uuid, zipfile


def digest(path: Path) -> str:
    """Stream an artifact SHA256."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Reject unsafe members before extracting any payload."""
    seen=set();items=archive.infolist()
    for e in items:
        p=PurePosixPath(e.filename)
        if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in e.filename or e.filename in seen or stat.S_ISLNK(e.external_attr>>16):
            raise ValueError('Unsafe ZIP member')
        seen.add(e.filename)
    if archive.testzip() is not None:raise ValueError('ZIP CRC failure')
    return items



def strict_metrics(value):
    """Represent perfect PSNR as null plus a flag; reject other nonfinite values."""
    import math
    if isinstance(value,list):return [strict_metrics(item) for item in value]
    if not isinstance(value,dict):return value
    result={}
    for key,item in value.items():
        if isinstance(item,float) and not math.isfinite(item):
            if item==float('inf') and key.endswith('psnr'):
                result[key]=None;result[key+'_is_infinite']=True
            else:raise ValueError('Nonfinite non-PSNR result')
        else:result[key]=strict_metrics(item)
    return result


def evaluate_full(model, dataset, device, destination: Path, export_ids: set[str]) -> dict:
    """Evaluate each uncropped image once, save progress and fixed qualitative examples."""
    import torch
    from utils.metrics import evaluate_pair, aggregate
    from utils.io import save_rgb
    from utils.runtime import synchronize
    model.eval()
    destination.mkdir(parents=True, exist_ok=False)
    rows=[]
    with torch.no_grad():
        for index in range(len(dataset)):
            item=dataset[index]
            image_id=item['id']
            if not image_id.isalnum():raise ValueError('Unsafe image identity')
            x=item['input'][None].to(device)
            synchronize(device);start=time.monotonic()
            prediction=model(x)
            synchronize(device);elapsed=time.monotonic()-start
            if prediction.shape!=x.shape or not torch.isfinite(prediction).all():
                raise ValueError('Invalid full-image restoration')
            prediction=prediction.float().cpu()[0]
            del x
            score=evaluate_pair(prediction,item['target'],0)
            reference=evaluate_pair(item['input'],item['target'],0)
            strict_metrics({'psnr':score['psnr'],'ssim':score['ssim'],
                            'input_psnr':reference['psnr'],'input_ssim':reference['ssim']})
            row={'id':image_id,'shape':list(item['target'].shape),'forward_seconds':elapsed,
                 **score,'input_psnr':reference['psnr'],'input_ssim':reference['ssim']}
            rows.append(row)
            if image_id in export_ids:
                for name,tensor in [('clean',item['target']),('compressed',item['input']),('restored',prediction)]:
                    save_rgb(tensor,destination/f'{image_id}_{name}.png')
                    h,w=tensor.shape[-2:];size=min(256,h,w);top=(h-size)//2;left=(w-size)//2
                    save_rgb(tensor[:,top:top+size,left:left+size],destination/f'{image_id}_{name}_center.png')
            progress={'status':'running','count':len(rows),'expected_count':len(dataset),'images':rows}
            progress_path=destination/'progress.json'
            temporary=destination/'progress.pending.json'
            temporary.write_text(json.dumps(strict_metrics(progress),indent=2,allow_nan=False)+'\n');os.replace(temporary,progress_path)
            print(f"Saved image {image_id}: {len(rows)}/{len(dataset)}; PSNR={score['psnr']:.6f}",flush=True)
            del prediction
    result={**aggregate(rows),'images':rows,'input_psnr':sum(r['input_psnr'] for r in rows)/len(rows),
            'input_ssim':sum(r['input_ssim'] for r in rows)/len(rows),'status':'complete',
            'reference_crop':'full_image','quality':20,'protocol':'project_rgb_float_v1',
            'ssim_protocol':'project_ssim_v1','border':0,'clamp':'metric/export boundary only',
            'metric_quantization':False,'precision':'fp32','tile_mode':'full','batch_size':1,
            'export_ids':sorted(export_ids),'visual_crop':'fixed central up-to-256 square; no quality-based selection',
            'forward_timing_scope':'synchronized first-pass model forward; excludes transfer, I/O and CPU metrics; no warmup',
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}
    progress['status']='complete'
    temporary.write_text(json.dumps(strict_metrics(progress),indent=2,allow_nan=False)+'\n');os.replace(temporary,progress_path)
    return strict_metrics(result)


def worker(args) -> None:
    """Run one guarded CUDA job with a strict allocator limit and no fallback."""
    project=Path(args.project);os.chdir(project);sys.path.insert(0,str(project))
    import torch
    from utils.runtime import select_device
    from utils.checkpoint import load_checkpoint
    from utils.config import config_from_dict
    from utils.reproducibility import source_identity
    from utils.degradation import codec_metadata
    from utils.dataset_loader import RestorationDataset,validate_splits
    from models import build_model
    torch.set_num_threads(2)
    device=select_device('cuda')
    total=torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0,12*1024**3/total),torch.cuda.current_device() if device.index is None else device.index)
    frozen=project/'reports/studies/cuda_jpeg_5000_v1/frozen_study.json'
    f=json.loads(frozen.read_text())
    if digest(frozen)!='400aa655d20b23a41c065a190811e38a52762e8a9ab9e6bc4c87e94bb0d52fb8':raise ValueError('Frozen protocol changed')
    selection=next(s for s in f['selections'] if s['run_id']==f'{args.arm}_seed42')
    if digest(Path(args.checkpoint))!=selection['sha256']:raise ValueError('Frozen weights changed')
    if source_identity()['files']!=f['source_files'] or codec_metadata()!=f['codec']:raise ValueError('Frozen source or codec changed')
    destination=Path(args.destination)
    if args.worker=='benchmark':
        import runpy
        sys.argv=['scripts/benchmark.py','--weights',args.checkpoint,'--device','cuda','--precision','fp32',
                  '--sizes','256','512','--warmup','10','--iterations','30','--max-seconds','120',
                  '--output',str(destination)]
        runpy.run_path(str(project/'scripts/benchmark.py'),run_name='__main__')
        return
    state=load_checkpoint(args.checkpoint)
    if state.get('profile_only') or state['step']!=selection['development_best_step'] or state['total_steps']!=5000:
        raise ValueError('Unexpected selected checkpoint state')
    cfg=config_from_dict(state['config'])
    cfg.dataset.root=str(project);cfg.dataset.kind='manifest'
    cfg.dataset.val_manifest='data/manifests/div2k_jpeg_v1/screening/test.jsonl'
    cfg.jpeg_study.mode='off';cfg.loss.consistency_weight=0;cfg.degradation.jpeg_quality=20
    dataset=RestorationDataset(cfg,'test')
    if len(dataset)!=16 or dataset.fingerprint not in f['test_manifest_hashes']:raise ValueError('Protected source mismatch')
    if {r['id'] for r in dataset.records}!={f'{i:04d}' for i in range(801,817)}:raise ValueError('Unexpected image IDs')
    validate_splits([r for r in state['data_identity'] if r['split']!='test']+dataset.records)
    model=build_model(cfg.model).to(device).eval();model.load_state_dict(state['model'],strict=True)
    torch.cuda.reset_peak_memory_stats(device)
    result=evaluate_full(model,dataset,device,destination,{'0801','0802','0803'})
    result.update({'checkpoint_sha256':selection['sha256'],'step':state['step'],
                   'dataset_hash':dataset.fingerprint,'split':'test','codec':codec_metadata(),
                   'frozen_study_sha256':digest(frozen),'model_parameters':sum(p.numel() for p in model.parameters()),
                   'training':False,'device':'cuda'})
    (destination/'full_image.json').write_text(json.dumps(result,indent=2)+'\n')


def make_panels(destination: Path) -> None:
    """Join identical fixed crops from the three arms without rescaling pixels."""
    from PIL import Image,ImageDraw
    for image_id in ('0801','0802','0803'):
        for kind in ('clean','compressed'):
            identities={digest(destination/arm/f'{image_id}_{kind}_center.png') for arm in 'ABC'}
            if len(identities)!=1:raise ValueError('Visual reference or compressed input differs between arms')
        paths=[destination/'A'/f'{image_id}_{kind}_center.png' for kind in ('clean','compressed')]
        paths += [destination/arm/f'{image_id}_restored_center.png' for arm in 'ABC']
        images=[]
        for path in paths:
            with Image.open(path) as opened:images.append(opened.copy())
        width,height=images[0].size
        if any(im.size!=(width,height) for im in images):raise ValueError('Visual crop shape mismatch')
        panel=Image.new('RGB',(5*width,height+32),'white');draw=ImageDraw.Draw(panel)
        for index,(label,image) in enumerate(zip(('Clean','JPEG Q20','A: random','B: stratified','C: consistency'),images)):
            panel.paste(image,(index*width,32));draw.text((index*width+5,8),label,fill='black')
        panel.save(destination/f'{image_id}_comparison.png')


def main() -> None:
    """Prepare existing artifacts and execute six bounded, independently saved jobs."""
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--approved',action='store_true')
    ap.add_argument('--worker',choices=['full','benchmark'])
    ap.add_argument('--project');ap.add_argument('--arm',choices=list('ABC'))
    ap.add_argument('--checkpoint');ap.add_argument('--destination')
    args=ap.parse_args()
    if not args.approved:raise PermissionError('Explicit completion-check authorization required')
    if args.worker:
        if not all((args.project,args.arm,args.checkpoint,args.destination)):ap.error('Worker arguments missing')
        worker(args);return
    args.device='cuda'
    drive=Path('/content/drive/MyDrive/RestorationProject')
    study=drive/'studies/cuda_jpeg_5000_v1'
    destination=study/'completion_checks_v1_cuda_index_fix'
    if destination.exists():raise FileExistsError('Completion checks already initiated; inspect Drive execution.json before recovery')
    project=Path('/content/Image_Degradatation_Restoration')
    code=drive/'colab_restoration_v1.zip'
    if digest(code)!='37066216dcce9ae8f53fa449b22dcdef20a7a83e098001bb39bd62baaf87cb3f':raise ValueError('Original project bundle hash mismatch')
    if not project.exists():
        with zipfile.ZipFile(code) as z:
            entries=safe(z)
            if any(PurePosixPath(e.filename).parts[0]!=project.name for e in entries):raise ValueError('Unexpected project root')
            z.extractall('/content')
    os.chdir(project)
    sys.path[:0]=[str(project),str(project/'notebooks')]
    from cloud_workflow import verify_bundle,restore_snapshot
    from utils.reproducibility import source_identity
    from utils.degradation import codec_metadata
    from utils.runtime import environment,select_device
    from utils.study import compare_phase_reports
    device=select_device(args.device)
    env=environment(device)
    print(f'Evaluation device: {args.device}; hardware: {env["device_name"]}; inference precision: FP32',flush=True)
    verify_bundle(project)
    package=drive/'colab_frozen_test16_v1.zip'
    if digest(package)!='3c570cda7806d96535264373652fa85634a4d450a118937130cc44729e982519':raise ValueError('Frozen data bundle hash mismatch')
    with zipfile.ZipFile(package) as z:
        entries=safe(z);inv=json.loads(z.read('FINAL_EVAL_BUNDLE.json'))
        if {e.filename for e in entries}!=set(inv['files'])|{'FINAL_EVAL_BUNDLE.json'}:raise ValueError('Unexpected evaluation payload')
        for name,record in inv['files'].items():
            data=z.read(name)
            if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Evaluation payload hash mismatch')
            path=project/name
            if not path.resolve().is_relative_to(project.resolve()):raise ValueError('Unsafe destination')
            if path.exists():
                if digest(path)!=record['sha256']:raise FileExistsError(path)
            else:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    frozen=project/'reports/studies/cuda_jpeg_5000_v1/frozen_study.json'
    f=json.loads(frozen.read_text())
    if source_identity()['files']!=f['source_files']:raise ValueError('Frozen model source changed')
    if codec_metadata()!=f['codec']:raise ValueError('Colab codec changed; do not score mismatched JPEG inputs')
    work=project/'reports'/f'completion_checks_v1_{uuid.uuid4().hex[:8]}'
    work.mkdir(parents=True,exist_ok=False)
    checkpoints={}
    for arm,folder in {'A':'A_seed42','B':'B_seed42','C':'C_seed42_resume_45934e2e'}.items():
        restored=work/arm
        restore_snapshot(study/folder/'step_005000_micro_00020000.zip',restored)
        checkpoints[arm]=restored/'best_model.pth'
        if digest(checkpoints[arm])!=next(s['sha256'] for s in f['selections'] if s['run_id']==f'{arm}_seed42'):raise ValueError('Frozen checkpoint mismatch')
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copy2(frozen,destination/'frozen_study.json')
    shutil.copy2(Path(__file__).resolve(),destination/'completion_launcher.py')
    (destination/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    status={'status':'running','training':False,'spending_usd':0,'device':'cuda','precision':'fp32',
            'full_image_cap_seconds_per_arm':1200,'benchmark_cap_seconds_per_arm':120,
            'child_wall_cap_seconds':3960,'allocator_limit_GiB':12,'completed':[],
            'authorization':'User explicitly instructed starting the three proposed completion checks on 2026-10-26.',
            'checkpoint_selection':'Frozen development selections; no tuning or reselection',
            'frozen_study_sha256':digest(frozen),'launcher_sha256':digest(Path(__file__))}
    def save_status():
        temporary=destination/'execution.pending.json'
        temporary.write_text(json.dumps(status,indent=2)+'\n');os.replace(temporary,destination/'execution.json')
    save_status()
    child_env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    try:
        for arm,checkpoint in checkpoints.items():
            for kind,cap in [('full',1200),('benchmark',120)]:
                target=destination/arm if kind=='full' else destination/f'{arm}_latency.json'
                command=[sys.executable,str(Path(__file__).resolve()),'--approved','--worker',kind,
                         '--project',str(project),'--arm',arm,'--checkpoint',str(checkpoint),'--destination',str(target)]
                print(f'Starting {arm} {kind}; CUDA FP32, cap {cap}s; no training',flush=True)
                started=time.monotonic()
                with (destination/f'{arm}_{kind}.log').open('w') as log:
                    with subprocess.Popen(command,cwd=project,env=child_env,stdout=log,stderr=subprocess.STDOUT) as result:
                        last_count=0
                        while result.poll() is None:
                            if time.monotonic()-started >= cap:
                                result.kill();result.wait()
                                raise TimeoutError(f'{arm} {kind} exceeded {cap}s; saved per-image progress retained, no automatic retry')
                            progress=target/'progress.json' if kind=='full' else None
                            if progress is not None and progress.exists():
                                current=json.loads(progress.read_text())['count']
                                if current!=last_count:
                                    print(f'{arm}: {current}/16 full images saved to Drive',flush=True);last_count=current
                            time.sleep(1)
                if result.returncode:
                    raise RuntimeError(f'{arm} {kind} failed; inspect {arm}_{kind}.log; no automatic fallback')
                report_path=target/'full_image.json' if kind=='full' else target
                report=json.loads(report_path.read_text())
                if kind=='full' and (report.get('status')!='complete' or report.get('count')!=16):raise ValueError('Incomplete full-image panel')
                if kind=='benchmark' and (not report.get('completed') or any(r['measured_passes']!=30 for r in report['results'])):raise ValueError('Incomplete benchmark')
                status['completed'].append({'arm':arm,'kind':kind,'seconds':time.monotonic()-started,'report_sha256':digest(report_path)})
                save_status()
                if kind=='full':print(f"Saved {arm}: {report['count']} full images; PSNR={report['psnr']}; SSIM={report['ssim']:.6f}",flush=True)
                else:print(f'Saved {arm} latency and peak memory',flush=True)
        make_panels(destination)
        summary={'status':'complete','full_image':{a:json.loads((destination/a/'full_image.json').read_text()) for a in 'ABC'},
                 'latency':{a:json.loads((destination/f'{a}_latency.json').read_text()) for a in 'ABC'},
                 'limitations':['Single training seed, sixteen held-out images.','Full-image Q20 is separate from the phase-patch endpoint.','No new novelty or broad benchmark claim.']}
        (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        status['status']='complete';save_status()
        export=study/'restoration_completion_checks_v1_cuda_index_fix.zip'
        if export.exists():raise FileExistsError(export)
        files=sorted(p for p in destination.rglob('*') if p.is_file())
        inventory={str(p.relative_to(destination)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in files}
        with zipfile.ZipFile(export,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:archive.write(path,str(path.relative_to(destination)))
            archive.writestr('INVENTORY.json',json.dumps(inventory,indent=2)+'\n')
        with zipfile.ZipFile(export) as archive:safe(archive)
        print('COMPLETION CHECKS FINISHED:',destination,flush=True)
        print(f'Export: {export}; MiB={export.stat().st_size/1024**2:.2f}; SHA256={digest(export)}',flush=True)
    except BaseException as error:
        status['status']='incomplete';status['error']=repr(error);save_status();raise


if __name__=='__main__':main()
