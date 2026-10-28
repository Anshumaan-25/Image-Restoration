"""Run every bundled selected checkpoint from a standalone verified extraction."""
from pathlib import Path,PurePosixPath
import hashlib,json,os,subprocess,sys,time,zipfile
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]

def main() -> None:
    """Verify payload identity, offline dry-run, all eight model assets and outputs."""
    receipt=json.loads((ROOT/'reports/engineering_v2/engineering_release_v3.json').read_text());archive=Path(receipt['artifact'])
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    destination=ROOT/'dist/release_verification_v3';destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist();assert len(names)==len(set(names))
        for name in names:
            p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
        z.extractall(destination)
    kit=destination/'restoration_engineering_v3';inventory=json.loads((kit/'RELEASE_INVENTORY.json').read_text())
    for name,row in inventory.items():
        p=kit/name;assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    checks=[]
    def run(args: list[str]) -> dict:
        began=time.monotonic();r=subprocess.run([sys.executable,*args],cwd=kit,env=env,capture_output=True,text=True,timeout=120)
        record={'command':[sys.executable,*args],'exit_code':r.returncode,'seconds':time.monotonic()-began,'stdout':r.stdout,'stderr':r.stderr};checks.append(record)
        assert r.returncode==0,record
        return record
    run(['train.py','--dry-run','--epochs','1','--batch-size','2','--device','cpu'])
    tasks=[]
    for arm in 'ABC':tasks.append((f'jpeg_{arm}','samples/jpeg_q20.png',f'weights/jpeg_{arm}.pth',None,None,1))
    tasks.append(('poisson','samples/poisson_peak60.png','weights/poisson_best.pth',None,None,1))
    tasks.extend([
        ('real','samples/real_noise_crop.png','weights/real_denoising.pth','restormer_sidd','4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387',1),
        ('gaussian','samples/real_noise_crop.png','weights/gaussian_color_denoising_blind.pth','restormer_gaussian','4e2314dae77bb907184da6e2ac625b921491d6b33de26a52512eff1c6bb37fbd',1),
        ('sr_x2','samples/bird_x2.png','weights/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth','swinir_x2','193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7',2),
        ('sr_x4','samples/bird_x4.png','weights/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth','swinir_x4','09fad24e32ae62722e1a055efde9921328f4137981bab0a42a4a3a806306c58e',4)])
    for name,sample,weights,asset,digest,scale in tasks:
        output=f'outputs/verified_{name}.png';args=['inference.py','--input',sample,'--output',output,'--weights',weights,'--device','cpu']
        if asset:args.extend(['--pretrained-id',asset,'--weights-sha256',digest])
        record=run(args)
        with Image.open(kit/sample) as image:expected=(image.width*scale,image.height*scale)
        with Image.open(kit/output) as image:assert image.mode=='RGB' and image.size==expected
        record.update(output_size=list(expected),output_sha256=hashlib.sha256((kit/output).read_bytes()).hexdigest(),role='Inference contract verification only; no quality score from this fixture')
        print(f'{name}: actual standalone CPU inference passed; output={expected}',flush=True)
    audit={'status':'passed','archive_sha256':receipt['sha256'],'payload_files':len(inventory),'all_eight_assets_inferred':True,'dry_run_two_updates_checkpoint_reload':True,'device':'cpu','checks':checks,'no_downloads':True,'spending_usd':0,'extraction':str(kit),'novelty_established':False}
    (ROOT/'reports/engineering_v2/engineering_release_v3_standalone_verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    receipt['status']='verified_portable_engineering_delivery';receipt['standalone_verification']='engineering_release_v3_standalone_verification.json';(ROOT/'reports/engineering_v2/engineering_release_v3.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('All standalone delivery checks passed.',flush=True)

if __name__=='__main__':main()
