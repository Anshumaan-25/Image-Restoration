"""Bundle current approved engineering code plus existing Poisson train/dev data only."""
from pathlib import Path
import hashlib,json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from utils.reproducibility import source_identity
from utils.io import file_sha256


def main() -> None:
    """Prepare a separate Colab checkout; never modify the old JPEG bundle/runtime."""
    base=ROOT/'reports/engineering_v2';stage=ROOT/'dist/Image_Restoration_Extension_v2'
    if stage.exists():raise FileExistsError('Preserve existing extension package')
    stage.mkdir(parents=True)
    sources=source_identity()['files']
    paths=set(sources)
    paths.add('notebooks/cloud_workflow.py')
    for p in (ROOT/'tests/fixtures/upstream').glob('*'):paths.add(str(p.relative_to(ROOT)))
    manifests=['data/manifests/div2k_jpeg_v1/screening/train.jsonl','data/manifests/poisson_v2/val.jsonl']
    for name in manifests:
        paths.add(name)
        for line in (ROOT/name).read_text().splitlines():
            row=json.loads(line)
            for key in ('target_path','input_path'):
                if row.get(key):paths.add(row[key])
    inventory={}
    for name in sorted(paths):
        source=ROOT/name
        if name in sources and file_sha256(source)!=sources[name]:raise ValueError('Source changed during packaging')
        target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        inventory[name]={'bytes':target.stat().st_size,'sha256':file_sha256(target)}
    (stage/'EXTENSION_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    archive=ROOT/'dist/colab_extension_v2.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(stage.rglob('*')):
            if path.is_file():z.write(path,str(Path(stage.name)/path.relative_to(stage)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    if source_identity()['files']!=sources:raise ValueError('Source changed during packaging')
    digest=file_sha256(archive)
    launcher=base/'colab_poisson_launcher.py'
    text=launcher.read_text()
    if text.count('BUNDLE_SHA_PLACEHOLDER')!=1:raise ValueError('Launcher already sealed')
    launcher.write_text(text.replace('BUNDLE_SHA_PLACEHOLDER',digest))
    result={'status':'prepared_not_cloud_executed','bundle':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'launcher_sha256':file_sha256(launcher),'runtime_root':'/content/Image_Restoration_Extension_v2','source_files':sources,'data_counts':{'train_full_sources':32,'development_centre128_sources':8},'protected_test_payload':False,'pretrained_weight_payload':False,'purpose':'Actual-device Poisson profile followed only by specifically approved bounded training; no automatic training or test scoring.'}
    (base/'cloud_extension_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    downloads=Path('/Users/anshu/Downloads')
    target=downloads/launcher.name
    if target.exists():raise FileExistsError(target)
    shutil.copy2(launcher,target)
    print(json.dumps({k:v for k,v in result.items() if k!='source_files'},indent=2))


if __name__=='__main__':main()
