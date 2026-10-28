"""Prepare a fixed source-disjoint clean development crop panel from existing DIV2K."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from utils.io import load_rgb,save_rgb,file_sha256
from utils.dataset_loader import read_manifest,validate_splits


def main() -> None:
    """Preserve parent IDs/hashes and split before any crop; never synthesize a benchmark."""
    source=ROOT/'data/manifests/div2k_jpeg_v1/screening/val.jsonl'
    records=read_manifest(source,ROOT)
    images=ROOT/'data/processed/poisson_dev128_v2';manifest=ROOT/'data/manifests/poisson_v2/val.jsonl'
    if images.exists() or manifest.exists():raise FileExistsError('Preserve existing fixed panel')
    images.mkdir(parents=True);manifest.parent.mkdir(parents=True,exist_ok=True)
    output=[]
    for row in records:
        clean=load_rgb(ROOT/row['target_path']);h,w=clean.shape[-2:];top,left=(h-128)//2,(w-128)//2
        target=images/(row['id']+'.png');save_rgb(clean[:,top:top+128,left:left+128],target)
        output.append({**row,'target_path':str(target.relative_to(ROOT)),'sha256':file_sha256(target),
                       'height':128,'width':128,'metadata':{'parent_sha256':row['sha256'],
                       'parent_path':row['target_path'],'crop_tl_hw':[top,left,128,128],
                       'selection':'Fixed geometric centre of all dev8; no image-quality selection'}})
    train=read_manifest(ROOT/'data/manifests/div2k_jpeg_v1/screening/train.jsonl',ROOT)
    validate_splits(train+output)
    manifest.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in output))
    read_manifest(manifest,ROOT)
    (ROOT/'reports/engineering_v2/poisson_data_preparation.json').write_text(json.dumps({'status':'prepared_not_trained','train_sources':32,'development_sources':8,'train_manifest':'data/manifests/div2k_jpeg_v1/screening/train.jsonl','val_manifest':str(manifest.relative_to(ROOT)),'val_manifest_sha256':file_sha256(manifest),'split_before_crop':True,'protected_test_scored':False,'crop_size':128,'parent_identities_retained':True},indent=2)+'\n')
    print('Poisson train32/dev8 centre128 panel ready; no training or scoring.')


if __name__=='__main__':main()
