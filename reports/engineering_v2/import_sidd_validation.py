"""Validate an explicitly supplied approved SIDD archive and import its public MAT pairs."""
from pathlib import Path
import argparse,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'notebooks')]
from cloud_workflow import safe_members
from utils.io import file_sha256
from utils.sidd_blocks import prepare_sidd_blocks


def main() -> None:
    """Extract only the expected paired MAT files; retain all originals and diagnostics."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True)
    args=parser.parse_args();source=Path(args.archive)
    if source.stat().st_size!=461780640:raise ValueError('Archive differs from the specifically approved authors mirror size; review provenance before importing')
    destination=ROOT/'data/raw/SIDD_validation_mat_v2'
    if destination.exists():raise FileExistsError('Preserve previous import; inspect before recovery')
    names=['ValidationNoisyBlocksSrgb.mat','ValidationGtBlocksSrgb.mat']
    with zipfile.ZipFile(source) as archive:
        entries=safe_members(archive)
        if archive.testzip() is not None:raise ValueError('SIDD archive CRC failed')
        selected={}
        for name in names:
            matches=[entry for entry in entries if Path(entry.filename).name==name]
            if len(matches)!=1:raise ValueError(f'Expected exactly one {name}')
            selected[name]=matches[0]
        if sum(entry.file_size for entry in selected.values())>1024**3:raise ValueError('Unexpected uncompressed SIDD data size')
        destination.mkdir(parents=True,exist_ok=False)
        for name,entry in selected.items():
            with archive.open(entry) as original,(destination/name).open('xb') as output:
                import shutil
                shutil.copyfileobj(original,output)
    result=prepare_sidd_blocks(destination/names[0],destination/names[1],ROOT/'data/processed/SIDD_validation_v2','nkhwc')
    result.update({'archive_sha256':file_sha256(source),'archive':str(source),'origin':'authors_public_validation_mirror','training_performed':False})
    (ROOT/'reports/engineering_v2/sidd_import_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"Imported {result['count']} true SIDD public validation pairs; no inference or training")


if __name__=='__main__':main()
