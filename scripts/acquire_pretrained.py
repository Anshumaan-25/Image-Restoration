"""Explicitly approved acquisition of exact whitelisted official supervised assets."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.pretrained_registry import PRETRAINED


def main() -> None:
    """Download selected approved files once; retain partials, never auto-retry."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved',action='store_true')
    parser.add_argument('--ids',nargs='+',choices=list(PRETRAINED),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume-partials',action='store_true',help='Explicitly recover a retained partial with verified HTTP byte range')
    parser.add_argument('--max-seconds-per-asset',type=float,default=900)
    args=parser.parse_args()
    if not args.approved:raise PermissionError('Specific approval for these pretrained downloads is required')
    import math
    if not math.isfinite(args.max_seconds_per_asset) or args.max_seconds_per_asset<=0:parser.error('Positive finite asset time cap required')
    if len(set(args.ids))!=len(args.ids):parser.error('Duplicate asset identifiers')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    for identifier in args.ids:
        record=PRETRAINED[identifier];out=root/record['filename'];partial=out.with_suffix(out.suffix+'.partial')
        receipt=root/(identifier+'_receipt.json')
        if (out.exists() or partial.exists() or receipt.exists()) and not args.resume_partials:raise FileExistsError('Preserve prior acquisition; choose a new output directory or explicitly resume retained partials')
    for identifier in args.ids:
        record=PRETRAINED[identifier];out=root/record['filename'];partial=out.with_suffix(out.suffix+'.partial')
        receipt=root/(identifier+'_receipt.json')
        if out.exists():
            if not receipt.exists():raise FileExistsError('Completed file without receipt; inspect before reuse')
            existing=json.loads(receipt.read_text())
            if existing.get('id')!=identifier or out.stat().st_size!=record['bytes']:raise ValueError('Prior acquisition metadata differs')
            if hashlib.sha256(out.read_bytes()).hexdigest()!=existing['sha256']:raise ValueError('Prior completed download corrupt')
            print(f'Reusing verified completed {identifier}',flush=True);continue
        h=hashlib.sha256();count=0;started=time.monotonic()
        if partial.exists():
            with partial.open('rb') as retained:
                for chunk in iter(lambda:retained.read(1024*1024),b''):h.update(chunk);count+=len(chunk)
            if not 0<count<record['bytes']:raise ValueError('Partial size invalid; inspect retained artifact')
        headers={'User-Agent':'restoration-project-approved-acquisition'}
        if count:headers['Range']=f'bytes={count}-'
        request=urllib.request.Request(record['url'],headers=headers)
        print(f'Acquiring {identifier}: {record["bytes"]} bytes from official release',flush=True)
        with urllib.request.urlopen(request,timeout=30) as response:
            if count and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {count}-')):
                raise ValueError('Server did not honor exact recovery byte range; retained partial unchanged')
            stream=partial.open('ab' if count else 'xb')
            with stream:
                for chunk in iter(lambda:response.read(1024*1024),b''):
                    count+=len(chunk)
                    if count>record['bytes']:raise ValueError('Download exceeded approved exact asset size; partial retained')
                    if time.monotonic()-started>args.max_seconds_per_asset:raise TimeoutError('Asset acquisition exceeded its time cap; partial retained')
                    stream.write(chunk);h.update(chunk)
        if count!=record['bytes']:raise ValueError('Truncated official asset; partial retained')
        os.replace(partial,out)
        result={**record,'id':identifier,'sha256':h.hexdigest(),'bytes_downloaded':count,'seconds':time.monotonic()-started,
                'origin':'official_supervised_pretrained','publisher_digest_available':False,
                'hash_status':'Local acquisition digest; no independent publisher digest supplied',
                'status':'downloaded_not_yet_architecture_validated','training_performed':False,'spending_usd':0}
        (root/(identifier+'_receipt.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(f'Saved {identifier}; SHA256={result["sha256"]}',flush=True)


if __name__=='__main__':main()
