"""Verify the actual completed cloud archive, identities and per-block aggregation."""
from pathlib import Path,PurePosixPath
import hashlib,json,math,statistics,zipfile
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from utils.reproducibility import source_identity

def main() -> None:
    """Import only verified safe payloads; never replace an existing import."""
    archive=Path('/Users/anshu/Downloads/sidd_completed_17877781.zip');digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest=='b5fb7aa44a12b469574d9c50e234baffcb5e30fa0f9ea1f209a8d056ef801717'
    destination=ROOT/'experiments/studies/sidd_full_v3_import_20261027'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist();assert len(names)==len(set(names))
        inventory=json.loads(z.read('export_inventory.json'));assert set(names)==set(inventory)|{'export_inventory.json'}
        for name,record in inventory.items():
            p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
            member=z.getinfo(name);assert member.external_attr>>16&0o170000!=0o120000
            data=z.read(name);assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
        destination.mkdir(exist_ok=False);z.extractall(destination)
    result=json.loads((destination/'results.json').read_text());plan=json.loads((destination/'frozen_plan.json').read_text());profile=json.loads((destination/'cuda_profile.json').read_text())
    assert result==json.loads((destination/'progress.json').read_text())
    assert result['status']=='complete' and result['count']==result['expected_count']==1280 and result['error'] is None and result['training'] is False
    assert result['elapsed_seconds']<=1200
    rows=result['images'];assert len(rows)==len({r['id'] for r in rows})==1280
    manifest=[json.loads(s) for s in (destination/'validation_manifest.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==[r['id'] for r in manifest]
    assert plan['source']['files']==source_identity()['files']
    assert plan['weights']['sha256']=='4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387'
    assert plan['precision']=='fp32' and plan['metric_border']==0 and plan['metric_channel']=='RGB' and plan['metric_quantization'] is False
    assert plan['expected_count']==1280 and plan['official_hidden_benchmark'] is False
    assert profile['reference_parity_pairs']==20 and profile['max_reference_ssim_error']<=1e-10 and profile['projected_full_seconds']<960 and profile['peak_allocated_bytes']<=12*1024**3
    assert plan['launcher_sha256']==hashlib.sha256((ROOT/'reports/engineering_v2/colab_sidd_full_validation_optimized_v3.py').read_bytes()).hexdigest()
    reaggregated={}
    for role,items,summary in [('restoration',rows,result),('input',[r['reference'] for r in rows],result['reference'])]:
        reaggregated[role]={}
        for key in ('psnr','ssim'):
            assert all(math.isfinite(r[key]) for r in items)
            value=statistics.fmean(r[key] for r in items);assert abs(value-summary[key])<1e-10;reaggregated[role][key]=value
    local=[json.loads(s) for s in (ROOT/'data/processed/SIDD_validation_v2/val.jsonl').read_text().splitlines()]
    assert len(local)==1280
    assert manifest==local, 'Cloud importer records differ from independently verified local paired import'
    receipt={'status':'passed','date':'2026-10-27','archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'destination':str(destination),'count':1280,'identities_match_independent_local_import':True,'source_files':len(plan['source']['files']),'all_results_finite':True,'per_image_reaggregation':reaggregated,'elapsed_scoring_seconds':result['elapsed_seconds'],'profile':profile,'weights_sha256':plan['weights']['sha256'],'protocol':'project_rgb_float_v1; RGB; border0; no quantization; valid Gaussian11 SSIM; CPU float64 separable verified against frozen reference','scope':'All official public validation blocks; not hidden benchmark or exact published-protocol reproduction','training':False,'spending_usd':0,'novelty_established':False}
    (ROOT/'reports/engineering_v2/sidd_full_results_audit_20261027.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
