"""Validated matched-arm experiment plans and image-grouped comparison statistics."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import json
import math
import numpy as np
import yaml
from utils.config import load_config, config_from_dict
from utils.io import file_sha256


def study_plan(path: str | Path, device: str, batch_size: int | None = None,
               accumulation_steps: int | None = None, precision: str | None = None) -> dict[str, Any]:
    """Resolve three matched arms without starting training or opening image payloads."""
    raw = yaml.safe_load(Path(path).read_text())
    required = {'name','arms','seeds','max_seconds_per_run'}
    if not isinstance(raw,dict) or set(raw) != required or not isinstance(raw['name'],str) or not raw['name']:
        raise ValueError('Study requires exactly name/arms/seeds/max_seconds_per_run')
    if not isinstance(raw['arms'],dict) or set(raw['arms']) != {'A','B','C'}:
        raise ValueError('Study requires A/B/C configurations')
    seeds = raw['seeds']
    if not isinstance(seeds,list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    cap=raw['max_seconds_per_run']
    if type(cap) not in (int,float) or not math.isfinite(cap) or cap <= 0:
        raise ValueError('Each run requires a finite positive wall-time cap')
    if device not in {'cpu','mps','cuda'}:
        raise ValueError('Study execution requires an explicit device')
    arms={}
    canonical=[]
    for arm,path in raw['arms'].items():
        cfg=load_config(path)
        cfg.training.device=device
        cfg.training.max_seconds=cap
        if batch_size is not None:
            cfg.training.batch_size=batch_size
        if accumulation_steps is not None:
            cfg.training.accumulation_steps=accumulation_steps
        if precision is not None:
            cfg.training.precision=precision
        cfg=config_from_dict(cfg.to_dict())
        expected='random' if arm == 'A' else 'stratified'
        if cfg.jpeg_study.mode != expected or (arm in {'A','B'} and cfg.loss.consistency_weight != 0) or (arm == 'C' and cfg.loss.consistency_weight <= 0):
            raise ValueError('Arms must be random/pixel, stratified/pixel, stratified/consistency')
        if cfg.training.max_steps is None or cfg.dataset.kind != 'manifest':
            raise ValueError('Research study requires manifest data and a fixed update budget')
        normalized=deepcopy(cfg.to_dict())
        normalized['jpeg_study'].pop('mode')
        normalized['loss'].pop('consistency_weight')
        normalized['training'].pop('seed')
        normalized.pop('output')
        canonical.append(normalized)
        arms[arm]=cfg
    if any(item != canonical[0] for item in canonical[1:]):
        raise ValueError('Study arms differ beyond phase mode and consistency weight')
    manifests={}
    for field in ('train_manifest','val_manifest'):
        target=Path(next(iter(arms.values())).dataset.root)/getattr(next(iter(arms.values())).dataset,field)
        manifests[field]={'path':str(target),'sha256':file_sha256(target)}
    runs=[]
    for seed in seeds:
        for arm,cfg in arms.items():
            resolved=deepcopy(cfg.to_dict())
            resolved['training']['seed']=seed
            runs.append({'id':f'{arm}_seed{seed}','arm':arm,'seed':seed,'config':resolved})
    return {'schema_version':1,'name':raw['name'],'status':'planned','runs':runs,'manifests':manifests,
            'total_optimizer_updates':sum(r['config']['training']['max_steps'] for r in runs),
            'max_run_seconds':cap,'max_total_training_seconds':cap*len(runs),
            'views_per_clean_crop':2,'selection':'development_mean_per_image_worst_phase_psnr',
            'protected_test_used':False,'novelty_established':False}


def compare_phase_reports(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
                          seed: int = 42, repetitions: int = 10000) -> dict[str, Any]:
    """Pair seeds, average seed deltas per image, then bootstrap whole images."""
    if not baseline or len(baseline) != len(candidate) or type(repetitions) is not int or repetitions < 1:
        raise ValueError('Equal nonempty report lists and positive bootstrap repetitions required')
    def keyed(reports: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        result={}
        for report in reports:
            if report.get('endpoint') != 'mean_per_image_worst_phase_psnr' or not isinstance(report.get('images'),list):
                raise ValueError('Phase endpoint reports with per-image results required')
            key=report['training_seed']
            if key in result:
                raise ValueError('Duplicate training seed')
            result[key]=report
        return result
    left,right=keyed(baseline),keyed(candidate)
    if left.keys() != right.keys():
        raise ValueError('Seed sets differ')
    fields=('dataset_hash','split','protocol','endpoint','phase_panel','quality','patch_size','codec','model','procedural_test_only','frozen_study_sha256','phase_constructor','reference_crop','training_config','selection_horizon','training_source_hash')
    reference=baseline[0]
    ids=None
    differences=[]
    for key in sorted(left):
        a,b=left[key],right[key]
        for field in fields:
            if field not in a or field not in b or a[field] != b[field] or a[field] != reference[field]:
                raise ValueError(f'Incomparable evaluation protocol: {field}')
        maps=[]
        for report in (a,b):
            images={i['id']:i['worst_phase_psnr'] for i in report['images']}
            if len(images) != len(report['images']) or len(images) != report['count']:
                raise ValueError('Duplicate images or incorrect evaluation count')
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in images.values()):
                raise ValueError('Finite image endpoints required for paired differences')
            maps.append(images)
        if maps[0].keys() != maps[1].keys() or not maps[0]:
            raise ValueError('Image identity sets differ or are empty')
        current=sorted(maps[0])
        if ids is not None and ids != current:
            raise ValueError('Image identities differ across seeds')
        ids=current
        differences.append([maps[1][i]-maps[0][i] for i in ids])
    array=np.asarray(differences,dtype=np.float64)
    per_image=array.mean(axis=0)
    generator=np.random.default_rng(seed)
    # Bounded allocation: bootstrap iterations in chunks, never phases as IID samples.
    draws=[]
    for start in range(0,repetitions,256):
        indices=generator.integers(len(ids),size=(min(256,repetitions-start),len(ids)))
        draws.extend(per_image[indices].mean(axis=1).tolist())
    return {'endpoint':'candidate_minus_baseline_mean_per_image_worst_phase_psnr',
            'delta_db':float(per_image.mean()),'image_bootstrap_ci95_db':np.quantile(draws,[.025,.975]).tolist(),
            'bootstrap_unit':'source_image_after_averaging_paired_seed_deltas','bootstrap_repetitions':repetitions,
            'image_count':len(ids),'seeds':sorted(left),'seed_mean_deltas_db':array.mean(axis=1).tolist(),
            'seed_delta_std_db':float(array.mean(axis=1).std(ddof=1)) if len(left)>1 else None,
            'per_image':[{'id':i,'delta_db':float(d)} for i,d in zip(ids,per_image)],
            'limitation':'Image bootstrap is conditional on observed training seeds; small samples do not establish publishable novelty.'}
