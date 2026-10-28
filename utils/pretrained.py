"""Explicit checksum-verified official inference weights, separate from resumable training."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import re
import torch
from torch import nn
from utils.pretrained_registry import PRETRAINED
from utils.config import Config,config_from_dict
from utils.io import file_sha256


def extract_parameters(payload: Any,key: str,allow_direct: bool=False) -> dict[str,torch.Tensor]:
    """Select only the documented tensor state; never substitute EMA or partial keys."""
    if not isinstance(payload,dict):raise ValueError('Expected official tensor-state dictionary')
    state=payload.get(key)
    if state is None and allow_direct:state=payload
    if not isinstance(state,dict) or not state or any(not isinstance(k,str) or not isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('Missing documented parameter mapping; no EMA or random-weight fallback')
    if any(not torch.isfinite(value).all() for value in state.values()):raise ValueError('Pretrained weights contain nonfinite tensors')
    return state


def load_pretrained(identifier: str,path: str | Path,expected_sha256: str) -> tuple[nn.Module,Config,dict[str,Any]]:
    """Strict-load an already supplied whitelisted asset; network access is never used.

    The required SHA is the user's independently retained acquisition receipt hash.
    Official release metadata has no publisher digest, so do not misrepresent this
    local integrity check as publisher authentication or verified training overlap.
    """
    if identifier not in PRETRAINED:raise ValueError('Unapproved pretrained identifier')
    if not isinstance(expected_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None:
        raise ValueError('Explicit lowercase SHA256 from acquisition receipt required')
    record=PRETRAINED[identifier];filename=Path(path)
    if filename.stat().st_size!=record['bytes']:raise ValueError('Official asset byte size mismatch')
    actual=file_sha256(filename)
    if actual!=expected_sha256:raise ValueError('Pretrained SHA256 mismatch')
    payload=torch.load(filename,map_location='cpu',weights_only=True)
    parameters=extract_parameters(payload,record['param_key'],record['allow_direct_state_dict'])
    cfg=config_from_dict({'model':{'name':record['model'],'scale':record['scale']},
                          'degradation':{'profile':record['profile']}})
    from models import build_model
    model=build_model(cfg.model)
    model.load_state_dict(parameters,strict=True)
    model.eval()
    receipt={**record,'id':identifier,'sha256':actual,'origin':'official_supervised_pretrained',
             'training_seed':None,'project_training_performed':False,
             'publisher_hash_available':False,'training_overlap':'Author-described sources; per-image training membership not independently verified',
             'weights_path':str(filename.resolve())}
    return model,cfg,receipt
