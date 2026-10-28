"""Explicit, lossless import of official uint8 SIDD sRGB validation MAT block pairs."""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
from typing import Any
import json
import numpy as np
from PIL import Image
from utils.io import file_sha256


def _open_array(path: Path,key: str,layout: str,stack: ExitStack) -> Any:
    """Open only the named array using the explicitly selected MAT storage layout."""
    if layout=='matlab_hdf5_reversed':
        try:import h5py
        except ImportError as error:raise RuntimeError('MAT v7.3 import requires explicitly installed h5py') from error
        if not h5py.is_hdf5(path):raise ValueError('Selected HDF5 layout but input is not HDF5 MAT')
        handle=stack.enter_context(h5py.File(path,'r'))
        if key not in handle:raise ValueError(f'Missing named SIDD MAT key: {key}')
        array=handle[key]
    elif layout=='nkhwc':
        try:from scipy.io import loadmat
        except ImportError as error:raise RuntimeError('MAT v5 import requires explicitly installed scipy') from error
        try:data=loadmat(path,variable_names=[key])
        except NotImplementedError as error:raise ValueError('MAT v7.3 requires explicit matlab_hdf5_reversed layout') from error
        if key not in data:raise ValueError(f'Missing named SIDD MAT key: {key}')
        array=data[key]
    else:raise ValueError('Layout must be explicitly nkhwc or matlab_hdf5_reversed')
    if array.dtype!=np.dtype('uint8') or len(array.shape)!=5 or any(v<1 for v in array.shape):
        raise ValueError('SIDD adapter requires a nonempty five-dimensional uint8 array; no guessed scaling or quantization')
    return array


def prepare_sidd_blocks(noisy_mat: str | Path,clean_mat: str | Path,output: str | Path,
                        layout: str,noisy_key: str='ValidationNoisyBlocksSrgb',
                        clean_key: str='ValidationGtBlocksSrgb') -> dict[str,Any]:
    """Write lossless RGB PNG pairs and val manifest, preserving parent/block identities.

    nkhwc declares MATLAB v5 arrays [source,block,H,W,C]. HDF5 reversed declares
    [C,W,H,block,source]. Public validation stays validation, never relabeled test.
    This exports paired sRGB uint8 blocks only; it is not a raw/SIDD-Medium importer.
    """
    noisy_path,clean_path,directory=Path(noisy_mat),Path(clean_mat),Path(output)
    if directory.exists():raise FileExistsError('Choose a new SIDD output directory')
    if not noisy_key or not clean_key:raise ValueError('Explicit nonempty MAT keys required')
    with ExitStack() as stack:
        noisy=_open_array(noisy_path,noisy_key,layout,stack)
        clean=_open_array(clean_path,clean_key,layout,stack)
        if noisy.shape!=clean.shape:raise ValueError('Noisy/clean block array geometry differs')
        if layout=='nkhwc':sources,blocks,height,width,channels=noisy.shape
        else:channels,width,height,blocks,sources=noisy.shape
        if channels!=3 or min(height,width)<11:raise ValueError('Expected RGB blocks at least 11x11 for declared SSIM')
        hashes={'noisy_mat_sha256':file_sha256(noisy_path),'clean_mat_sha256':file_sha256(clean_path)}
        directory.mkdir(parents=True,exist_ok=False)
        (directory/'noisy').mkdir();(directory/'clean').mkdir()
        records=[]
        state={'status':'incomplete','layout':layout,'encoding':'uint8_srgb_lossless_png',
               'noisy_key':noisy_key,'clean_key':clean_key,**hashes,'expected_count':sources*blocks,
               'split':'val','scene_identity':'MAT parent index preserved; true capture-scene identity not supplied'}
        (directory/'import.json').write_text(json.dumps(state,indent=2)+'\n')
        for source in range(sources):
            for block in range(blocks):
                image_id=f'sidd_val_{source:03d}_{block:03d}'
                paths={}
                for name,array in [('noisy',noisy),('clean',clean)]:
                    pixels=(np.asarray(array[source,block]) if layout=='nkhwc' else
                            np.asarray(array[:,:,:,block,source]).transpose(2,1,0))
                    if pixels.shape!=(height,width,3):raise ValueError('Explicit block decoding geometry failed')
                    filename=directory/name/(image_id+'.png');Image.fromarray(pixels).save(filename)
                    paths[name]=str(filename.relative_to(directory))
                records.append({'id':image_id,'dataset':'SIDD','version':'public_validation_srgb_blocks',
                                'split':'val','source_id':f'sidd_val_parent_{source:03d}',
                                'target_path':paths['clean'],'input_path':paths['noisy'],
                                'sha256':file_sha256(directory/paths['clean']),
                                'input_sha256':file_sha256(directory/paths['noisy']),
                                'height':height,'width':width,'bit_depth':8,'color_space':'sRGB','scale':1,
                                'degradation':'sidd_srgb_pairs_v1','metadata':{**hashes,'parent_index':source,
                                'block_index':block,'mat_layout':layout,'mat_encoding':'uint8',
                                'noisy_key':noisy_key,'clean_key':clean_key}})
        manifest=directory/'val.jsonl'
        manifest.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in records))
        from utils.dataset_loader import read_manifest
        read_manifest(manifest,directory)
        state.update(status='complete',count=len(records),manifest_sha256=file_sha256(manifest))
        (directory/'import.json').write_text(json.dumps(state,indent=2)+'\n')
        return state
