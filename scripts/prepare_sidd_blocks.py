"""Import already supplied, explicitly laid-out SIDD validation block MAT pairs."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.sidd_blocks import prepare_sidd_blocks


def main() -> None:
    """Create paired PNGs and immutable validation metadata; never download or train."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noisy-mat',required=True)
    parser.add_argument('--clean-mat',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--layout',required=True,choices=['nkhwc','matlab_hdf5_reversed'])
    parser.add_argument('--noisy-key',default='ValidationNoisyBlocksSrgb')
    parser.add_argument('--clean-key',default='ValidationGtBlocksSrgb')
    args=parser.parse_args()
    result=prepare_sidd_blocks(args.noisy_mat,args.clean_mat,args.output,args.layout,args.noisy_key,args.clean_key)
    print(f"Imported {result['count']} SIDD validation blocks; no synthesis or research scoring")


if __name__=='__main__':main()
