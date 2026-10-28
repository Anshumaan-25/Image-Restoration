"""Compare saved phase evaluation reports with paired image-level uncertainty."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
_script_directory=Path(__file__).resolve().parent
sys.path=[str(_script_directory.parent)]+[p for p in sys.path if Path(p).resolve()!=_script_directory]
from utils.logger import write_json
from utils.study import compare_phase_reports


def main() -> None:
    """Require matching seeds/protocols; refuse existing output files."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',nargs='+',required=True)
    parser.add_argument('--candidate',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--bootstrap-repetitions',type=int,default=10000)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result=compare_phase_reports([json.loads(Path(p).read_text()) for p in args.baseline],
                                 [json.loads(Path(p).read_text()) for p in args.candidate],
                                 repetitions=args.bootstrap_repetitions)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,result)
    print(f"delta_db={result['delta_db']} images={result['image_count']} report={output}")


if __name__=='__main__':
    main()
