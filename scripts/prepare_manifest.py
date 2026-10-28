"""Prepare source-disjoint clean RGB manifests; never download images."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
_script_directory = Path(__file__).resolve().parent
sys.path = [str(_script_directory.parent)] + [p for p in sys.path if Path(p).resolve() != _script_directory]

from utils.config import load_config
from utils.dataset_loader import RestorationDataset, read_manifest, validate_splits
from utils.io import file_sha256, load_rgb


def main() -> None:
    """Build a manifest for supplied files, or validate configured train/val splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--images", help="Image directory relative to data root")
    parser.add_argument("--input-images", help="Optional measured/LR directory; pair by stem (or stem+xS)")
    parser.add_argument("--scale", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--dataset", default="user_supplied_rgb")
    parser.add_argument("--version", default="1")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.validate_only:
        if not args.config:
            parser.error("--validate-only requires --config")
        cfg = load_config(args.config)
        training = RestorationDataset(cfg, "train", True)
        validation = RestorationDataset(cfg, "val")
        validate_splits(training.records + validation.records)
        print(f"validated train={len(training)} val={len(validation)} procedural_test_only={cfg.dataset.kind == 'procedural'}")
        return
    if not args.images or not args.split or not args.output:
        parser.error("Building requires --images, --split and --output")
    root = Path(args.data_root).resolve()
    images = sorted(p for p in (root / args.images).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise ValueError("No supplied RGB images found")
    records = []
    inputs = {}
    if args.input_images:
        for path in sorted((root / args.input_images).rglob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            identifier = path.stem.removesuffix(f"x{args.scale}") if args.scale > 1 else path.stem
            if identifier in inputs:
                raise ValueError(f"Duplicate input identifier: {identifier}")
            inputs[identifier] = path
        if set(inputs) != {path.stem for path in images}:
            raise ValueError("Input/target identifiers do not match exactly")
    elif args.scale != 1:
        parser.error("Scale-specific official pairs require --input-images")
    for path in images:
        image = load_rgb(path)
        input_path = inputs.get(path.stem)
        if input_path is not None:
            expected = (image.shape[-2] // args.scale, image.shape[-1] // args.scale)
            if image.shape[-2] % args.scale or image.shape[-1] % args.scale or load_rgb(input_path).shape[-2:] != expected:
                raise ValueError(f"Pair dimension mismatch: {path}")
        records.append({"id": path.stem, "dataset": args.dataset, "version": args.version,
                        "split": args.split, "source_id": path.stem, "scene_id": None,
                        "target_path": str(path.relative_to(root)),
                        "input_path": str(input_path.relative_to(root)) if input_path else None,
                        "input_sha256": file_sha256(input_path) if input_path else None,
                        "height": image.shape[-2], "width": image.shape[-1], "bit_depth": 8,
                        "color_space": "sRGB", "sha256": file_sha256(path), "scale": args.scale})
    validate_splits(records)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row) + "\n" for row in records))
    read_manifest(destination, root)
    print(f"manifest={destination} count={len(records)}; check all splits together before training")


if __name__ == "__main__":
    main()
