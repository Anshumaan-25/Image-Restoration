"""Single/directory RGB inference; checkpoint metadata controls the architecture."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import config_from_dict
from utils.io import load_rgb, save_rgb
from utils.runtime import select_device


def main() -> None:
    """Restore images with explicit conflict/overwrite handling and no downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pretrained-id", help="Explicit whitelisted official asset; separate from project checkpoints")
    parser.add_argument("--weights-sha256", help="Required acquisition-receipt SHA256 with --pretrained-id")
    parser.add_argument("--model")
    parser.add_argument("--scale", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.pretrained_id:
        from utils.pretrained import load_pretrained
        model,cfg,_ = load_pretrained(args.pretrained_id,args.weights,args.weights_sha256)
    else:
        if args.weights_sha256:parser.error("--weights-sha256 requires --pretrained-id")
        state = load_checkpoint(args.weights)
        cfg = config_from_dict(state["config"])
        model = build_model(cfg.model)
        model.load_state_dict(state["model"], strict=True)
    if args.model and args.model != cfg.model.name or args.scale is not None and args.scale != cfg.model.scale:
        raise ValueError("CLI model/scale conflicts with checkpoint metadata")
    device = select_device(args.device)
    model = model.to(device)
    model.eval()
    source, destination = Path(args.input), Path(args.output)
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) if source.is_dir() else [source]
    if not files:
        raise ValueError("Input directory contains no PNG/JPEG images")
    outputs = [destination / (p.stem + ".png") if source.is_dir() else destination for p in files]
    if len(set(outputs)) != len(outputs) or any(p.exists() for p in outputs):
        raise FileExistsError("Output collision; use a new output path/directory")
    with torch.no_grad():
        for filename, output in zip(files, outputs):
            restored = model(load_rgb(filename)[None].to(device))
            save_rgb(restored, output)
    print(f"restored={len(files)} device={device} output={destination}")


if __name__ == "__main__":
    main()
