"""Frozen project-protocol evaluation from a metadata-bearing checkpoint."""
from __future__ import annotations

import argparse
import json
import hashlib
import uuid
from pathlib import Path
import torch

from models import build_model
from train import validate
from utils.checkpoint import load_checkpoint
from utils.config import config_from_dict
from utils.dataset_loader import RestorationDataset, validate_splits
from utils.logger import write_json
from utils.runtime import select_device
from utils.phase_evaluation import evaluate_phases
from utils.jpeg_phase import DEV_PHASES, ALL_PHASES
from utils.io import file_sha256
from utils.degradation import codec_metadata
from utils.reproducibility import source_identity


def main() -> None:
    """Evaluate all manifest images; unsupported benchmark replay fails explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", help="Directory containing val.jsonl and its relative images")
    parser.add_argument("--manifest", help="Manifest relative to --data-dir")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--task", choices=["jpeg", "denoising", "super_resolution"])
    parser.add_argument("--scale", type=int)
    parser.add_argument("--protocol", default="project_rgb_float_v1", choices=["project_rgb_float_v1"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase-panel", choices=["dev", "all"], help="Phase-study patch endpoint, not full-image quality")
    parser.add_argument("--jpeg-quality", type=int, choices=[10,20,40])
    parser.add_argument("--peak-photons", type=float, help="Explicit fixed Poisson evaluation level, separate from training mixture")
    parser.add_argument("--frozen-study", help="Frozen study JSON required for protected phase-test evaluation")
    parser.add_argument("--full-image", action="store_true", help="Ordinary JPEG full-image quality, separate from phase patches")
    parser.add_argument("--output", help="New JSON report path; defaults to a unique checkpoint-adjacent file")
    args = parser.parse_args()
    args.output = args.output or str(Path(args.checkpoint).parent / ("eval_" + uuid.uuid4().hex[:8] + ".json"))
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    state = load_checkpoint(args.checkpoint)
    if state.get("profile_only"):
        raise ValueError("A profiling checkpoint is not a selected research model")
    cfg = config_from_dict(state["config"])
    study_enabled = cfg.jpeg_study.mode != "off"
    if args.full_image and (not study_enabled or args.phase_panel):
        parser.error("--full-image requires a phase study and excludes --phase-panel")
    expected = "super_resolution" if cfg.model.scale > 1 else ("jpeg" if cfg.degradation.profile.startswith("jpeg") else "denoising")
    if args.task and args.task != expected or args.scale is not None and args.scale != cfg.model.scale:
        raise ValueError("CLI task/scale conflicts with checkpoint metadata")
    if args.data_dir:
        cfg.dataset.kind, cfg.dataset.root = "manifest", args.data_dir
        cfg.dataset.val_manifest = args.manifest or "val.jsonl"
    elif args.manifest:
        parser.error("--manifest requires --data-dir")
    evaluation_cfg = config_from_dict(cfg.to_dict())
    if args.peak_photons is not None:
        import math
        if cfg.degradation.profile != "poisson_linear_rgb_v1" or not math.isfinite(args.peak_photons) or args.peak_photons <= 0:
            parser.error("--peak-photons requires a positive finite level and a Poisson checkpoint")
        evaluation_cfg.degradation.peak_photons = args.peak_photons
    if args.jpeg_quality is not None and study_enabled:
        evaluation_cfg.degradation.jpeg_quality = args.jpeg_quality
    if args.full_image:
        evaluation_cfg.jpeg_study.mode = "off"
        evaluation_cfg.loss.consistency_weight = 0
        if args.jpeg_quality is not None:
            evaluation_cfg.degradation.jpeg_quality = args.jpeg_quality
    dataset = RestorationDataset(evaluation_cfg, args.split)
    if dataset.records:
        if "data_identity" not in state:
            raise ValueError("Checkpoint lacks source identities needed for held-out evaluation")
        other_splits = [r for r in state["data_identity"] if r["split"] != args.split]
        validate_splits(other_splits + dataset.records)
    if args.phase_panel and not study_enabled:
        raise ValueError("Phase panel requires a phase-study checkpoint")
    if args.jpeg_quality is not None and not study_enabled:
        raise ValueError("Quality mismatch evaluation is available for the phase study only")
    frozen_hash = None
    if study_enabled and args.split == "test":
        if not args.frozen_study or (not args.full_image and args.phase_panel != "all"):
            parser.error("Protected phase test requires a frozen study and either all phases or full-image evaluation")
        frozen = json.loads(Path(args.frozen_study).read_text())
        if frozen.get("status") != "frozen" or dataset.fingerprint not in frozen.get("test_manifest_hashes", []):
            raise ValueError("Frozen study does not authorize this test manifest")
        selected = frozen.get("selected_checkpoints", [])
        if file_sha256(args.checkpoint) not in selected:
            raise ValueError("Checkpoint is not among frozen development selections")
        if frozen.get("codec") != codec_metadata():
            raise ValueError("Evaluation JPEG codec differs from the frozen study")
        if frozen.get("source_files") != source_identity()["files"]:
            raise ValueError("Source files changed after study freeze")
        if (args.jpeg_quality or cfg.degradation.jpeg_quality) not in frozen.get("qualities", []):
            raise ValueError("Quality is absent from the frozen protocol")
        frozen_hash = file_sha256(args.frozen_study)
    device = select_device(args.device)
    model = build_model(cfg.model).to(device)
    model.load_state_dict(state["model"], strict=True)
    metrics = (evaluate_phases(model, dataset, device, ALL_PHASES if args.phase_panel == "all" else DEV_PHASES,
                               args.jpeg_quality) if study_enabled and not args.full_image else
               validate(model, dataset, device, cfg.model.scale if cfg.model.scale > 1 else 0))
    report = {**metrics, "protocol": args.protocol, "checkpoint": str(Path(args.checkpoint).resolve()),
              "step": state["step"], "dataset_hash": dataset.fingerprint, "split": args.split,
              "procedural_test_only": cfg.dataset.kind == "procedural", "device": str(device),
              "checkpoint_sha256": file_sha256(args.checkpoint), "training_seed": cfg.training.seed,
              "training_config": {k: v for k, v in cfg.training.__dict__.items() if k != "seed"},
              "selection_horizon": state["total_steps"], "degradation": evaluation_cfg.degradation.__dict__,
              "training_source_hash": hashlib.sha256(json.dumps(state["source"]["files"],sort_keys=True).encode()).hexdigest(),
              "model": cfg.model.__dict__, "jpeg_study": cfg.jpeg_study.__dict__, "codec": codec_metadata(),
              "phase_constructor": "jpeg_aligned_phase_v1" if study_enabled and not args.full_image else None,
              "reference_crop": "center_P_plus_32_then_center_P" if study_enabled and not args.full_image else "full_image",
              "frozen_study_sha256": frozen_hash}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"evaluated={metrics['count']} report={args.output}")


if __name__ == "__main__":
    main()
