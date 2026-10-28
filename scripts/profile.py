"""Bounded actual-trainer update timing, separate from image-quality benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import uuid
# Do not let this filename shadow stdlib profile when torch imports cProfile.
_script_directory = Path(__file__).resolve().parent
sys.path = [str(_script_directory.parent)] + [p for p in sys.path if Path(p).resolve() != _script_directory]
import torch
from train import run_training
from utils.config import load_config
from utils.logger import write_json
from utils.runtime import select_device


def main() -> None:
    """Measure degradation/forward/backward/optimizer updates after warmup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--output", help="New report path; defaults to reports/profile_<unique>.json")
    args = parser.parse_args()
    args.output = args.output or f"reports/profile_{uuid.uuid4().hex[:8]}.json"
    if args.warmup < 0 or args.steps < 1:
        parser.error("warmup >=0 and steps >=1 required")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    cfg = load_config(args.config)
    cfg.training.device, cfg.training.epochs = args.device, None
    cfg.training.max_seconds = args.max_seconds
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.accumulation_steps is not None:
        cfg.training.accumulation_steps = args.accumulation_steps
    if args.precision is not None:
        cfg.training.precision = args.precision
    cfg.training.max_steps = args.warmup + args.steps
    cfg.training.validation_interval = cfg.training.checkpoint_interval = cfg.training.max_steps + 1
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    mps_driver_samples = []
    def record(step: int, seconds: float) -> None:
        if device.type == "mps":
            mps_driver_samples.append(torch.mps.driver_allocated_memory())
        if step > args.warmup:
            durations.append(seconds)
    summary = run_training(cfg, update_callback=record, profile_only=True)
    report = {"device": str(device), "precision": cfg.training.precision,
              "procedural_test_only": cfg.dataset.kind == "procedural", "measured_updates": len(durations),
              "warmup_updates": args.warmup, "median_update_seconds": statistics.median(durations) if durations else None,
              "mean_update_seconds": statistics.mean(durations) if durations else None, "batch_size": cfg.training.batch_size,
              "accumulation_steps": cfg.training.accumulation_steps, "patch_size": cfg.dataset.patch_size,
              "run_dir": summary["run_dir"], "peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None,
              "mps_current_allocated_bytes": None, "mps_driver_allocated_bytes": None,
              "model": cfg.model.__dict__, "jpeg_study": cfg.jpeg_study.__dict__,
              "completed": summary["completed"], "stop_reason": summary["stop_reason"],
              "model_parameters": summary["model_parameters"], "requested_measured_updates": args.steps,
              "views_per_clean_crop": 2 if cfg.jpeg_study.mode != "off" else 1,
              "mps_max_observed_driver_allocated_bytes": max(mps_driver_samples) if mps_driver_samples else None,
              "mps_memory_note": "Driver allocation sampled at optimizer boundaries, not a measured peak or CUDA VRAM",
              "validation_executed": False,
              "timing_scope": "sample loading/degradation, forward, backward, clipping, optimizer and scheduler; excludes validation/log/checkpoint"}
    if device.type == "cuda":
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    elif device.type == "mps":
        report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"profile={args.output} measured_updates={len(durations)}")
    if not summary["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
