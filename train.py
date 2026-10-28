"""Small supervised trainer with offline dry-run and optimizer-boundary resume."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import math
from itertools import count as count_ordinals
import time
import uuid
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from losses import CompositeLoss
from losses.consistency import phase_consistency
from models import build_model
from utils.checkpoint import compatibility_hash, load_checkpoint, save_checkpoint, run_lock
from utils.config import Config, config_from_dict, load_config
from utils.dataset_loader import RestorationDataset, validate_splits
from utils.degradation import codec_metadata
from utils.logger import log_row, write_json
from utils.metrics import aggregate, evaluate_pair
from utils.reproducibility import capture_rng, restore_rng, seed_everything, source_identity
from utils.runtime import environment, precision_context, select_device, synchronize
from utils.phase_evaluation import evaluate_phases


@torch.no_grad()
def validate(model: nn.Module, dataset: RestorationDataset, device: torch.device,
             border: int = 0) -> dict[str, float | int]:
    """Score every fixed image without padded pixels or reduced precision."""
    model.eval()
    if dataset.cfg.jpeg_study.mode != "off":
        result = evaluate_phases(model, dataset, device)
        return {k: v for k, v in result.items() if k != "images"}
    scores = []
    for index in range(len(dataset)):
        item = dataset[index]
        pred = model(item["input"][None].to(device)).float().cpu()[0]
        scores.append(evaluate_pair(pred, item["target"], border))
    return aggregate(scores)


def _schedule(update: int, total: int, warmup: int, minimum_ratio: float) -> float:
    warmup = min(warmup, max(0, total - 1))
    if update < warmup:
        return (update + 1) / warmup
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * progress)) / 2


def run_training(cfg: Config, run_dir: str | Path | None = None, resume: str | Path | None = None,
                 stop_after_updates: int | None = None,
                 update_callback: Callable[[int, float], None] | None = None,
                 profile_only: bool = False) -> dict[str, Any]:
    """Create a unique run or exclusively resume the original run directory."""
    cfg = config_from_dict(cfg.to_dict())
    if resume:
        directory = Path(resume).resolve().parent
        if run_dir is not None and Path(run_dir).resolve() != directory:
            raise ValueError("Resume must retain the original run directory")
    else:
        directory = Path(run_dir) if run_dir is not None else Path(cfg.output.root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        directory.mkdir(parents=True, exist_ok=False)
    with run_lock(directory):
        return _run_training(cfg, directory, resume, stop_after_updates, update_callback, profile_only)


def _run_training(cfg: Config, run_dir: str | Path, resume: str | Path | None,
                  stop_after_updates: int | None, update_callback: Callable[[int, float], None] | None,
                  profile_only: bool) -> dict[str, Any]:
    """Train a configured schedule; optional test interruption preserves its horizon."""
    cfg = config_from_dict(cfg.to_dict())
    t = cfg.training
    torch.set_num_threads(2)
    device = select_device(t.device)
    with precision_context(device, t.precision):
        pass
    seed_everything(t.seed)
    train_data = RestorationDataset(cfg, "train", training=True)
    val_data = RestorationDataset(cfg, "val", training=False)
    validate_splits(train_data.records + val_data.records)
    datasets = {"train": train_data.fingerprint, "val": val_data.fingerprint}
    data_identity = train_data.records + val_data.records
    per_update = t.batch_size * t.accumulation_steps
    sample_limit = t.epochs * len(train_data) if t.epochs is not None else t.max_steps * per_update
    total_steps = math.ceil(math.ceil(sample_limit / t.batch_size) / t.accumulation_steps)
    model = build_model(cfg.model).to(device)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("Reference models are evaluated, not trained")
    criterion = CompositeLoss(cfg.loss.kind, cfg.loss.epsilon, cfg.loss.perceptual_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: _schedule(u, total_steps, t.warmup_steps, t.min_lr/t.learning_rate))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and t.precision == "fp16")
    fingerprint = compatibility_hash(cfg, datasets)
    step, cursor, best, best_step, skipped, microsteps = 0, 0, -float("inf"), 0, 0, 0
    clipped_updates = 0
    consecutive_skipped = 0
    checkpoint = load_checkpoint(resume) if resume is not None else None
    if checkpoint:
        if checkpoint.get("profile_only", False) != profile_only:
            raise ValueError("Profiling and research checkpoints cannot be interchanged on resume")
        if checkpoint["compatibility"] != fingerprint:
            raise ValueError("Resume configuration/data/schedule mismatch; create a new run instead")
        if checkpoint["total_steps"] != total_steps:
            raise ValueError("Resume schedule horizon mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        step, cursor = checkpoint["step"], checkpoint["cursor"]
        best, best_step = checkpoint["best_metric"], checkpoint["best_step"]
        skipped, microsteps = checkpoint["skipped_updates"], checkpoint["microsteps"]
        clipped_updates = checkpoint.get("clipped_updates", 0)
        consecutive_skipped = checkpoint.get("consecutive_skipped", 0)
        restore_rng(checkpoint["rng"])
    if resume:
        directory = Path(resume).resolve().parent
        if run_dir is not None and Path(run_dir).resolve() != directory:
            raise ValueError("Resume must retain the original run directory")
    else:
        directory = Path(run_dir) if run_dir is not None else Path(cfg.output.root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        (directory / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    env = {**environment(device), "codec": codec_metadata(), "procedural_test_only": cfg.dataset.kind == "procedural"}
    source = source_identity()
    if checkpoint:
        if checkpoint["source"]["files"] != source["files"]:
            raise ValueError("Source files changed since checkpoint; create a new run instead of exact resume")
        source = checkpoint["source"]  # Preserve initialization provenance; resume metadata is separate.
    write_json(directory / ("resume_environment.json" if resume else "environment.json"), env)
    write_json(directory / "manifests.json", datasets)
    print(f"device={device} precision={t.precision} run={directory} updates={step}/{total_steps}", flush=True)
    sampler = range(cursor, sample_limit) if t.epochs is not None else count_ordinals(cursor)
    loader = DataLoader(train_data, batch_size=t.batch_size, sampler=sampler,
                        num_workers=t.num_workers, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(t.seed))
    iterator = iter(loader)
    started = time.monotonic()
    last_metrics: dict[str, Any] = {}
    last_validation_step = -1
    stop_reason = "complete"

    def state() -> dict[str, Any]:
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "config": cfg.to_dict(), "step": step, "cursor": cursor,
                "total_steps": total_steps, "microsteps": microsteps, "skipped_updates": skipped,
                "clipped_updates": clipped_updates,
                "consecutive_skipped": consecutive_skipped,
                "best_metric": best, "best_step": best_step, "rng": capture_rng(),
                "compatibility": fingerprint, "datasets": datasets, "environment": env, "source": source,
                "data_identity": data_identity, "profile_only": profile_only,
                "resume_source": source_identity() if resume else None,
                "metric_protocol": "project_rgb_float_v1", "ssim_protocol": "project_ssim_v1"}

    while step < total_steps:
        if step and t.max_seconds is not None and time.monotonic() - started >= t.max_seconds:
            stop_reason = "wall_time_limit"
            break
        if update_callback:
            synchronize(device)
        update_started = time.monotonic()
        group = []
        for _ in range(t.accumulation_steps):
            batch = next(iterator, None)
            if batch is None:
                break
            group.append(batch)
        if not group:
            stop_reason = "data_exhausted"
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        count = sum(batch["input"].shape[0] for batch in group)
        total_loss = 0.0
        for batch in group:
            x, target = batch["input"].to(device), batch["target"].to(device)
            with precision_context(device, t.precision):
                if "input_alt" in batch:
                    both = model(torch.cat((x, batch["input_alt"].to(device)), dim=0))
                    pred, alternative = both.chunk(2, dim=0)
                    loss, _ = criterion(both, torch.cat((target, target), dim=0))
                    loss = loss + cfg.loss.consistency_weight * phase_consistency(pred, alternative)
                else:
                    pred = model(x)
                    loss, _ = criterion(pred, target)
            weight = x.shape[0] / count
            scaler.scale(loss * weight).backward()
            total_loss += float(loss.detach()) * weight
            cursor += x.shape[0]
            microsteps += 1
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(model.parameters(), t.clip_grad)
        if not scaler.is_enabled() and not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradients; optimizer was not stepped")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        successful = scaler.get_scale() >= previous_scale
        optimizer.zero_grad(set_to_none=True)
        if successful:
            step += 1
            consecutive_skipped = 0
            clipped_updates += int(torch.isfinite(norm) and norm > t.clip_grad)
            scheduler.step()
            if update_callback:
                synchronize(device)
                update_callback(step, time.monotonic() - update_started)
        else:
            skipped += 1
            consecutive_skipped += 1
            if consecutive_skipped >= 16:
                raise FloatingPointError("16 consecutive AMP overflow skips; inspect precision, data and learning rate")
        time_limit_reached = t.max_seconds is not None and time.monotonic() - started >= t.max_seconds and step < total_steps
        if not profile_only and not time_limit_reached and successful and (step % t.validation_interval == 0 or step == total_steps):
            last_metrics = validate(model, val_data, device, cfg.model.scale if cfg.model.scale > 1 else 0)
            last_validation_step = step
            selection = last_metrics.get("worst_phase_psnr", last_metrics["psnr"])
            if selection > best:
                best, best_step = selection, step
                save_checkpoint(directory / "best_model.pth", state())
        log_row(directory / "metrics.csv", {"step": step, "microsteps": microsteps, "loss": total_loss,
                 "learning_rate": optimizer.param_groups[0]["lr"],
                 "psnr": last_metrics.get("psnr") if last_validation_step == step else None,
                 "ssim": last_metrics.get("ssim") if last_validation_step == step else None,
                 "worst_phase_psnr": last_metrics.get("worst_phase_psnr") if last_validation_step == step else None,
                 "validation_step": last_validation_step if last_validation_step == step else None,
                 "skipped_updates": skipped, "clipped_updates": clipped_updates,
                 "gradient_norm_before_clip": float(norm)})
        if successful and step % t.checkpoint_interval == 0:
            save_checkpoint(directory / "latest.pth", state())
        if stop_after_updates is not None and step >= stop_after_updates and step < total_steps:
            stop_reason = "test_interruption"
            break
        if time_limit_reached:
            stop_reason = "wall_time_limit"
            break
    if not profile_only and stop_reason != "wall_time_limit" and last_validation_step != step:
        last_metrics = validate(model, val_data, device, cfg.model.scale if cfg.model.scale > 1 else 0)
        selection = last_metrics.get("worst_phase_psnr", last_metrics["psnr"])
        if selection > best:
            best, best_step = selection, step
            save_checkpoint(directory / "best_model.pth", state())
    save_checkpoint(directory / "latest.pth", state())
    synchronize(device)
    summary = {"run_dir": str(directory.resolve()), "step": step, "target_steps": total_steps, "cursor": cursor,
               "completed": step == total_steps, "stop_reason": stop_reason, "best_step": best_step,
               "skipped_updates": skipped, "microsteps": microsteps, "elapsed_seconds": time.monotonic()-started,
               "clipped_updates": clipped_updates,
               "procedural_test_only": cfg.dataset.kind == "procedural", **last_metrics}
    summary["profile_only"] = profile_only
    summary["model_parameters"] = sum(p.numel() for p in model.parameters())
    write_json(directory / "summary.json", summary)
    return summary


def main() -> None:
    """CLI; dry-run intentionally selects a tiny offline test configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    if args.epochs is not None and args.max_steps is not None and not args.dry_run:
        parser.error("--epochs and --max-steps are mutually exclusive")
    cfg = load_config(Path(__file__).parent / "configs/smoke.yaml" if args.dry_run else args.config)
    if args.device:
        cfg.training.device = args.device
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.max_seconds is not None:
        cfg.training.max_seconds = args.max_seconds
    if not args.dry_run:
        if args.epochs is not None:
            cfg.training.epochs, cfg.training.max_steps = args.epochs, None
        elif args.max_steps is not None:
            cfg.training.max_steps, cfg.training.epochs = args.max_steps, None
        if cfg.training.device == "auto":
            parser.error("Research training requires --device cpu, mps or cuda explicitly")
    elif args.resume:
        parser.error("Use a saved config with --resume; dry-run starts a new offline test")
    result = run_training(cfg, args.run_dir, args.resume)
    # Verify saved state is actually loadable, not merely written.
    loaded = load_checkpoint(Path(result["run_dir"]) / "latest.pth")
    model = build_model(config_from_dict(loaded["config"]).model)
    model.load_state_dict(loaded["model"], strict=True)
    with torch.no_grad():
        model.eval()(torch.rand(1, 3, 13, 17))
    print(f"completed={result['completed']} steps={result['step']} checkpoint/reload passed", flush=True)
    if not result["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
