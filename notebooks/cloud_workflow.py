"""Thin free-cloud launcher: shared trainer, matched plans and private snapshots.

No downloads, account operations or training occur on import. Keep this helper
outside the frozen research-source directories; record its hash separately.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
from typing import Any
import uuid
import zipfile


def sha256(path: Path) -> str:
    """Hash a file without loading an image archive into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if sys.version_info >= (3, 11) else _hash_stream(stream)


def _hash_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Reject traversal, duplicate names and symlinks before extracting anything."""
    entries = archive.infolist()
    names: set[str] = set()
    for entry in entries:
        path = PurePosixPath(entry.filename)
        if (not entry.filename or "\\" in entry.filename or path.is_absolute()
                or any(part in {".", ".."} for part in entry.filename.split("/"))
                or stat.S_ISLNK(entry.external_attr >> 16) or entry.filename in names):
            raise ValueError(f"Unsafe archive member: {entry.filename}")
        names.add(entry.filename)
    return entries


def verify_bundle(project: Path) -> dict[str, Any]:
    """Check the complete transfer inventory, including all 40 image hashes."""
    inventory = json.loads((project / "COLAB_BUNDLE.json").read_text())
    for name, expected in inventory["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe inventory path")
        path = project / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Transfer integrity failure: {name}")
    if inventory["split_counts"] != {"train": 32, "val": 8} or inventory["protected_test_included"]:
        raise ValueError("Expected only the approved screening train32/dev8 payload")
    return inventory


def prepare_plan(project: Path, destination: Path, max_seconds: int = 3600) -> Path:
    """Write three fresh CUDA/FP16 5k arms; planning never starts training."""
    import yaml
    from utils.config import config_from_dict, load_config
    from utils.study import study_plan

    if Path.cwd().resolve() != project.resolve():
        raise ValueError("Run from the project root so manifest paths remain portable")
    if type(max_seconds) is not int or max_seconds < 1:
        raise ValueError("Positive integral seconds required")
    destination.mkdir(parents=True, exist_ok=False)
    arms = {}
    for arm, filename in {"A": "random_phase", "B": "stratified_phase", "C": "phase_consistency"}.items():
        cfg = load_config(project / f"configs/jpeg/{filename}.yaml")
        cfg.training.device, cfg.training.precision = "cuda", "fp16"
        cfg.training.batch_size, cfg.training.accumulation_steps = 2, 4
        cfg.training.max_steps, cfg.training.epochs = 5000, None
        cfg.training.max_seconds = max_seconds
        cfg.training.seed = 42
        cfg = config_from_dict(cfg.to_dict())
        path = destination / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        arms[arm] = str(path.relative_to(project))
    study = destination / "study.yaml"
    study.write_text(yaml.safe_dump({"name": destination.name, "arms": arms, "seeds": [42],
                                    "max_seconds_per_run": max_seconds}, sort_keys=False))
    plan = study_plan(study, "cuda")
    plan["cloud_launcher_sha256"] = sha256(Path(__file__))
    plan["cloud_policy"] = {"spending_usd": 0, "long_run_approved": False,
                            "local_mps_baseline_reused": False, "protected_test_included": False}
    plan_file = destination / "plan.json"
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    return plan_file


def seal_profile(project: Path, plan_file: Path, profile_file: Path) -> Path:
    """Bind a completed actual-device profile/environment to a new review file."""
    import torch
    from utils.degradation import codec_metadata
    from utils.reproducibility import source_identity
    from utils.runtime import environment, select_device

    device = select_device("cuda")
    profile = json.loads(profile_file.read_text())
    expected = {"device": "cuda", "precision": "fp16", "batch_size": 2,
                "accumulation_steps": 4, "patch_size": 128, "measured_updates": 50,
                "completed": True, "views_per_clean_crop": 2, "validation_executed": False}
    if any(profile.get(key) != value for key, value in expected.items()):
        raise ValueError("Actual CUDA profile does not match the proposed workload")
    allocated, reserved = profile["peak_cuda_allocated_bytes"], profile["peak_cuda_reserved_bytes"]
    total = torch.cuda.get_device_properties(device).total_memory
    if not isinstance(allocated, int) or not isinstance(reserved, int) or allocated > 12 * 1024**3 or reserved > .95 * total:
        raise ValueError("Memory headroom insufficient; review common settings before training")
    plan = json.loads(plan_file.read_text())
    import math
    seconds = profile.get("mean_update_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Finite positive measured update time required")
    if any(seconds * run["config"]["training"]["max_steps"] >= run["config"]["training"]["max_seconds"]
           for run in plan["runs"]):
        raise ValueError("Update-only projection exhausts the cap; revise the common budget before training")
    plan["cuda_profile"] = {"path": str(profile_file.relative_to(project)), "sha256": sha256(profile_file),
                            "update_only_projected_seconds_all_arms": seconds * plan["total_optimizer_updates"],
                            "projection_excludes": "validation, checkpoints, Drive backups and setup"}
    plan["cuda_environment"] = {**environment(device), "codec": codec_metadata()}
    plan["training_source_files"] = source_identity()["files"]
    reviewed = plan_file.with_name("reviewed_plan.json")
    with reviewed.open("x") as stream:
        stream.write(json.dumps(plan, indent=2) + "\n")
    return reviewed


def snapshot_run(run: Path, destination: Path) -> Path | None:
    """Save a checkpoint-consistent ZIP; call only while this trainer is paused.

    Logs newer than latest.pth are omitted. Older immutable snapshots are kept.
    A mismatch between best/latest defers the snapshot instead of mixing states.
    """
    from utils.checkpoint import load_checkpoint

    if not (run / "latest.pth").is_file():
        return None
    latest = load_checkpoint(run / "latest.pth")
    payload = {"latest.pth": (run / "latest.pth").read_bytes()}
    if latest["best_step"]:
        best = load_checkpoint(run / "best_model.pth")
        if best["step"] != latest["best_step"] or best["compatibility"] != latest["compatibility"]:
            return None
        payload["best_model.pth"] = (run / "best_model.pth").read_bytes()
    for name in ("config.yaml", "environment.json", "resume_environment.json", "manifests.json"):
        if (run / name).is_file():
            payload[name] = (run / name).read_bytes()
    if (run / "metrics.csv").is_file():
        with (run / "metrics.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = [row for row in reader if int(row["step"]) <= latest["step"]
                    and int(row["microsteps"]) <= latest["microsteps"]]
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            payload["metrics.csv"] = output.getvalue().encode()
    if (run / "summary.json").is_file():
        summary = json.loads((run / "summary.json").read_text())
        if summary["step"] == latest["step"] and summary["microsteps"] == latest["microsteps"]:
            payload["summary.json"] = (run / "summary.json").read_bytes()
    manifest = {"schema_version": 1, "step": latest["step"], "microsteps": latest["microsteps"],
                "best_step": latest["best_step"], "files": {
                    name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}}
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"step_{latest['step']:06d}_micro_{latest['microsteps']:08d}.zip"
    if target.exists():
        if inspect_snapshot(target)["files"] != manifest["files"]:
            raise FileExistsError(f"Different snapshot already exists: {target}")
        return target
    # Build locally before transferring; a killed Drive transfer leaves .partial.
    import tempfile
    with tempfile.TemporaryFile() as temporary:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, data in payload.items():
                archive.writestr(name, data)
            archive.writestr("snapshot.json", json.dumps(manifest, sort_keys=True))
        temporary.seek(0)
        partial = target.with_suffix(f".{uuid.uuid4().hex}.partial")
        with partial.open("xb") as output:
            import shutil
            shutil.copyfileobj(temporary, output)
        inspect_snapshot(partial)
        os.replace(partial, target)
    print(f"Saved backup: {target}", flush=True)
    return target


def inspect_snapshot(path: Path) -> dict[str, Any]:
    """Reject corrupt/incomplete/unexpected snapshots without loading a model."""
    with zipfile.ZipFile(path) as archive:
        entries = safe_members(archive)
        manifest = json.loads(archive.read("snapshot.json"))
        allowed = {"latest.pth", "best_model.pth", "config.yaml", "environment.json",
                   "resume_environment.json", "manifests.json", "metrics.csv", "summary.json"}
        if (manifest.get("schema_version") != 1 or not set(manifest["files"]) <= allowed
                or "latest.pth" not in manifest["files"]
                or {entry.filename for entry in entries} != set(manifest["files"]) | {"snapshot.json"}):
            raise ValueError("Unexpected snapshot inventory")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Corrupt snapshot: {name}")
    return manifest


def restore_snapshot(path: Path, destination: Path) -> Path:
    """Restore a verified private snapshot into a new directory; never overwrite."""
    inspect_snapshot(path)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)
    return destination / "latest.pth"


def run_arm(project: Path, plan_file: Path, arm_id: str, backup_root: Path,
            approved: bool = False, resume_archive: Path | None = None) -> dict[str, Any]:
    """Execute one explicitly approved arm, backing up every 200 updates and exit.

    A resume is a separate invocation: review its remaining wall budget first.
    No automatic restart, device fallback, test scoring or next-arm execution.
    """
    if approved is not True:
        raise PermissionError("Review the actual CUDA profile and bounded plan before enabling long training")
    import torch
    from train import run_training
    from utils.checkpoint import load_checkpoint
    from utils.config import config_from_dict
    from utils.degradation import codec_metadata
    from utils.reproducibility import source_identity
    from utils.runtime import environment, select_device

    if Path.cwd().resolve() != project.resolve():
        raise ValueError("Run from the project root")
    device = select_device("cuda")
    plan = json.loads(plan_file.read_text())
    if plan["cloud_launcher_sha256"] != sha256(Path(__file__)):
        raise ValueError("Cloud helper changed since planning; review a new plan")
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", plan["name"]):
        raise ValueError("Unsafe study name")
    profile_path = project / plan["cuda_profile"]["path"]
    if sha256(profile_path) != plan["cuda_profile"]["sha256"]:
        raise ValueError("CUDA profile identity changed")
    current = {**environment(device), "codec": codec_metadata()}
    for key in ("python", "torch", "cuda", "device_name", "dependencies", "codec"):
        if current.get(key) != plan["cuda_environment"].get(key):
            raise ValueError(f"Study environment changed: {key}; do not mix unmatched arms")
    if source_identity()["files"] != plan["training_source_files"]:
        raise ValueError("Training source changed since CUDA preflight")
    for manifest in plan["manifests"].values():
        if sha256(project / manifest["path"]) != manifest["sha256"]:
            raise ValueError("Study manifest changed since planning")
    matches = [run for run in plan["runs"] if run["id"] == arm_id]
    if len(matches) != 1:
        raise ValueError("Select exactly one planned arm")
    cfg = config_from_dict(matches[0]["config"])
    if cfg.training.device != "cuda" or cfg.training.precision != "fp16" or cfg.training.max_seconds is None:
        raise ValueError("This reviewed launcher requires bounded CUDA/FP16")
    run = project / "experiments/cloud" / plan["name"] / arm_id
    resume = None
    if resume_archive is not None:
        run = run.parent / f"{arm_id}_resume_{uuid.uuid4().hex[:8]}"
        resume = restore_snapshot(resume_archive, run)
        prior = load_checkpoint(resume)["environment"]
        for key in ("python", "torch", "cuda", "device_name", "dependencies", "codec"):
            if prior.get(key) != current.get(key):
                raise ValueError(f"Resume environment changed: {key}; retain the snapshot for review")
    destination = backup_root / plan["name"] / run.name
    destination.mkdir(parents=True, exist_ok=True)
    # Preserve the reviewed plan/helper identity beside every arm's snapshots.
    for name, data in {"plan.json": plan_file.read_bytes(), "cloud_workflow.py": Path(__file__).read_bytes()}.items():
        target = destination / name
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(target)
        if not target.exists():
            target.write_bytes(data)
    def backup(step: int, seconds: float) -> None:
        if step >= 101 and (step - 101) % 200 == 0:
            snapshot_run(run, destination)
    started = time.monotonic()
    try:
        result = run_training(cfg, run_dir=run, resume=resume, update_callback=backup)
    finally:
        try:
            snapshot_run(run, destination)
        finally:
            torch.cuda.empty_cache()
    print(f"Arm elapsed including setup/backup: {time.monotonic() - started:.1f}s", flush=True)
    if not result["completed"]:
        raise RuntimeError("Arm incomplete: retained checkpoint/backup. Review before resuming; stop the study here.")
    return result
