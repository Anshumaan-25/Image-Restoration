"""Verify approved local DIV2K archives and prepare immutable project splits.

This script never downloads data or starts training. Archives stay in data/raw.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import zipfile

_script_directory = Path(__file__).resolve().parent
sys.path = [str(_script_directory.parent)] + [p for p in sys.path if Path(p).resolve() != _script_directory]

import torch
from utils.dataset_loader import manifest_fingerprint, read_manifest, validate_splits
from utils.io import file_sha256, load_rgb


@dataclass(frozen=True)
class ArchiveSpec:
    """Approved archive identity and its exact expected image IDs."""
    folder: str
    first: int
    last: int
    bytes: int


ARCHIVES = (ArchiveSpec("DIV2K_train_HR", 1, 800, 3530603713),
            ArchiveSpec("DIV2K_valid_HR", 801, 900, 448993893))


def verify_members(archive: zipfile.ZipFile, spec: ArchiveSpec) -> list[zipfile.ZipInfo]:
    """Reject wrong images, duplicate entries, links and unexpected paths."""
    expected = {f"{spec.folder}/{index:04d}.png" for index in range(spec.first, spec.last + 1)}
    files = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"Unsafe archive member: {member.filename}")
        if member.is_dir() and member.filename == spec.folder + "/":
            continue
        if member.filename not in expected or member.flag_bits & 1:
            raise ValueError(f"Unexpected archive member: {member.filename}")
        files.append(member)
    if len(files) != len(expected) or {m.filename for m in files} != expected:
        raise ValueError("Archive image IDs are missing or duplicated")
    return sorted(files, key=lambda m: m.filename)


def extract_verified_archive(path: Path, destination: Path, spec: ArchiveSpec) -> dict[str, Any]:
    """Verify ZIP integrity, then create files without replacing existing data."""
    if path.stat().st_size != spec.bytes:
        raise ValueError(f"Archive size mismatch: {path}")
    print(f"Checking SHA256 and ZIP CRC: {path.name}", flush=True)
    sha256 = file_sha256(path)
    reused = 0
    with zipfile.ZipFile(path) as archive:
        members = verify_members(archive, spec)
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Archive CRC failure: {corrupt}")
        for ordinal, member in enumerate(members, 1):
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with archive.open(member) as source:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                if target.is_symlink() or file_sha256(target) != digest.hexdigest():
                    raise FileExistsError(f"Existing image differs; not overwritten: {target}")
                reused += 1
            else:
                # Stage our new file, then link exclusively; never replace a raced destination.
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".extracting", delete=False) as temporary:
                    staged = Path(temporary.name)
                    try:
                        with archive.open(member) as source:
                            shutil.copyfileobj(source, temporary, length=1024 * 1024)
                        temporary.flush()
                        os.fsync(temporary.fileno())
                    except BaseException:
                        staged.unlink(missing_ok=True)
                        raise
                try:
                    os.link(staged, target)
                finally:
                    staged.unlink(missing_ok=True)  # Our staging copy; the verified image remains.
            if ordinal % 50 == 0 or ordinal == len(members):
                print(f"Extracted/verified {spec.folder}: {ordinal}/{len(members)}", flush=True)
    return {"name": spec.folder + ".zip", "bytes": spec.bytes, "sha256": sha256,
            "images": len(members), "existing_images_reused": reused, "zip_crc": "passed",
            "url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/" + spec.folder + ".zip"}


def immutable_write(path: Path, text: str) -> None:
    """Permit idempotent reuse, but refuse to change an existing manifest."""
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Existing manifest differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)


def build_manifests(root: Path, archives: list[dict[str, Any]]) -> dict[str, Any]:
    """Decode/hash all 900 references and create the approved full/pilot splits."""
    all_records = []
    hashes = {a["name"].removesuffix(".zip"): a["sha256"] for a in archives}
    for index in range(1, 901):
        folder = "DIV2K_train_HR" if index <= 800 else "DIV2K_valid_HR"
        relative = Path("data/raw") / folder / f"{index:04d}.png"
        image = load_rgb(root / relative)
        split = "train" if index <= 720 else ("val" if index <= 800 else "test")
        all_records.append({"id": f"{index:04d}", "dataset": "DIV2K", "version": "official_HR_2017",
            "split": split, "source_id": f"{index:04d}", "scene_id": None,
            "input_path": None, "input_sha256": None, "target_path": relative.as_posix(),
            "sha256": file_sha256(root / relative), "height": image.shape[-2], "width": image.shape[-1],
            "bit_depth": 8, "color_space": "sRGB", "scale": 1, "reference_domain": "clean_RGB_PNG",
            "archive": folder + ".zip", "archive_sha256": hashes[folder],
            "split_protocol": "div2k_jpeg_project_v1"})
        if index % 50 == 0:
            print(f"Decoded and hashed references: {index}/900", flush=True)
    validate_splits(all_records)
    manifests = {}
    directory = root / "data/manifests/div2k_jpeg_v1"
    for split, pilot_count in (("train", 32), ("val", 8), ("test", 16)):
        records = [r for r in all_records if r["split"] == split]
        for pilot in (False, True):
            selected = records[:pilot_count] if pilot else records
            path = directory / ("screening" if pilot else "full") / (split + ".jsonl")
            immutable_write(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in selected))
            manifests[str(path.relative_to(root))] = {"count": len(selected), "file_sha256": file_sha256(path),
                "records_sha256": manifest_fingerprint(selected), "first_id": selected[0]["id"],
                "last_id": selected[-1]["id"]}
    # Independently exercise the same content/geometry validation used at training startup.
    verified = []
    for split in ("train", "val", "test"):
        print(f"Independent full-manifest validation: {split}", flush=True)
        verified.extend(read_manifest(directory / "full" / (split + ".jsonl"), root))
    validate_splits(verified)
    return {"manifests": manifests, "total_images": len(all_records), "split_leakage_checks": "passed",
            "dimensions": {"min_height": min(r["height"] for r in all_records),
                           "min_width": min(r["width"] for r in all_records),
                           "max_height": max(r["height"] for r in all_records),
                           "max_width": max(r["width"] for r in all_records)}}


def main() -> None:
    """Prepare only already-approved, already-downloaded local archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_script_directory.parent)
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(2)
    started = time.monotonic()
    artifacts = []
    for spec in ARCHIVES:
        path = root / "data/raw" / (spec.folder + ".zip")
        partial = path.with_suffix(path.suffix + ".part")
        source = path if path.exists() else partial
        artifacts.append(extract_verified_archive(source, root / "data/raw", spec))
        if source == partial:
            os.link(partial, path)  # Exclusive finalization; retain content if another job races us.
            partial.unlink()
    report = {"status": "acquired_and_validated", "approval_date_local": "2026-10-21",
        "approval": "User explicitly approved both official DIV2K HR archives and split preparation",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "archives": artifacts,
        "integrity_note": "Size, local SHA256, exact file set and ZIP CRC verified; no publisher SHA256 was supplied",
        **build_manifests(root, artifacts), "elapsed_seconds": time.monotonic() - started,
        "training_started": False, "pretrained_weights_downloaded": False}
    report["elapsed_seconds"] = time.monotonic() - started
    output = root / "reports/div2k_acquisition.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"DIV2K acquisition complete: {output}", flush=True)


if __name__ == "__main__":
    main()
