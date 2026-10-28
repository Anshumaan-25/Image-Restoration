"""Read-only backup verification; safe to run in a CPU-only Colab session."""
from google.colab import drive
from pathlib import Path
import hashlib
import json
import zipfile

drive.mount("/content/drive")
study = Path("/content/drive/MyDrive/RestorationProject/studies/cuda_jpeg_5000_v1")

for arm in ("A", "B", "C"):
    snapshots = sorted((study / f"{arm}_seed42").glob("step_*_micro_*.zip"))
    if not snapshots:
        raise FileNotFoundError(f"No complete ZIP found for arm {arm}")
    latest = snapshots[-1]
    with zipfile.ZipFile(latest) as archive:
        metadata = json.loads(archive.read("snapshot.json"))
        if metadata.get("schema_version") != 1 or "latest.pth" not in metadata["files"]:
            raise ValueError(f"Invalid snapshot metadata: {latest.name}")
        if set(archive.namelist()) != set(metadata["files"]) | {"snapshot.json"}:
            raise ValueError(f"Unexpected snapshot files: {latest.name}")
        for name, expected_hash in metadata["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected_hash:
                raise ValueError(f"Backup checksum mismatch: {arm}/{name}")
        summary = (json.loads(archive.read("summary.json"))
                   if "summary.json" in metadata["files"] else {})
    print(f"{arm}: saved update {metadata['step']}/5000; "
          f"best update {metadata['best_step']}; "
          f"completed={summary.get('completed', 'not recorded')}; CHECKSUMS OK")
    print(latest.name)
