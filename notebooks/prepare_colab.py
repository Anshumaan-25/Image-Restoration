"""Package the existing approved train32/dev8 data for private free-cloud use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from cloud_workflow import sha256


def build_bundle(project: Path, output: Path) -> dict:
    """Create a new ZIP with code, two manifests, 40 PNGs and a hash inventory."""
    if any(path.exists() for path in (output, output.with_suffix(".sha256"), output.with_suffix(".json"))):
        raise FileExistsError("Choose a new bundle/checksum/manifest name")
    files: set[Path] = set()
    for directory in ("models", "losses", "utils", "scripts", "tests", "configs"):
        files.update(path for path in (project / directory).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".yaml"})
    for name in ("train.py", "eval.py", "inference.py", "requirements.txt", "AGENTS.md",
                 "README.md", "notebooks/cloud_workflow.py", "notebooks/test_cloud_workflow.py",
                 "notebooks/prepare_colab.py", "docs/cloud_runbook.md"):
        files.add(project / name)
    # Architecture authority accompanies the code; historical run weights do not.
    files.update((project / "docs/architecture").rglob("*.md"))
    counts, identities, hashes = {}, set(), set()
    image_bytes = 0
    for split, expected in (("train", 32), ("val", 8)):
        manifest = project / f"data/manifests/div2k_jpeg_v1/screening/{split}.jsonl"
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        if len(records) != expected:
            raise ValueError(f"Expected {expected} {split} sources")
        start = 1 if split == "train" else 721
        if [record["source_id"] for record in records] != [f"{value:04d}" for value in range(start, start + expected)]:
            raise ValueError("Only the frozen train0001–0032/dev0721–0728 subset may be packaged")
        counts[split] = len(records)
        files.add(manifest)
        for record in records:
            relative = Path(record["target_path"])
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".png":
                raise ValueError("Unsafe or non-PNG data path")
            if record["split"] != split or record["source_id"] in identities or record["sha256"] in hashes:
                raise ValueError("Split/identity/hash leakage")
            identities.add(record["source_id"])
            hashes.add(record["sha256"])
            image = project / relative
            if sha256(image) != record["sha256"]:
                raise ValueError(f"Image hash mismatch: {relative}")
            image_bytes += image.stat().st_size
            files.add(image)
    inventory = {"schema_version": 1, "purpose": "private_cloud_train32_dev8",
                 "split_counts": counts, "image_bytes": image_bytes,
                 "protected_test_included": False, "pretrained_weights_included": False,
                 "files": {str(path.relative_to(project)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                           for path in sorted(files)}}
    if any(path.is_symlink() for path in files):
        raise ValueError("Symlinks cannot enter the transfer bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(files):
            archive.write(path, f"Image_Degradatation_Restoration/{path.relative_to(project)}")
        archive.writestr("Image_Degradatation_Restoration/COLAB_BUNDLE.json", json.dumps(inventory, indent=2))
    result = {"archive": str(output), "sha256": sha256(output), "bytes": output.stat().st_size,
              "file_count": len(files), "split_counts": counts, "image_bytes": image_bytes,
              "protected_test_included": False}
    output.with_suffix(".sha256").write_text(f"{result['sha256']}  {output.name}\n")
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(Path(__file__).resolve().parents[1], args.output.resolve()), indent=2))
