"""Manifest-backed restoration pairs and deterministic procedural test fixtures."""
from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
from typing import Any
import hashlib
import json
import torch
from torch.utils.data import Dataset

from utils.config import Config
from utils.degradation import apply_gaussian_noise, apply_poisson_noise, bicubic_fixture, jpeg_roundtrip
from utils.io import file_sha256, load_rgb
from utils.reproducibility import stable_seed
from utils.jpeg_phase import clean_context, phase_pair, phase_view


def read_manifest(path: str | Path, root: str | Path) -> list[dict[str, Any]]:
    """Validate content, decoded metadata and explicit pair geometry before a run."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("Manifest contains no images")
    required = {"id", "dataset", "version", "split", "source_id", "target_path", "sha256",
                "height", "width", "bit_depth", "color_space", "scale"}
    ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError("Manifest record is missing required metadata")
        for key in ("id", "dataset", "version", "split", "source_id", "target_path", "sha256"):
            if not isinstance(record[key], str) or not record[key]:
                raise ValueError(f"Manifest {key} must be a nonempty string")
        if record["id"] in ids:
            raise ValueError(f"Duplicate image ID: {record['id']}")
        ids.add(record["id"])
        if record["bit_depth"] != 8 or record["color_space"] != "sRGB":
            raise ValueError("Only 8-bit sRGB manifests are supported")
        if any(type(record[k]) is not int or record[k] < 1 for k in ("height", "width")):
            raise ValueError("Manifest dimensions must be positive integers")
        if type(record["scale"]) is not int or record["scale"] not in (1, 2, 4):
            raise ValueError("Invalid manifest scale")
        for field in ("target_path", "input_path"):
            if not record.get(field):
                continue
            relative = Path(record[field])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Manifest image paths must be relative to the data root")
            filename = Path(root) / relative
            expected_hash = record["sha256"] if field == "target_path" else record.get("input_sha256")
            if expected_hash is None or file_sha256(filename) != expected_hash:
                raise ValueError(f"Content hash mismatch: {filename}")
            image = load_rgb(filename)
            expected = (record["height"], record["width"])
            if field == "input_path":
                expected = tuple(v // record["scale"] for v in expected)
            if image.shape[-2:] != expected:
                raise ValueError(f"Pair geometry/metadata mismatch: {filename}")
        if record["height"] % record["scale"] or record["width"] % record["scale"]:
            raise ValueError("Target dimensions not divisible by scale")
    validate_splits(records)
    return records


def validate_splits(records: list[dict[str, Any]]) -> None:
    """Reject parent/source, known-scene and content identity crossing splits."""
    seen: dict[tuple[str, str], str] = {}
    ids: set[tuple[str, str]] = set()
    for r in records:
        pair = (r["split"], r["id"])
        if pair in ids:
            raise ValueError(f"Duplicate record: {pair}")
        ids.add(pair)
        for key in ("source_id", "scene_id", "sha256", "input_sha256"):
            if r.get(key) is None:
                continue
            # Dataset-qualified source IDs; file hashes are globally compared.
            namespace = "content" if key.endswith("sha256") else f"{r['dataset']}:{key}"
            identity = (namespace, str(r[key]))
            if identity in seen and seen[identity] != r["split"]:
                raise ValueError(f"Data leakage across splits: {key}={r[key]}")
            seen[identity] = r["split"]


def manifest_fingerprint(records: list[dict[str, Any]]) -> str:
    """Hash immutable metadata, including verified image content hashes."""
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def procedural_image(size: int, seed: int) -> torch.Tensor:
    """Authored gradients/checker texture; explicitly not a natural-image benchmark."""
    generator = torch.Generator().manual_seed(seed)
    u = torch.linspace(0, 1, size)
    yy, xx = torch.meshgrid(u, u, indexing="ij")
    phase = torch.rand(3, 1, 1, generator=generator)
    signal = torch.stack((xx, yy, (torch.sin(xx * 35) * torch.cos(yy * 29) + 1) / 2))
    return (0.7 * signal + 0.2 * phase + 0.1 * torch.rand(3, size, size, generator=generator)).clamp(0, 1)


class RestorationDataset(Dataset):
    """Index is a sample ordinal; its RNG is independent of worker/prefetch order."""

    def __init__(self, cfg: Config, split: str, training: bool = False,
                 manifest: str | Path | None = None) -> None:
        self.cfg, self.split, self.training = cfg, split, training
        self._clean_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._cache_bytes = 0
        self.records: list[dict[str, Any]] = []
        if cfg.dataset.kind == "manifest":
            path = manifest or (cfg.dataset.train_manifest if training else cfg.dataset.val_manifest)
            if path is None:
                raise ValueError("A manifest path is required for real-data runs")
            self.records = read_manifest(Path(cfg.dataset.root) / path, cfg.dataset.root)
            if {r["split"] for r in self.records} != {split}:
                raise ValueError(f"Expected only {split!r} records in this manifest")
            paired = cfg.degradation.profile in {"sidd_srgb_pairs_v1", "div2k_bicubic_pairs_v1"}
            if any(bool(r.get("input_path")) != paired for r in self.records):
                raise ValueError("Measured profiles require explicit pairs; synthetic profiles require clean-only sources")
            if any(r["scale"] != cfg.model.scale for r in self.records):
                raise ValueError("Manifest scale conflicts with the model")
            required_size = cfg.dataset.patch_size + (32 if cfg.jpeg_study.mode != "off" else 0)
            if (training or cfg.jpeg_study.mode != "off") and any(min(r["height"], r["width"]) < required_size for r in self.records):
                raise ValueError("Training source is smaller than requested target patch")
        self.fingerprint = manifest_fingerprint(self.records) if self.records else f"procedural_v1:{split}:{cfg.dataset.patch_size}:{len(self)}"

    def __len__(self) -> int:
        """Number of source images, not the number of random crops."""
        return len(self.records) if self.records else (self.cfg.dataset.train_samples if self.training else self.cfg.dataset.val_samples)

    def clean_source(self, index: int) -> tuple[torch.Tensor, str]:
        """Load an owned reference, optionally reusing a bounded decoded CPU cache.

        The cache changes only I/O cost, never sample keys or tensor values. Return
        an independent tensor so callers cannot corrupt a later sample in place.
        Sources larger than the explicit byte budget are read without caching.
        """
        if self.records:
            record = self.records[index]
            budget = self.cfg.dataset.cache_max_mib * 1024**2
            if index in self._clean_cache:
                image = self._clean_cache.pop(index)
                self._clean_cache[index] = image
                return image.clone(), record["id"]
            image = load_rgb(Path(self.cfg.dataset.root) / record["target_path"])
            size = image.numel() * image.element_size()
            if budget and size <= budget:
                while self._cache_bytes + size > budget:
                    _, evicted = self._clean_cache.popitem(last=False)
                    self._cache_bytes -= evicted.numel() * evicted.element_size()
                self._clean_cache[index] = image
                self._cache_bytes += size
                return image.clone(), record["id"]
            return image, record["id"]
        size = self.cfg.dataset.patch_size + (32 if self.cfg.jpeg_study.mode != "off" else 0)
        return procedural_image(size, stable_seed(self.cfg.training.seed, self.split, index, "content")), f"procedural_{self.split}_{index}"

    def __getitem__(self, ordinal: int) -> dict[str, Any]:
        """Return CPU input/target and source metadata; validation does not crop."""
        cfg, scale = self.cfg, self.cfg.model.scale
        index = ordinal % len(self)
        key = stable_seed(cfg.training.seed, self.split, ordinal)
        generator = torch.Generator().manual_seed(key)
        if self.records:
            record = self.records[index]
            target, image_id = self.clean_source(index)
            degraded = load_rgb(Path(cfg.dataset.root) / record["input_path"]) if record.get("input_path") else None
        else:
            target, image_id = self.clean_source(index)
            degraded = None
        if cfg.jpeg_study.mode != "off":
            context = clean_context(target, cfg.dataset.patch_size, generator if self.training else None,
                                    cfg.jpeg_study.augment and self.training)
            phases = phase_pair(cfg.training.seed, ordinal, cfg.jpeg_study.mode) if self.training else ((0, 0), (4, 0))
            first, target = phase_view(context, cfg.dataset.patch_size, phases[0], cfg.degradation.jpeg_quality)
            second, _ = phase_view(context, cfg.dataset.patch_size, phases[1], cfg.degradation.jpeg_quality)
            return {"input": first, "input_alt": second, "target": target, "id": image_id,
                    "metadata": {"sample_ordinal": ordinal, "seed": key, "profile": cfg.degradation.profile,
                                 "phase_a": phases[0], "phase_b": phases[1]}}
        if cfg.degradation.profile == "jpeg_rgb_444_v1":
            # Encode the whole source first; cropping afterward retains its grid phase.
            degraded = jpeg_roundtrip(target, cfg.degradation.jpeg_quality)
        if self.training:
            p = cfg.dataset.patch_size
            # Pick an LR-aligned coordinate; never interpolate to manufacture a pair.
            top = int(torch.randint((target.shape[-2] - p) // scale + 1, (), generator=generator)) * scale
            left = int(torch.randint((target.shape[-1] - p) // scale + 1, (), generator=generator)) * scale
            target = target[:, top:top+p, left:left+p]
            if degraded is not None:
                degraded = degraded[:, top//scale:(top+p)//scale, left//scale:(left+p)//scale]
        g = cfg.degradation
        operator_metadata: dict[str, Any] = {}
        if degraded is None:
            if g.profile.startswith("awgn_"):
                sigma = float(torch.empty(()).uniform_(g.sigma_min, g.sigma_max, generator=generator)) if self.training else g.sigma
                degraded = apply_gaussian_noise(target, sigma, generator, clipped=g.profile == "awgn_clipped_v1")
                operator_metadata["sigma_8bit"] = sigma
            elif g.profile == "poisson_linear_rgb_v1":
                peak = (g.poisson_training_peaks[int(torch.randint(len(g.poisson_training_peaks),(),generator=generator))]
                        if self.training else g.peak_photons)
                degraded = apply_poisson_noise(target, peak, generator)
                operator_metadata["peak_photons"] = peak
            elif g.profile == "jpeg_rgb_444_v1":
                degraded = jpeg_roundtrip(target, g.jpeg_quality)
            elif g.profile == "torch_bicubic_fixture_v1":
                degraded = bicubic_fixture(target, scale)
            else:
                raise ValueError("Measured degradation requires an explicit input pair")
        return {"input": degraded.contiguous(), "target": target.contiguous(), "id": image_id,
                "metadata": {"sample_ordinal": ordinal, "seed": key, "profile": g.profile, **operator_metadata}}
