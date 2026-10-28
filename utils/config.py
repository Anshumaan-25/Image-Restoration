"""Strict restoration configuration, including matched JPEG study and learned SR."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import math

import yaml


@dataclass
class ModelConfig:
    """Architecture identity; nonstandard sizes are labeled test variants."""
    name: str = "dncnn_rgb_l1"
    width: int = 64
    depth: int = 20
    scale: int = 1
    blocks: int = 16


@dataclass
class DatasetConfig:
    """Manifest paths resolve from root; procedural images are tests only."""
    kind: str = "manifest"
    root: str = "."
    train_manifest: str | None = None
    val_manifest: str | None = None
    patch_size: int = 128
    train_samples: int = 32
    val_samples: int = 8
    cache_max_mib: int = 0


@dataclass
class DegradationConfig:
    """One named operator per experiment."""
    profile: str = "jpeg_rgb_444_v1"
    sigma: float = 25.0
    sigma_min: float = 10.0
    sigma_max: float = 50.0
    peak_photons: float = 60.0
    poisson_training_peaks: list[float] = field(default_factory=lambda: [30., 60., 120.])
    jpeg_quality: int = 20


@dataclass
class TrainingConfig:
    """Successful optimizer updates define the schedule."""
    device: str = "auto"
    precision: str = "fp32"
    batch_size: int = 8
    accumulation_steps: int = 2
    num_workers: int = 0
    max_steps: int | None = 1000
    epochs: int | None = None
    max_seconds: float | None = None
    learning_rate: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 100
    clip_grad: float = 1.0
    seed: int = 42
    validation_interval: int = 100
    checkpoint_interval: int = 100


@dataclass
class LossConfig:
    """Supervised pixel objective; pretrained feature losses are staged."""
    kind: str = "l1"
    epsilon: float = 1e-3
    perceptual_weight: float = 0.0
    consistency_weight: float = 0.0


@dataclass
class JPEGStudyConfig:
    """Matched two-view phase study; off preserves the ordinary task pipeline."""
    mode: str = "off"
    augment: bool = True


@dataclass
class OutputConfig:
    """Runs never share an output directory by default."""
    root: str = "experiments/runs"


@dataclass
class Config:
    """Resolved configuration, serializable into resumable checkpoints."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    jpeg_study: JPEGStudyConfig = field(default_factory=JPEGStudyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, independent representation."""
        return asdict(self)


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value: Any, name: str, minimum: float = 0.0, strict: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {minimum}")


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Reject unknown sections/fields and invalid values without coercion."""
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    section_types = {"model": ModelConfig, "dataset": DatasetConfig,
                     "degradation": DegradationConfig, "training": TrainingConfig,
                     "loss": LossConfig, "jpeg_study": JPEGStudyConfig, "output": OutputConfig}
    unknown = set(raw) - set(section_types)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    sections: dict[str, Any] = {}
    for name, cls in section_types.items():
        values = raw.get(name, {})
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError(f"Invalid or unknown {name} fields")
        sections[name] = cls(**values)
    cfg = Config(**sections)
    m, d, g, t, loss = cfg.model, cfg.dataset, cfg.degradation, cfg.training, cfg.loss
    if m.name not in {"dncnn_rgb_l1", "identity", "bicubic", "edsr_baseline", "restormer_denoising", "swinir_light_sr"}:
        raise ValueError(f"Model {m.name!r} is not implemented in this milestone")
    if m.name in {"restormer_denoising", "swinir_light_sr"} and (m.width,m.depth,m.blocks)!=(64,20,16):
        raise ValueError("CNN width/depth/blocks overrides do not configure the fixed approved transformer variants")
    for name, value in [("model.width", m.width), ("model.depth", m.depth), ("model.blocks", m.blocks),
                        ("model.scale", m.scale), ("dataset.patch_size", d.patch_size),
                        ("dataset.train_samples", d.train_samples), ("dataset.val_samples", d.val_samples),
                        ("training.batch_size", t.batch_size), ("training.accumulation_steps", t.accumulation_steps),
                        ("validation_interval", t.validation_interval), ("checkpoint_interval", t.checkpoint_interval)]:
        _integer(value, name)
    if m.depth < 3 or m.scale not in (1, 2, 4):
        raise ValueError("Depth must be >=3 and scale must be 1, 2 or 4")
    if d.kind not in {"procedural", "manifest"}:
        raise ValueError("dataset.kind must be procedural or manifest")
    _integer(d.cache_max_mib, "dataset.cache_max_mib", minimum=0)
    if d.cache_max_mib and t.num_workers != 0:
        raise ValueError("Decoded-source cache requires num_workers=0 to bound total CPU memory")
    for key, value in [("root", d.root), ("output.root", cfg.output.root)]:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a nonempty path string")
    for value in (d.train_manifest, d.val_manifest):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("Manifest paths must be nonempty strings or null")
    profiles = {"awgn_unclipped_v1", "awgn_clipped_v1", "poisson_linear_rgb_v1",
                "jpeg_rgb_444_v1", "sidd_srgb_pairs_v1", "div2k_bicubic_pairs_v1",
                "torch_bicubic_fixture_v1"}
    if g.profile not in profiles:
        raise ValueError("Unknown degradation profile")
    is_sr = g.profile in {"div2k_bicubic_pairs_v1", "torch_bicubic_fixture_v1"}
    if is_sr != (m.scale in (2, 4)) or (is_sr and m.name not in {"bicubic", "edsr_baseline", "swinir_light_sr"}):
        raise ValueError("SR requires a supported SR model with scale 2 or 4")
    if m.name in {"bicubic", "edsr_baseline", "swinir_light_sr"} and not is_sr:
        raise ValueError("SR models require an SR profile")
    if m.name == "restormer_denoising" and g.profile not in {"awgn_unclipped_v1", "awgn_clipped_v1", "poisson_linear_rgb_v1", "sidd_srgb_pairs_v1"}:
        raise ValueError("Restormer is approved for RGB denoising profiles only")
    if d.patch_size % m.scale:
        raise ValueError("Target patch size must be divisible by scale")
    if d.kind == "procedural" and g.profile in {"sidd_srgb_pairs_v1", "div2k_bicubic_pairs_v1"}:
        raise ValueError("Measured/official pairs cannot be generated procedurally")
    for name, value in [("sigma", g.sigma), ("sigma_min", g.sigma_min), ("sigma_max", g.sigma_max)]:
        _number(value, name)
    if g.sigma_min > g.sigma_max:
        raise ValueError("sigma_min exceeds sigma_max")
    _number(g.peak_photons, "peak_photons", strict=True)
    if not isinstance(g.poisson_training_peaks,list) or not g.poisson_training_peaks:
        raise ValueError("poisson_training_peaks must be a nonempty list")
    for peak in g.poisson_training_peaks:
        _number(peak,"poisson_training_peaks entry",strict=True)
    if len(set(g.poisson_training_peaks))!=len(g.poisson_training_peaks):
        raise ValueError("poisson_training_peaks entries must be unique")
    _integer(g.jpeg_quality, "jpeg_quality")
    if g.jpeg_quality > 100:
        raise ValueError("jpeg_quality must be <=100")
    _integer(t.num_workers, "num_workers", 0)
    _integer(t.warmup_steps, "warmup_steps", 0)
    _integer(t.seed, "seed", 0)
    if (t.epochs is None) == (t.max_steps is None):
        raise ValueError("Set exactly one of training.epochs and training.max_steps")
    for name, value in [("max_steps", t.max_steps), ("epochs", t.epochs)]:
        if value is not None:
            _integer(value, name)
    if t.device not in {"auto", "cpu", "mps", "cuda"} or t.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Invalid device or precision")
    if t.max_seconds is not None:
        _number(t.max_seconds, "max_seconds", strict=True)
    for name, value in [("learning_rate", t.learning_rate), ("clip_grad", t.clip_grad), ("epsilon", loss.epsilon)]:
        _number(value, name, strict=True)
    for name, value in [("min_lr", t.min_lr), ("weight_decay", t.weight_decay), ("perceptual_weight", loss.perceptual_weight)]:
        _number(value, name)
    if t.min_lr > t.learning_rate:
        raise ValueError("min_lr exceeds learning_rate")
    if loss.kind not in {"l1", "charbonnier"}:
        raise ValueError("Only L1/Charbonnier primary objectives are implemented")
    if loss.perceptual_weight:
        raise ValueError("Perceptual loss is staged; no implicit pretrained download")
    _number(loss.consistency_weight, "consistency_weight")
    study = cfg.jpeg_study
    if study.mode not in {"off", "random", "stratified"} or type(study.augment) is not bool:
        raise ValueError("JPEG study mode must be off/random/stratified and augment boolean")
    if study.mode != "off" and (g.profile != "jpeg_rgb_444_v1" or m.name not in {"dncnn_rgb_l1", "identity"}):
        raise ValueError("Phase study requires same-size JPEG RGB restoration")
    if study.mode == "off" and loss.consistency_weight:
        raise ValueError("Consistency requires the two-view JPEG study")
    if study.mode != "off" and d.patch_size < 11:
        raise ValueError("Phase evaluation requires patch size >=11 for SSIM")
    return cfg


def load_config(path: str | Path) -> Config:
    """Load YAML using safe parsing and strict validation."""
    with Path(path).open() as handle:
        return config_from_dict(yaml.safe_load(handle))
