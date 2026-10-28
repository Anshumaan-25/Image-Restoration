"""Image domain, degradation statistics, manifest leakage and crop contracts."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from utils.config import load_config, config_from_dict
from utils.dataset_loader import RestorationDataset, read_manifest, validate_splits
from utils.degradation import apply_gaussian_noise, apply_poisson_noise, bicubic_fixture, jpeg_roundtrip
from utils.io import file_sha256, load_rgb, save_rgb


def make_record(root: Path, name: str, split: str, seed: int = 1) -> dict:
    """Create a distinct authored RGB fixture and its exact metadata."""
    array = np.random.default_rng(seed).integers(0, 256, (24, 28, 3), dtype=np.uint8)
    path = root / f"{name}.png"
    Image.fromarray(array).save(path)
    return {"id": name, "dataset": "fixture", "version": "1", "split": split,
            "source_id": name, "target_path": path.name, "input_path": None,
            "sha256": file_sha256(path), "height": 24, "width": 28,
            "bit_depth": 8, "color_space": "sRGB", "scale": 1}


class DataTests(unittest.TestCase):
    """No external photographs or weights are needed for these acceptance gates."""

    def test_awgn_statistics_and_unclipped_boundary(self) -> None:
        x = torch.full((3, 128, 128), .5)
        gen = torch.Generator().manual_seed(33)
        noise = apply_gaussian_noise(x, 25, gen) - x
        self.assertLess(abs(float(noise.mean())), .002)
        self.assertAlmostEqual(float(noise.std()), 25/255, delta=.002)
        noisy = apply_gaussian_noise(torch.zeros_like(x), 50, torch.Generator().manual_seed(1))
        self.assertLess(float(noisy.min()), 0)
        self.assertGreaterEqual(float(apply_gaussian_noise(torch.zeros_like(x), 50, clipped=True).min()), 0)

    def test_poisson_and_fixture_shapes(self) -> None:
        x = torch.rand(3, 24, 28)
        noisy = apply_poisson_noise(x, 60, torch.Generator().manual_seed(1))
        self.assertEqual(noisy.shape, x.shape)
        self.assertTrue(torch.isfinite(noisy).all())
        self.assertTrue(0 <= noisy.min() <= noisy.max() <= 1)
        self.assertTrue(torch.equal(apply_poisson_noise(torch.zeros_like(x), 60), torch.zeros_like(x)))
        self.assertEqual(bicubic_fixture(x, 4).shape, (3, 6, 7))
        for operation in (lambda: apply_poisson_noise(x, 0), lambda: bicubic_fixture(x[:, :23], 4)):
            with self.assertRaises(ValueError):
                operation()

    def test_jpeg_repeatable_real_roundtrip(self) -> None:
        x = torch.rand(3, 24, 28)
        a, b = jpeg_roundtrip(x), jpeg_roundtrip(x)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(x, a))
        self.assertEqual(a.shape, x.shape)

    def test_io_domain_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_record(root, "rgb", "train")
            x = load_rgb(root / "rgb.png")
            self.assertEqual(x.shape, (3, 24, 28))
            save_rgb(x+2, root / "white.png")
            self.assertTrue(torch.equal(load_rgb(root / "white.png"), torch.ones_like(x)))
            with self.assertRaises(FileExistsError):
                save_rgb(x, root / "white.png")
            for mode in ("L", "RGBA"):
                Image.new(mode, (12, 12)).save(root / "bad.png")
                with self.assertRaises(ValueError):
                    load_rgb(root / "bad.png")
            Image.fromarray(np.zeros((12, 12), dtype=np.uint16)).save(root / "bad16.png")
            with self.assertRaises(ValueError):
                load_rgb(root / "bad16.png")

    def test_manifest_hash_and_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = make_record(root, "a", "train")
            b = make_record(root, "b", "val", 2)
            path = root / "train.jsonl"
            path.write_text(json.dumps(a)+"\n")
            self.assertEqual(len(read_manifest(path, root)), 1)
            validate_splits([a, b])
            for key in ("source_id", "sha256"):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_splits([a, {**b, key: a[key]}])
            path.write_text(json.dumps({**a, "sha256": "invalid"}))
            with self.assertRaises(ValueError):
                read_manifest(path, root)

    def test_validation_repeatable_across_workers(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
        dataset = RestorationDataset(cfg, "val")
        before = torch.get_rng_state().clone()
        a, b = dataset[0], dataset[0]
        self.assertTrue(torch.equal(a["input"], b["input"]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        zero = list(DataLoader(dataset, batch_size=1, num_workers=0))
        multi = list(DataLoader(dataset, batch_size=1, num_workers=1))
        for first, second in zip(zero, multi):
            self.assertTrue(torch.equal(first["input"], second["input"]))

    def test_aligned_measured_sr_crops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = make_record(root, "target", "train")
            lr = load_rgb(root / "target.png")[:, ::2, ::2]
            save_rgb(lr, root / "input.png")
            record.update(scale=2, input_path="input.png", input_sha256=file_sha256(root / "input.png"))
            (root / "train.jsonl").write_text(json.dumps(record))
            cfg = config_from_dict({"model": {"name": "bicubic", "scale": 2},
                "dataset": {"root": str(root), "train_manifest": "train.jsonl", "patch_size": 16},
                "degradation": {"profile": "div2k_bicubic_pairs_v1"}})
            sample = RestorationDataset(cfg, "train", True)[3]
            self.assertTrue(torch.equal(sample["target"][:, ::2, ::2], sample["input"]))
