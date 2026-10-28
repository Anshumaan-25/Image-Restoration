"""End-to-end CLIs including inference/export and strict conflict handling."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import torch
from train import run_training
from utils.config import load_config
from utils.io import load_rgb, save_rgb


class CLITests(unittest.TestCase):
    """Use only newly created temporary fixtures, no research datasets."""

    def test_direct_profiler_and_manifest_entry_points(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "profile.json"
            result = subprocess.run([sys.executable, "scripts/profile.py", "--config", "configs/smoke.yaml",
                "--device", "cpu", "--warmup", "0", "--steps", "1", "--output", str(report)],
                cwd=repository, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(report.read_text())["measured_updates"], 1)
            (root / "images").mkdir()
            save_rgb(torch.rand(3, 16, 18), root / "images/image.png")
            manifest = root / "val.jsonl"
            result = subprocess.run([sys.executable, "scripts/prepare_manifest.py", "--data-root", str(root),
                "--images", "images", "--split", "val", "--output", str(manifest)],
                cwd=repository, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(manifest.read_text())["id"], "image")

    def test_eval_inference_and_mismatch(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        cfg = load_config(repository / "configs/smoke.yaml")
        cfg.training.device = "cpu"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_training(cfg, root / "run")
            weights = root / "run/best_model.pth"
            def execute(script: str, *args: str) -> subprocess.CompletedProcess:
                return subprocess.run([sys.executable, str(repository / script), *map(str, args)],
                                      cwd=repository, capture_output=True, text=True, timeout=60)
            report = root / "evaluation.json"
            result = execute("eval.py", "--checkpoint", weights, "--device", "cpu", "--output", report)
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads(report.read_text())
            self.assertEqual(metrics["count"], 2)
            self.assertTrue(metrics["procedural_test_only"])
            source, output = root / "source.png", root / "restored.png"
            save_rgb(torch.rand(3, 13, 17), source)
            args = ("--input", source, "--output", output, "--weights", weights, "--device", "cpu")
            result = execute("inference.py", *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(load_rgb(output).shape, (3, 13, 17))
            self.assertNotEqual(execute("inference.py", *args).returncode, 0)
            result = execute("inference.py", "--input", source, "--output", root / "wrong.png",
                             "--weights", weights, "--model", "identity", "--device", "cpu")
            self.assertNotEqual(result.returncode, 0)
