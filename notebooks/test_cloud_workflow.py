"""Offline checks for cloud transfer, planning and interruption recovery."""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "notebooks"))
from cloud_workflow import (inspect_snapshot, prepare_plan, restore_snapshot, seal_profile,
                            run_arm, safe_members, sha256, snapshot_run, verify_bundle)
from utils.checkpoint import save_checkpoint


class CloudWorkflowTests(unittest.TestCase):
    """Exercise failure paths without allocating an accelerator or training."""

    def test_safe_members_rejects_traversal_symlink_and_duplicates(self) -> None:
        for names in (("../escape",), ("/absolute",), ("x\\escape",), ("x", "x")):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    for name in names:
                        archive.writestr(name, b"x")
            with zipfile.ZipFile(buffer) as archive, self.assertRaises(ValueError):
                safe_members(archive)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            entry = zipfile.ZipInfo("link")
            entry.external_attr = 0o120777 << 16
            archive.writestr(entry, "target")
        with zipfile.ZipFile(buffer) as archive, self.assertRaises(ValueError):
            safe_members(archive)

    def test_transfer_integrity_and_protected_payload_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample").write_bytes(b"clean")
            inventory = {"files": {"sample": {"bytes": 5, "sha256": sha256(root / "sample")}},
                         "split_counts": {"train": 32, "val": 8}, "protected_test_included": False}
            (root / "COLAB_BUNDLE.json").write_text(json.dumps(inventory))
            verify_bundle(root)
            (root / "sample").write_bytes(b"dirty")
            with self.assertRaises(ValueError):
                verify_bundle(root)
            (root / "sample").write_bytes(b"clean")
            inventory["protected_test_included"] = True
            (root / "COLAB_BUNDLE.json").write_text(json.dumps(inventory))
            with self.assertRaises(ValueError):
                verify_bundle(root)

    def test_plan_is_matched_cuda_and_requires_explicit_training_gate(self) -> None:
        previous = Path.cwd()
        os.chdir(PROJECT)
        try:
            with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
                destination = Path(tmp) / "cloud_fixture"
                plan_file = prepare_plan(PROJECT, destination)
                plan = json.loads(plan_file.read_text())
                self.assertEqual(plan["total_optimizer_updates"], 15000)
                self.assertEqual(plan["max_total_training_seconds"], 10800)
                self.assertFalse(plan["cloud_policy"]["local_mps_baseline_reused"])
                normalized = []
                for run in plan["runs"]:
                    cfg = deepcopy(run["config"])
                    self.assertEqual((cfg["training"]["device"], cfg["training"]["precision"]), ("cuda", "fp16"))
                    self.assertEqual((cfg["training"]["batch_size"], cfg["training"]["accumulation_steps"]), (2, 4))
                    cfg["jpeg_study"].pop("mode")
                    cfg["loss"].pop("consistency_weight")
                    normalized.append(cfg)
                self.assertEqual(normalized[0], normalized[1])
                self.assertEqual(normalized[1], normalized[2])
                with self.assertRaises(FileExistsError):
                    prepare_plan(PROJECT, destination)
                with patch("train.run_training", side_effect=AssertionError("Training must not start")):
                    with self.assertRaises(PermissionError):
                        run_arm(PROJECT, plan_file, "A_seed42", Path(tmp) / "backups")
                with patch("torch.cuda.is_available", return_value=False), self.assertRaises(RuntimeError):
                    run_arm(PROJECT, plan_file, "A_seed42", Path(tmp) / "backups", approved=True)
        finally:
            os.chdir(previous)

    def test_snapshot_trims_future_logs_and_restores_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            state = {"model": {}, "config": {}, "step": 100, "microsteps": 400,
                     "cursor": 800, "best_step": 100, "compatibility": "same", "datasets": {},
                     "rng": {}, "environment": {}, "source": {}}
            for name in ("latest.pth", "best_model.pth"):
                save_checkpoint(run / name, state)
            (run / "metrics.csv").write_text("step,microsteps,loss\n100,400,0.1\n101,404,0.09\n")
            saved = snapshot_run(run, root / "drive")
            self.assertIsNotNone(saved)
            self.assertEqual(inspect_snapshot(saved)["step"], 100)
            self.assertEqual(snapshot_run(run, root / "drive"), saved)
            restored = restore_snapshot(saved, root / "restored")
            self.assertEqual(sha256(restored), sha256(run / "latest.pth"))
            with (root / "restored/metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            with self.assertRaises(FileExistsError):
                restore_snapshot(saved, root / "restored")
            # An advancing best without a matching latest must not form a backup.
            state["step"] = 200
            save_checkpoint(run / "best_model.pth", state)
            self.assertIsNone(snapshot_run(run, root / "drive"))

    def test_simulated_preflight_enforces_memory_identity_and_environment(self) -> None:
        from types import SimpleNamespace
        previous = Path.cwd()
        os.chdir(PROJECT)
        try:
            with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
                destination = Path(tmp) / "cloud_fixture"
                plan_file = prepare_plan(PROJECT, destination)
                profile_file = destination / "profile.json"
                profile = {"device": "cuda", "precision": "fp16", "batch_size": 2,
                           "accumulation_steps": 4, "patch_size": 128, "measured_updates": 50,
                           "completed": True, "views_per_clean_crop": 2, "validation_executed": False,
                           "peak_cuda_allocated_bytes": 1024**3, "peak_cuda_reserved_bytes": 2 * 1024**3,
                           "mean_update_seconds": 0.5}
                with patch("torch.cuda.is_available", return_value=True), \
                     patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=15 * 1024**3)), \
                     patch("utils.runtime.environment", return_value={"torch": "fixture"}), \
                     patch("torch.cuda.empty_cache"), \
                     patch("train.run_training", return_value={"completed": True}) as trainer:
                    for field, value in (("completed", False), ("peak_cuda_allocated_bytes", 13 * 1024**3),
                                         ("peak_cuda_reserved_bytes", 15 * 1024**3), ("mean_update_seconds", 2.0)):
                        profile_file.write_text(json.dumps({**profile, field: value}))
                        with self.assertRaises(ValueError):
                            seal_profile(PROJECT, plan_file, profile_file)
                    profile_file.write_text(json.dumps(profile))
                    reviewed = seal_profile(PROJECT, plan_file, profile_file)
                    with patch("utils.runtime.environment", return_value={"torch": "changed"}), self.assertRaises(ValueError):
                        run_arm(PROJECT, reviewed, "A_seed42", Path(tmp) / "drive", approved=True)
                    trainer.assert_not_called()
                    run_arm(PROJECT, reviewed, "A_seed42", Path(tmp) / "drive", approved=True)
                    self.assertEqual(trainer.call_args.args[0].training.max_steps, 5000)
                    trainer.reset_mock()
                    profile_file.write_text('{}')
                    with self.assertRaises(ValueError):
                        run_arm(PROJECT, reviewed, "A_seed42", Path(tmp) / "drive", approved=True)
                    trainer.assert_not_called()
        finally:
            os.chdir(previous)

    def test_snapshot_corruption_is_rejected_before_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "bad.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("latest.pth", b"bad")
                archive.writestr("snapshot.json", json.dumps({"schema_version": 1,
                    "files": {"latest.pth": hashlib.sha256(b"good").hexdigest()}}))
            with self.assertRaises(ValueError):
                restore_snapshot(archive_path, root / "restored")
            self.assertFalse((root / "restored").exists())

    def test_real_cpu_checkpoint_survives_snapshot_resume_exactly(self) -> None:
        import torch
        from train import run_training
        from utils.checkpoint import load_checkpoint
        from utils.config import load_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(PROJECT / "configs/smoke.yaml")
            cfg.training.device, cfg.training.max_steps, cfg.training.epochs = "cpu", 4, None
            cfg.training.validation_interval = cfg.training.checkpoint_interval = 2
            full = run_training(cfg, root / "full")
            run_training(cfg, root / "split", stop_after_updates=2)
            backup = snapshot_run(root / "split", root / "drive")
            restored = restore_snapshot(backup, root / "restored")
            result = run_training(cfg, root / "restored", resume=restored)
            a, b = load_checkpoint(root / "full/latest.pth"), load_checkpoint(restored)
            self.assertTrue(full["completed"] and result["completed"])
            self.assertEqual((a["step"], a["cursor"], a["microsteps"], a["best_step"]),
                             (b["step"], b["cursor"], b["microsteps"], b["best_step"]))
            for name in a["model"]:
                self.assertTrue(torch.equal(a["model"][name], b["model"][name]), name)
            self.assertEqual((root / "full/metrics.csv").read_bytes(), (root / "restored/metrics.csv").read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
