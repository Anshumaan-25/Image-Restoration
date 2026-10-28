"""Optimizer-boundary resume must reproduce uninterrupted CPU training exactly."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from train import run_training
from utils.checkpoint import load_checkpoint, run_lock
from utils.config import load_config


class CheckpointTests(unittest.TestCase):
    """Compare full state, not only final prediction shape."""

    def test_concurrent_writer_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with run_lock(tmp):
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    with run_lock(tmp):
                        self.fail("A second writer acquired the same run")
            with run_lock(tmp):
                pass

    def test_nonfinite_gradient_fails_before_checkpoint(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
        cfg.training.device = "cpu"
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "invalid_gradient"
            with patch("torch.nn.utils.clip_grad_norm_", return_value=torch.tensor(float("inf"))):
                with self.assertRaisesRegex(FloatingPointError, "Nonfinite gradients"):
                    run_training(cfg, run)
            self.assertFalse((run / "latest.pth").exists())

    def test_exact_resume_and_compatibility(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
        cfg.training.device = "cpu"
        cfg.training.max_steps = 4
        cfg.training.warmup_steps = 1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = run_training(cfg, root / "full")
            partial = run_training(cfg, root / "split", stop_after_updates=2)
            self.assertFalse(partial["completed"])
            resumed = run_training(cfg, resume=root / "split/latest.pth")
            self.assertTrue(full["completed"] and resumed["completed"])
            a, b = load_checkpoint(root / "full/latest.pth"), load_checkpoint(root / "split/latest.pth")
            self.assertEqual(a["cursor"], b["cursor"])
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertTrue(torch.equal(a["rng"]["torch"], b["rng"]["torch"]))
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            for key, value in a["optimizer"]["state"].items():
                for field, item in value.items():
                    self.assertTrue(torch.equal(item, b["optimizer"]["state"][key][field]), field)
            cfg.training.max_steps = 5
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run_training(cfg, resume=root / "split/latest.pth")

    def test_partial_accumulation_and_wall_stop(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
        cfg.training.device = "cpu"
        cfg.training.max_steps, cfg.training.epochs = None, 1
        cfg.dataset.train_samples = 5
        cfg.training.accumulation_steps = 2
        with tempfile.TemporaryDirectory() as temporary:
            result = run_training(cfg, Path(temporary) / "partial_group")
            self.assertTrue(result["completed"])
            self.assertEqual(result["cursor"], 5)
            self.assertEqual(result["step"], 2)
        cfg.training.epochs, cfg.training.max_steps = None, 20
        cfg.training.max_seconds = 1e-8
        with tempfile.TemporaryDirectory() as temporary:
            result = run_training(cfg, Path(temporary) / "bounded")
            self.assertFalse(result["completed"])
            self.assertEqual(result["stop_reason"], "wall_time_limit")
            self.assertTrue((Path(result["run_dir"]) / "latest.pth").is_file())

    def test_run_directory_collision(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                run_training(cfg, temporary)


class SimulatedOverflowScaler:
    """Exercise scaler control flow on CPU; this is not CUDA hardware validation."""
    def __init__(self, always: bool = False) -> None:
        self.calls = 0
        self.value = 64.0
        self.always = always

    def is_enabled(self) -> bool:
        return True

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        pass

    def get_scale(self) -> float:
        return self.value

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.calls and not self.always:
            optimizer.step()

    def update(self) -> None:
        if self.calls == 0 or self.always:
            self.value /= 2
        self.calls += 1

    def state_dict(self) -> dict:
        return {'calls': self.calls, 'value': self.value}

    def load_state_dict(self, state: dict) -> None:
        self.calls, self.value = state['calls'], state['value']


class OverflowControlTests(unittest.TestCase):
    """Skipped AMP attempts must not consume the requested successful-step budget."""
    def test_successful_step_budget_survives_skip(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/smoke.yaml')
        cfg.training.device = 'cpu'
        cfg.training.max_steps = 2
        cfg.training.accumulation_steps = 1
        scaler = SimulatedOverflowScaler()
        with tempfile.TemporaryDirectory() as tmp, patch('torch.amp.GradScaler', return_value=scaler):
            result = run_training(cfg, Path(tmp) / 'skip')
            state = load_checkpoint(Path(result['run_dir']) / 'latest.pth')
        self.assertTrue(result['completed'])
        self.assertEqual((result['step'], result['skipped_updates'], result['cursor']), (2, 1, 6))
        self.assertEqual(state['consecutive_skipped'], 0)
        self.assertEqual(state['scheduler']['last_epoch'], 2)

    def test_repeated_overflow_fails_finitely(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/smoke.yaml')
        cfg.training.device = 'cpu'
        scaler = SimulatedOverflowScaler(always=True)
        with tempfile.TemporaryDirectory() as tmp, patch('torch.amp.GradScaler', return_value=scaler):
            with self.assertRaisesRegex(FloatingPointError, '16 consecutive'):
                run_training(cfg, Path(tmp) / 'always_overflow')
        self.assertEqual(scaler.calls, 16)
