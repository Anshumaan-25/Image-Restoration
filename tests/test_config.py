"""Configuration and device policy acceptance tests, without optional downloads."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from utils.config import config_from_dict, load_config
from utils.runtime import precision_context, select_device


class ConfigTests(unittest.TestCase):
    """Invalid scientific/runtime settings must fail before training."""

    def test_smoke_round_trip(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
        self.assertEqual(config_from_dict(cfg.to_dict()), cfg)

    def test_reject_invalid_settings(self) -> None:
        cases = [{"unknown": {}}, {"training": {"typo": 1}},
                 {"training": {"batch_size": True}}, {"training": {"accumulation_steps": 0}},
                 {"training": {"epochs": 1}}, {"training": {"max_steps": None}},
                 {"degradation": {"sigma": float("nan")}}, {"loss": {"kind": "mse"}},
                 {"loss": {"perceptual_weight": .05}}, {"model": {"name": "diffusion"}},
                 {"model": {"scale": 4}}, {"degradation": {"jpeg_quality": 101}},
                 {"dataset": {"root": 123}}]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                config_from_dict(deepcopy(raw))

    def test_explicit_device_failure(self) -> None:
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_device("auto"), torch.device("cpu"))
            with self.assertRaises(RuntimeError):
                select_device("mps")
            with self.assertRaises(RuntimeError):
                select_device("cuda")
        with self.assertRaises(ValueError):
            precision_context(torch.device("cpu"), "fp16")
