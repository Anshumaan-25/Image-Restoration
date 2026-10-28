"""Independent analytic and NumPy reference checks for scientific metrics."""
import json
import math
import unittest
import numpy as np
import torch
from utils.metrics import aggregate, json_metrics, psnr, ssim


def reference_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Independent NumPy valid-window implementation of the specified formula."""
    g = np.exp(-(np.arange(11) - 5)**2 / (2*1.5**2))
    g /= g.sum()
    kernel = np.outer(g, g)
    def mean(x: np.ndarray) -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(x, (11, 11), axis=(-2, -1))
        return np.einsum("cijuv,uv->cij", windows, kernel)
    ma, mb = mean(a), mean(b)
    va, vb, cov = mean(a*a)-ma*ma, mean(b*b)-mb*mb, mean(a*b)-ma*mb
    return float(np.mean(((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))))


class MetricTests(unittest.TestCase):
    """Metrics must be well-defined at boundaries and across image sizes."""

    def test_identity_and_known_error(self) -> None:
        target = torch.zeros(3, 16, 18, dtype=torch.float64)
        self.assertTrue(math.isinf(psnr(target, target)))
        self.assertAlmostEqual(psnr(target + .1, target), 20, places=12)
        self.assertAlmostEqual(ssim(target, target), 1, places=12)
        row = json_metrics({"psnr": float("inf"), "count": 1})
        self.assertIsNone(row["psnr"])
        self.assertTrue(row["psnr_is_infinite"])
        json.dumps(row, allow_nan=False)

    def test_reference_parity(self) -> None:
        rng = np.random.default_rng(32)
        a, b = rng.random((3, 17, 19)), rng.random((3, 17, 19))
        self.assertAlmostEqual(ssim(torch.tensor(a), torch.tensor(b)), reference_ssim(a, b), places=10)

    def test_negative_ssim_not_clipped(self) -> None:
        x = torch.arange(32).repeat(32, 1).remainder(2).double().expand(3, -1, -1)
        self.assertLess(ssim(x, 1-x), 0)

    def test_per_image_aggregation(self) -> None:
        rows = [{"psnr": 10., "ssim": .5}, {"psnr": 30., "ssim": .9}]
        self.assertEqual(aggregate(rows)["psnr"], 20)
        self.assertTrue(math.isinf(aggregate(rows + [{"psnr": float("inf"), "ssim": 1.}])["psnr"]))
        with self.assertRaises(ValueError):
            aggregate([])
        with self.assertRaises(ValueError):
            aggregate([{"psnr": 1., "ssim": float("inf")}])
        with self.assertRaises(ValueError):
            psnr(torch.empty(3,0,2),torch.empty(3,0,2))

    def test_invalid_inputs_and_border(self) -> None:
        x = torch.rand(3, 16, 18)
        pred = x.clone()
        pred[:, 1:-1, 1:-1] = x[:, 1:-1, 1:-1]
        pred[:, 0] = 1
        self.assertTrue(math.isinf(psnr(pred, x, border=1)))
        for a, b in [(x[:1], x), (x*float("nan"), x), (x, x+2)]:
            with self.subTest(), self.assertRaises(ValueError):
                psnr(a, b)
        with self.assertRaises(ValueError):
            ssim(x[:, :10], x[:, :10])
        with self.assertRaises(ValueError):
            psnr(x, x, border=8)
