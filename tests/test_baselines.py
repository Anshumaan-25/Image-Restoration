"""Actual model backward, fidelity loss and variable input shape tests."""
import unittest
import torch
from losses import CharbonnierLoss, CompositeLoss
from models import Bicubic, DnCNNRGB, Identity


class BaselineTests(unittest.TestCase):
    """Verify models rather than differentiating an unrelated prediction tensor."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    def test_variable_dimensions_and_unclamped_outputs(self) -> None:
        model = DnCNNRGB(8, 3).eval()
        for height, width in [(1, 1), (7, 9), (13, 17), (128, 128)]:
            x = torch.rand(1, 3, height, width)
            with torch.no_grad():
                self.assertEqual(model(x).shape, x.shape)
        for parameter in model.parameters():
            parameter.data.zero_()
        model.body[-1].bias.data.fill_(-2)
        self.assertGreater(float(model(torch.zeros(1, 3, 3, 3)).max().detach()), 1)

    def test_backward_and_short_overfit(self) -> None:
        torch.manual_seed(1)
        model = DnCNNRGB(8, 3)
        criterion = CompositeLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        x = torch.rand(2, 3, 16, 16)
        target = x*.8
        values = []
        for _ in range(20):
            optimizer.zero_grad()
            loss, logs = criterion(model(x), target)
            values.append(loss.item())
            self.assertIsInstance(logs["pixel"], float)
            loss.backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters()))
            optimizer.step()
        self.assertLess(values[-1], values[0])

    def test_standard_baseline_forward_backward(self) -> None:
        model = DnCNNRGB()
        x = torch.rand(2, 3, 13, 17)
        pred = model(x)
        self.assertEqual(pred.shape, x.shape)
        loss, _ = CompositeLoss()(pred, x)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_reference_shapes_and_charbonnier(self) -> None:
        x = torch.rand(2, 3, 7, 9)
        self.assertTrue(torch.equal(Identity()(x), x))
        self.assertEqual(Bicubic(4)(x).shape, (2, 3, 28, 36))
        self.assertEqual(float(CharbonnierLoss()(x, x)), 0)
        with self.assertRaises(ValueError):
            CompositeLoss("mse")
        with self.assertRaises(ValueError):
            CompositeLoss(perceptual_weight=.1)
        with self.assertRaises(ValueError):
            CharbonnierLoss(float("nan"))
        with self.assertRaises(FloatingPointError):
            CompositeLoss("charbonnier")(torch.full((1, 3, 12, 12), 1e30), torch.zeros(1, 3, 12, 12))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available in this execution environment")
    def test_mps_forward_backward(self) -> None:
        torch.manual_seed(5)
        model = DnCNNRGB(8, 3).to("mps")
        x = torch.rand(2, 3, 17, 19).to("mps")
        loss, _ = CompositeLoss()(model(x), x*.9)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()))
