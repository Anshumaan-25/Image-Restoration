"""Fail-fast offline pipeline acceptance test. No missing required modules are skipped."""
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models import Bicubic, DnCNNRGB
from losses import CompositeLoss
from train import run_training
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.degradation import apply_gaussian_noise
from utils.runtime import select_device


def main() -> None:
    """Exercise actual forward/backward, spatial contracts and saved state."""
    torch.set_num_threads(2)
    model = DnCNNRGB(8, 3)
    clean = torch.rand(2, 3, 24, 24)
    noisy = torch.stack([apply_gaussian_noise(x, 25) for x in clean])
    prediction = model(noisy)
    assert prediction.shape == clean.shape
    loss, logs = CompositeLoss()(prediction, clean)
    loss.backward()
    assert torch.isfinite(loss) and isinstance(logs['pixel'], float)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    with torch.no_grad():
        model.eval()
        assert model(torch.rand(1, 3, 7, 9)).shape == (1, 3, 7, 9)
        assert Bicubic(4)(torch.rand(1, 3, 7, 9)).shape == (1, 3, 28, 36)
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/smoke.yaml')
    cfg.training.device = 'cpu'
    with tempfile.TemporaryDirectory() as temporary:
        summary = run_training(cfg, Path(temporary) / 'smoke')
        checkpoint = load_checkpoint(Path(summary['run_dir']) / 'latest.pth')
        assert summary['completed'] and checkpoint['step'] == 2
    print(f'SMOKE PASSED: actual model backward, SR shapes, two CPU updates, validation/checkpoint; auto device={select_device()}')


if __name__ == '__main__':
    main()
