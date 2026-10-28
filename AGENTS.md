# AGENTS.md — Approved restoration engineering instructions

Architecture: **approved v1.0**, explicit user approval of D01–D10 on 2026-10-20.
Read this file and [approved design](docs/architecture/approved_design.md) before edits/tests. Follow [implementation handoff](docs/architecture/implementation_handoff.md). [Approval record](docs/architecture/approval_record.md) and [ADR](docs/architecture/adr/0001-approved-restoration-design.md) explain the decisions. Earlier proposals/reviews and docs/history are historical, not competing instructions. Major architectural changes still require approval.

## Objective and delivery

Image Degradation Restoration Using Advanced Deep Learning Techniques, B.Tech minor project. Preserve SR x2/x4, RGB real/synthetic denoising (SIDD, AWGN sigma 10–50, approximate linear-RGB Poisson), and JPEG artifact removal. This week's approved priority is an executable shared system and one focused JPEG phase-consistency study; other tasks are staged. Do not represent planned modules or pipeline checks as completed experiments. Novelty is required academically but is not established: review prior art and report negative results honestly.

## Hard constraints

1. No GANs, adversarial discriminators/losses or unapproved adversarially trained components. Verify checkpoint provenance before use. LDM/custom Restormer-SR/raw workflows are deferred.
2. Fidelity first. No invented results or unsupported state-of-the-art claims. No training-output clamp/sigmoid; clamp only at declared metric/export boundaries.
3. Default supervised loss L1; configurable Charbonnier. Perceptual loss is optional, default weight 0. Pure pixel-MSE primary training remains prohibited; MSE for PSNR/features is allowed. D08 approves optional prediction consistency for the named JPEG experiment.
4. Freeze pretrained VGG/LPIPS weights, keep eval mode, preserve prediction-side gradients. No random-weight substitute or implicit downloads.
5. Model/batch images: RGB BCHW, C=3; dataset samples CPU CHW. Clean targets float32 [0,1]. Declared unclipped noise may exceed [0,1]; predictions/internal features are unrestricted finite values. Explicit BGR/RGB conversion with OpenCV.
6. CPU/MPS/CUDA support; startup auto preference CUDA, MPS, CPU. Explicit unavailable requests fail. No silent migration after OOM. Current torch.amp CUDA autocast/GradScaler; CPU/MPS FP32 default. Test precision on the actual device.
7. Strict $0 budget. No paid resources, large dataset/pretrained-weight downloads, long/expensive runs, or deletion of checkpoints/datasets/logs without specific approval. Offline procedural verification is authorized.

## Approved models and interfaces

- First learned baseline: dncnn_rgb_l1 (20 convolutions, width 64, BN, residual x-noise). Tiny depth/width overrides are test configurations.
- SR CNN: EDSR baseline (16 residual blocks, 64 channels, scale-specific pixel shuffle).
- Restormer denoising and task-specific lightweight supervised SwinIR x2/x4; optional RGB SwinIR JPEG. Exclude GAN-trained real-world SR checkpoints.
- NAFNet is an optional named substitution, not an automatic extra model; it is a CNN.
- Models take only tensors in forward. I/O/degradation outside models. Dynamic pad/unpad for architecture constraints, including tiny/odd inputs. Losses return scalar plus detached float logs. No computation graphs in logging histories.

## Data, metrics and reproducibility

Use immutable JSONL manifests with IDs, source/scene identity, split, metadata and SHA256. Split before crops; reject identity/hash leakage. Pair by explicit identifiers, never sorted zip. SIDD pairs bypass synthesis; benchmark SR uses supplied aligned LR/HR. Precisely name noise units/order/clipping, interpolation, antialiasing, codec/phase and seeds. JPEG research uses PNG references and RGB 4:4:4; already-JPEG sources are a separate double-compression setting.

One task/model/scale/operator per run. Fixed validation and deterministic sample keys. Configurable step budgets, batch/accumulation/precision; profile before research training. Accumulation is not larger-microbatch BN or pooled multi-GPU memory.

Record configuration, seed, Git commit/dirty source hashes, dependency/hardware environment, data/operator/protocol hashes and checkpoint provenance. Save model/optimizer/scheduler/scaler, update/cursor/best state, RNG states and resolved config atomically at optimizer boundaries. Test exact fixed-environment CPU resume; do not promise cross-platform bitwise replay.

PSNR/SSIM protocols explicitly state channel/border/range/quantization/aggregation. Perfect PSNR is infinity; mean per-image PSNR, not PSNR of pooled MSE. CPU float64 metrics (not MPS float64), eval/no_grad validation. SSIM may be negative; LPIPS is not a probability. Optional LPIPS requires declared weights/version/normalization. Report missing/failing cases and actual sample counts. Tiled transformer inference is approximate and labeled separately. Keep project results, pretrained evaluation and benchmark reproductions distinct.

## Engineering and verification

Python 3.10+, tested PyTorch 2.x; lock the validated torch/vision pair. Prefer an existing supported environment over unnecessary downloads. Public functions/classes need type hints/docstrings. Focused modular changes on a codex/ branch; no architecture placeholders. Implement runtime/config -> data/metrics -> baseline/trainer/resume -> profile -> candidate ablations -> approved comparators -> frozen evaluation/report. No agent delegation is required. The current [SDLC plan](docs/architecture/sdlc_plan.md), [runbook](docs/runbook.md) and [final audit](reports/final_audit.md) distinguish implemented code, verified checks and pending scientific work.

Before declaring an implemented milestone complete, run and report:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_test.py
python -m py_compile train.py eval.py inference.py models/*.py losses/*.py utils/*.py
python train.py --dry-run --epochs 1 --batch-size 2
```

Use stdlib unittest (pytest-compatible) to avoid unnecessary package downloads. Required failures exit nonzero; never hide missing required modules. Report changed files, executed checks and unresolved risks; no unexecuted tests claimed passed.

Preserve train.py --config/--resume/--dry-run; eval.py --checkpoint/--data-dir/--task/--scale; inference.py --input/--output/--model/--weights. Checkpoint metadata governs model/task/scale; conflicts fail. Offline dry-run: procedural data, two optimizer updates, validation, checkpoint/reload, no pretrained downloads. Research runs require explicit devices and bounded settings.
