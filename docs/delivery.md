# Verified engineering delivery — 27 October 2026

The approved executable restoration system and bounded experiments are delivered. This is an engineering completion statement, not a claim of publishable novelty, state-of-the-art performance or full-dataset training. Submitted reports were not changed.

## Delivered tracks

| Track | Runnable model | Evidence and boundary |
|---|---|---|
| JPEG artifacts | Project-trained DnCNN A/B/C, 5,000 updates each | Frozen one-seed train32/dev8/test16 pilot; all64 phases at Q10/Q20/Q40; ordinary full-image Q20, qualitative examples and T4 latency/memory verified. Extra consistency benefit is unestablished. |
| Approximate RGB Poisson | Project-trained DnCNN, 5,000 updates | Mixed photon levels30/60/120; selected update4500 using fixed peak60 dev8. Separate CPU development replay at30/60/120: PSNR28.290368/30.564402/32.701839. No protected Poisson test or new sensor-noise method claimed. |
| Synthetic Gaussian RGB | Official supervised pretrained Restormer | Fixed centre128 test16 at sigma15/25/50: PSNR34.608662/32.163032/29.007674. Unclipped sigma/255 noise; not project-trained. |
| Real RGB noise | Official supervised pretrained Restormer | All1,280 SIDD public-validation blocks: PSNR40.015571, SSIM0.922464. RGB float, border0, no quantization, valid Gaussian SSIM. Public validation is not a hidden benchmark. |
| Super-resolution x2/x4 | Official supervised pretrained lightweight SwinIR | Supplied aligned Set5 bicubic pairs: Y-channel PSNR38.139390/32.441986; separate pinned uint8 benchmark replay. |
| SR training baseline | Learned EDSR x2/x4 | Shape/backward/train-checkpoint-inference integration verified; no trained natural-image EDSR quality claim. |

Task, scale and checkpoint must be selected explicitly. There is no universal degradation detector or transformer cascade. No GAN, diffusion, paid service or implicit weight download is used.

## Start in the existing checkout

```bash
cd '/Users/anshu/Documents/Minor Project/Image_Degradatation_Restoration'
source .venv/bin/activate
python train.py --dry-run --epochs 1 --batch-size 2 --device cpu
```

This is a two-update procedural integration check, not research training. For a supplied image, use a new output path:

```bash
python inference.py --input outputs/examples_poisson_v3/poisson_peak60_input.png --output outputs/my_poisson.png --weights experiments/studies/poisson_completed_v3_import_20261027/run/best_model.pth --device mps
python inference.py --input data/raw/Set5_official_swinir/LR_bicubic/X2/birdx2.png --output outputs/my_sr.png --weights experiments/pretrained/approved_v2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --pretrained-id swinir_x2 --weights-sha256 193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7 --device mps
```

CPU, MPS and CUDA are supported; an explicit unavailable device fails. CPU/MPS inference uses FP32 by default. Full transformer inference on large images may exceed memory; it never silently tiles or migrates devices. CLI output is clamped and quantized to RGB PNG; training predictions are unrestricted.

## Portable package

`dist/restoration_engineering_v3.zip` contains current source/tests/configs, four official pretrained assets with licenses/receipts, selected project Poisson weights, all three frozen JPEG selections, small demonstration images and audit evidence. After extraction, use its `START_HERE.md`; package weight/sample paths are simplified. `RELEASE_INVENTORY.json` records every payload byte size and SHA256. No full dataset is bundled. Historical JPEG and pretrained-v2 packages remain intact.

An existing compatible environment is required. Local validated pair: Python3.12.13, torch2.12.0, torchvision0.27.0. Cloud results record Python3.13.16/torch2.11.0+cu130/T4. Platform locks and actual environments are retained; do not install a Mac lock into Colab. Optional SIDD import/fast scoring requires SciPy, already present in the measured environments. No installation occurs on import/inference.

## Verification and evidence

- `reports/engineering_v2/acceptance_final_delivery_20261027.json`:68 tests, zero skips; smoke, syntax, two-update CPU dry-run/checkpoint reload and dependency checks passed.
- `reports/engineering_v2/poisson_completed_import_audit_20261027.json`: nested snapshot inventory, full training state,5000/4500 update identities,84-file source match and all three development report contracts.
- `reports/engineering_v2/poisson_actual_inference_verification.json`: actual selected-model CPU CLI export exactly matches direct quantized prediction.
- `reports/engineering_v2/sidd_full_results_audit_20261027.json`: complete1280 unique identities, all finite scores, independent per-image aggregation, exact manifest equality to local paired import, official weight/source/launcher hashes and bounded profile/scoring.
- `reports/engineering_v2/sidd_fast_metrics_verification.json`: equivalent float64 separable Gaussian SSIM passes boundary/reference tests; cloud20-pair maximum SSIM difference1.55e-14. The frozen core metric source is unchanged.
- `reports/studies/cuda_jpeg_5000_v1/`: preserved frozen phase/full-image/latency evidence. Exact historical replay requires its frozen61-file source and codec.

SIDD’s original profile failed its timing gate before full scoring; its artifacts remain. The specifically approved optimized attempt preserved model/data/protocol and passed the same caps:0.2815s/block profile,0.660GiB allocated,407.99s complete scoring. Poisson recovery used354.76s of additional training; the original session’s total duration was not retained, so it is not reconstructed.

## Scientific limitations and future work

Publishable novelty remains unestablished. JPEG has one training seed and a small held-out subset; image-bootstrap intervals do not measure training-seed variability. The consistency arm does not establish improvement over stratified sampling. Poisson results are development-only and approximate RGB noise, not raw sensor simulation. SIDD is public validation and pretrained; Set5 is a small pretrained SR replay. These limits cannot be removed by relabeling engineering tests as experiments.

Further work, if desired, requires a separately frozen research question, new approval for long runs/downloads, independent held-out evaluation and multiple seeds. Current execution cells must not be rerun: they reject completed training/full scoring. See the final audit for exact delivery and unresolved scientific scope.
