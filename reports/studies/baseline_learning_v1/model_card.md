# Longer-baseline checkpoint card

Development-only JPEG restoration baseline, trained from scratch; no novel-method, official-benchmark, state-of-the-art or production claim. [Complete measured report](../../baseline_diagnostic.md), [metadata](summary.json), [verification](postprocess.json).

`dncnn_rgb_l1`, 20 convolutions / width 64 / BN, 669,379 parameters. Finite floating RGB BCHW -> same-size unrestricted prediction; no training clamp. File input: 8-bit sRGB RGB PNG/JPEG; unsupported color/alpha/bit depth fails. PNG output clamps [0,1]. No GANs, pretrained components, ensemble or task detection.

32 DIV2K training sources 0001–0032; eight selection-development sources 0721–0728. Seed 42, Q20 RGB 4:4:4, stratified aligned two views, L1/AdamW, 5,000 MPS FP32 updates, target 128, clean microbatch 2 / accumulation 4. Selected step **4,800**, based only on fixed 16-phase eight-image development worst-phase PSNR. Runtime **89.12 minutes**, zero skipped updates, $0.

Local selected weights: `experiments/studies/baseline_learning_v1/B_seed42/best_model.pth`; SHA256 `db32a6373fad66a5d2e86bc6b2cc00ffc8e9935bad3aeb8b8e42ff553c0bdcb9`. Latest weights represent step 5,000 and differ. Full run state/RNG/config/source/data identities remain local and ignored by Git. Git commit is null; explicit hashes identify the unchanged pipeline.

Primary dev: worst-phase PSNR **30.185873 dB** vs input 29.230958; SSIM 0.857976 vs 0.839884. Full-image dev: **32.139600 dB** vs 31.281650. Supplementary 72 unused-for-selection dev sources: worst-phase 29.914286 vs 29.032815. These are project development protocols; no protected-test or multiple-seed evidence. The 1,000-update C result is not a matched comparison to these weights.

```bash
.venv/bin/python inference.py --input reports/studies/baseline_learning_v1/0721_input.png --output outputs/NEW_B5000.png --weights experiments/studies/baseline_learning_v1/B_seed42/best_model.pth --device mps
```

CPU supported; actual CUDA unverified. Full-image memory must fit; no silent tiling or fallback. [Exact exported-pixel parity](trained_cli_verification.json) passed. Unknown-quality/4:2:0/double JPEG, raw, SR, SIDD, AWGN/Poisson and general document/detail recovery are not validated uses of these weights. Outputs can alter correct detail. Checkpoints/datasets are excluded from source bundles and must be preserved separately.
