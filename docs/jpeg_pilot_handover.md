# Verified JPEG pilot handover — 2026-10-26

The approved bounded JPEG pilot is complete and audited. This is an engineering deliverable with a single-seed ablation study; it does not establish publishable novelty. SR and denoising data/model paths remain available as implemented, but their research runs and the transformer comparators are staged.

## What was verified

- A/B/C: 5,000 updates each, train32/dev8, seed42; selected development steps 4700/4400/4900. Identical DnCNN RGB architecture, 669,379 parameters. No GAN or pretrained restoration weights.
- Protected project test16: nine jobs at Q20/Q10/Q40, all64 phases, 9,216 predictions. This project split repurposes DIV2K validation references and is not the official super-resolution test.
- Ordinary uncropped Q20: 16 images per arm, 48 predictions; RGB border0, prediction clamp only at metric boundary, no quantization, mean per-image PSNR and valid-window SSIM, CPU float64 metrics.
- Three fixed centre-region visual comparisons (0801/0802/0803), exported full-resolution examples and exact crop/content alignment.
- T4 CUDA FP32 batch1: 10 warmups and 30 measured forwards at 256/512 per arm. Full-image peak allocated/reserved memory 1.534/2.922 GiB. 512 median model-only latency 199–209 ms; excludes loading, transfers, degradation and metrics.
- All 47 core tests, smoke, syntax compilation, two-update CPU dry run/checkpoint reload and dependency check passed in the final acceptance record. Six wrapper tests include indexed/unindexed CUDA API regression; actual corrected CUDA execution completed all six jobs.

## Artifacts

Paths below are relative to the project root:

| Artifact | Location |
|---|---|
| Original selected checkpoints, preserved | `experiments/studies/cuda_jpeg_5000_v1_import_20261025/{A,B,C}_seed42/best_model.pth` |
| Phase evaluation imports | `experiments/studies/cuda_jpeg_5000_v1_final_import_20261026/` |
| Full-image results and visuals | `experiments/studies/cuda_jpeg_5000_v1_completion_import_20261026/` |
| Completion audit | `reports/studies/cuda_jpeg_5000_v1/completion_results_audit.json` |
| Phase audit | `reports/studies/cuda_jpeg_5000_v1/final_results_audit.json` |
| Final engineering checks | `reports/studies/cuda_jpeg_5000_v1/acceptance_final_completion_20261026.json` |
| Compact inference/source/results package | `dist/restoration_jpeg_verified_v1.zip` |

The compact package includes all three frozen checkpoints, executable source, dependency records, three cropped demo inputs, fixed comparison panels and evaluation evidence. It excludes dataset payloads, optimizer-history runs and submitted documents. The source checkpoint files retain their original SHA256 identities. There is no new test-selected default checkpoint: specify A, B or C explicitly.

## Local demonstration

Use the existing `.venv` from the project root. Output paths must be new; supported inputs are 8-bit RGB PNG/JPEG. MPS is suitable for light inference; explicit unavailable devices fail.

```bash
.venv/bin/python inference.py --input experiments/studies/cuda_jpeg_5000_v1_completion_import_20261026/A/0801_compressed_center.png --output outputs/demo_A.png --weights experiments/studies/cuda_jpeg_5000_v1_import_20261025/A_seed42/best_model.pth --device mps
```

Change `A_seed42` to `B_seed42` or `C_seed42` and use a different output filename to compare frozen models. Inference on a centre crop is a demonstration; convolution boundary context differs from cropping a restored full image. Do not claim exact replay of the full-image score from this demo. Exported PNG quantization is also separate from the floating evaluation protocol.

Inference requires no GPU cloud allocation or network. A supplied already-JPEG input may contain a different codec/subsampling or double compression and is outside the controlled study distribution. There is no automatic task detector or universal SR/denoising/JPEG checkpoint.

## Research interpretation and limits

Full-image Q20 means: A 31.500075/0.869641, B 31.548332/0.870386, C 31.541077/0.870243 (PSNR/SSIM). Input 30.647211/0.850583. These point estimates do not alone establish significance.

The primary phase endpoint gives B−A +0.035741 dB at Q20, with the recorded image-bootstrap interval above zero conditional on the observed seed. Every C−B interval includes zero across Q20/Q10/Q40. The additional consistency term has no established benefit; Q40 point estimates can regress. A/B/C have the same inference architecture; timing differences are not architectural speedups.

No test-driven retraining, tuning or checkpoint reselection was performed. One training seed, 32 training sources and 16 held-out sources limit external validity. Multiple seeds, larger panels, pretrained comparators and novelty work require a new scoped research decision and bounded compute authorization. No additional GPU work is required to inspect or demonstrate the completed pilot. Submitted documents remain unchanged.
