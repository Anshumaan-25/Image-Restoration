# Broader restoration tasks — engineering v2

Approved architecture v1.0 remains authoritative. This extension implements its previously staged Restormer denoising and lightweight supervised SwinIR x2/x4 roles. Historical JPEG source/checkpoint/result evidence is preserved; it is not rewritten to match the new source identity. Submitted reports are untouched.

## Current evidence

| Track | Implemented | Actual measured result | Remaining |
|---|---|---|---|
| Project-trained JPEG A/B/C | Complete | Frozen all-phase test and full-image Q20, visuals, T4 memory/latency; see JPEG handover | Novelty/consistency benefit not established |
| Pretrained Gaussian RGB denoising | Official Restormer strict-load, native torch adapter and source parity | Fixed centre128, test16, sigma15/25/50: RGB PSNR 34.608662 / 32.163032 / 29.007674; SSIM 0.928203 / 0.887790 / 0.811473 | These are pretrained patch results, not a project-trained denoising method or full-image benchmark |
| Pretrained SR x2/x4 | Official lightweight SwinIR, exact supplied Set5 LR/HR, tiny/odd pad/unpad | Set5 Y-channel replay: x2 38.139390 / 0.961121; x4 32.441986 / 0.897645 (PSNR/SSIM) | No claimed new SR method or project-trained transformer |
| Real SIDD sRGB | Actual official checkpoint strictly loaded; uint8 MAT-v5/HDF5 paired importer tested | All1,280 public-validation blocks: PSNR40.015571 / SSIM0.922464 | Project RGB float protocol, official supervised pretrained model; no hidden-benchmark claim. Fixed40 result retained as history. |
| Approximate linear-RGB Poisson | Correct mixed training peaks 30/60/120; fixed evaluation peak, deterministic metadata, explicit CLI override | Completed5,000 updates, selected4,500; dev8 peak30/60/120 PSNR28.290368/30.564402/32.701839; zero skips | Development evidence; no protected Poisson test or sensor-simulation claim. |
| EDSR x2/x4 | Learned models/train/resume/export tested | Procedural engineering checks | No project-trained natural-image SR checkpoint claimed |

Detailed machine-readable evidence is in `reports/engineering_v2/`. All four official pretrained assets and local acquisition receipts live in `experiments/pretrained/approved_v2/`. None was downloaded implicitly during model construction or inference. Official releases supplied no publisher SHA256; receipt hashes are local integrity records, not a claim of publisher-signed authentication.

## Inference with an official checkpoint

Preserve the existing `inference.py --input/--output/--model/--weights` interface. Project-trained checkpoints continue to determine their own task/model/scale. Official raw assets require an explicit whitelist identifier and retained acquisition SHA256:

```bash
.venv/bin/python inference.py --input data/raw/Set5_official_swinir/LR_bicubic/X2/birdx2.png --output outputs/my_sr_x2.png --weights experiments/pretrained/approved_v2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --pretrained-id swinir_x2 --weights-sha256 193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7 --device mps
```

Choose a new output filename. `swinir_x4`, `restormer_gaussian` and `restormer_sidd` have separate whitelisted files/configs and receipt hashes. No universal automatic task detector is provided. `scripts/evaluate_pretrained.py` records pretrained origin, unknown original training seed, declared training-data overlap limitations, operator, source/manifest/weight hashes, exact counts and per-image metrics. `--protocol swinir_sr_replay_v1` additionally uses the pinned official outer flip padding, uint8 boundary, BT.601 Y and scale border. Ordinary model inference uses the general reflection/replication padding contract; do not pool scores across these protocols.

## True SIDD validation ingestion

The authors' public validation archive was specifically approved (461,780,640 bytes). The initial public download was quota blocked, but the normal authenticated official Drive download succeeded. SHA256:1139c32896697e960159ae845070751e58894ddf607ebfe8f6d83ef012e2fbff. All1,280 uint8 noisy/clean pairs are imported. The original failure diagnostic remains historical. Do not substitute synthetic noise, SIDD training crops or random weights. The existing successful import must not be overwritten. For a fresh checkout supplied with the exact approved archive:

```bash
.venv/bin/python reports/engineering_v2/import_sidd_validation.py --archive /absolute/path/to/test.zip
```

Only the two uniquely named official validation MAT files are extracted after archive size, path safety and CRC checks. `scripts/prepare_sidd_blocks.py` can also import explicitly supplied MAT pairs with an explicit layout. Only uint8 sRGB is accepted without guessed scaling. Parent/source and block indexes, original MAT hashes and actual PNG pair hashes are retained. Validation remains validation; there is no hidden SIDD benchmark submission. Full SIDD inference remains a separately bounded run after real data is available.

## Completed cloud execution — 27 October 2026

The Poisson cache-optimized T4 profile passed0.3592s/update,0.632GiB allocated and zero skips. The original interrupted run retained4,100/5,000 updates. Its separately approved900-update recovery completed5,000 in a new folder, retained full optimizer/scheduler/scaler/RNG state and saved the final snapshot. Selected update4,500 is imported at `experiments/studies/poisson_completed_v3_import_20261027/run/best_model.pth`. The84-file training source identity matches the current core. Python/torch/CUDA/device/dependency/codec gates matched; the Linux kernel changed and no cross-runtime bitwise CUDA guarantee is made.

Poisson CPU development replay covers all8 fixed references at photon levels30/60/120: PSNR28.290368/30.564402/32.701839, SSIM0.820696/0.876466/0.912439. No protected Poisson scoring occurred. The final exportSHA256 is `f944a3474664439d73ba50ee10e749d106a5dea52e3597389d67b1b28eeeccb6`.

SIDD’s original profile projected1254.04s and correctly rejected full scoring. The separately approved optimized helper uses the same CPU float64 valid Gaussian11 SSIM formula with separable convolution; frozen training/core code was not changed. Offline and20 actual cloud-pair reference checks passed (cloud maximum SSIM error1.55e-14). Its profile projected360.30s, allocated708,752,896bytes, and full scoring completed all1,280 blocks in407.99s. Results: PSNR40.015571/SSIM0.922464; input23.662382/0.333469. This is public validation, not hidden benchmark evaluation.

Verified local SIDD import: `experiments/studies/sidd_full_v3_import_20261027/`; exportSHA256 `b5fb7aa44a12b469574d9c50e234baffcb5e30fa0f9ea1f209a8d056ef801717`. All identities/pair hashes match the independent local MAT import; per-image aggregation/counts/finite values and exact source/weight hashes pass audit. No run is pending. Do not rerun old cloud cells: prior failures remain historical, and completed-run guards prevent duplicate work. [Delivery instructions](delivery.md) and [final audit](../reports/engineering_v2/final_engineering_audit_20261027.md) replace the earlier execution handoff.

## Verification and source history

`reports/engineering_v2/acceptance_cache_v3.json`:68 tests plus smoke, syntax, two-update CPU dry-run/checkpoint reload and pip check passed. Actual official-weight CPU/reference outputs matched; measured CPU/MPS differences were <=3.28e-6 on small verification tensors. These checks do not substitute for actual CUDA transformer training profiles.

Original 61 JPEG pilot core sources are copied to `experiments/studies/jpeg_pilot_frozen_source_v1/`; the verified v1 inference/source package and original Colab bundle remain unchanged. Protected historical test replay must use that frozen version and codec, not the evolving v2 checkout. New checkpoints have a distinct source identity, and incompatible resume attempts must fail rather than mix versions.
