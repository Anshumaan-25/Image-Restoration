# Image Degradation Restoration

This GitHub repository contains the source, configs, tests, data manifests, licenses and recorded audit evidence. Dataset images, model checkpoints, local delivery archives and submitted student reports are excluded; inference with trained models requires the separately retained weights. The offline procedural dry run needs no dataset or pretrained download. Historical experiment metadata predates this repository’s first commit and records explicit source hashes instead of Git commit IDs.

Fidelity-first supervised restoration. Architecture decisions D01–D10 are approved; see [approved design](docs/architecture/approved_design.md), [coding handoff](docs/architecture/implementation_handoff.md) and [implementation status](docs/architecture/implementation_status.md). The [SDLC plan](docs/architecture/sdlc_plan.md), [execution runbook](docs/runbook.md) and [release audit](reports/final_audit.md) track delivery and evidence.

## Current executable milestone

Implemented: DnCNN-style RGB L1/Charbonnier baseline, learned EDSR x2/x4, identity/bicubic references, deterministic procedural/manifest data, explicit RGB I/O, AWGN/linear-RGB Poisson/JPEG operators, aligned supplied LR/HR pairs, CPU float64 PSNR/SSIM, modern CUDA AMP support, accumulation, resumable checkpoints and training/evaluation/inference CLIs. No implicit dataset or weight downloads.

The matched Colab JPEG pilot is complete and audited: A/B/C each trained for 5,000 updates on train32, with fixed dev8 checkpoint selection (one seed). Frozen protected test16 evaluation covered Q20/Q10/Q40 across all 64 block phases, followed by ordinary full-image Q20 scoring, fixed visual examples and CUDA FP32 latency/memory checks. These are project-specific pilot results, not a full-dataset benchmark or established novelty claim.

Full-image Q20 PSNR/SSIM: A **31.500075 / 0.869641**, B **31.548332 / 0.870386**, C **31.541077 / 0.870243**. Compressed-input mean PSNR is **30.647211 dB**. Peak full-image CUDA allocation/reservation: **1.534 / 2.922 GiB**. Median 512x512 model-only inference: **199–209 ms** on Tesla T4, batch one, FP32. Small timing differences between identical architectures are not algorithmic speedups.

The primary phase-patch Q20 B−A difference is **+0.035741 dB**, with an image-bootstrap interval above zero conditional on the single training seed. C−B intervals include zero at all tested qualities: an added consistency benefit and publishable novelty remain **unestablished**. No checkpoint was reselected or tuned using test outcomes. [Phase audit](reports/studies/cuda_jpeg_5000_v1/final_results_audit.json), [completion audit](reports/studies/cuda_jpeg_5000_v1/completion_results_audit.json), [handover and demo](docs/jpeg_pilot_handover.md).

Earlier MPS screening and the separate longer B diagnostic are retained as historical experiments, with different environment/operator settings. At that JPEG release, the broader task comparisons were still staged; the v2 section below records their current extension status. Further long runs and large downloads require specific approval. Submitted documents are unchanged.

## Broader task extension v2 — 2026-10-26

Approved Restormer RGB denoising and lightweight supervised SwinIR x2/x4 are now implemented, strictly compatible with pinned official state dictionaries, with dynamic tiny/odd-size handling and no extra runtime packages. All four individually approved official weights were acquired and validated; no GAN variant or random substitute is used. These remain **pretrained evaluation**, distinct from project-trained JPEG results.

Actual pretrained checks: Restormer on fixed centre128 test16 at AWGN sigma15/25/50 gives RGB PSNR **34.6087 / 32.1630 / 29.0077 dB**. Supplied Set5 bicubic pairs, pinned SwinIR Y-channel replay: **38.1394 dB x2 / 32.4420 dB x4**. Source/reference and CPU/MPS checks passed. The approved SIDD archive is now acquired and 1,280 real noisy/clean validation pairs are imported. On a fixed 40-block subset (block0 from each supplied MAT parent), pretrained Restormer measured RGB PSNR **40.0871 dB** and SSIM **0.931649**. This is public-validation subset evidence under the project floating-RGB protocol, not an official full SIDD/hidden-benchmark score. [Independent SIDD audit](reports/engineering_v2/sidd_subset40_results_audit.json).

Poisson natural-image training is complete:5,000 successful updates, zero AMP skips, selected update4,500 using fixed development peak60. Independent CPU replay on dev8 at photon levels30/60/120 measured PSNR **28.290368 / 30.564402 / 32.701839 dB**. Final snapshot and selected-model CLI output passed independent integrity/inference checks. This is development evidence, not protected Poisson testing or established novelty.

Full **1,280-block SIDD public validation** is also complete: official supervised pretrained Restormer measured RGB PSNR **40.015571 dB**, SSIM **0.922464** (input23.662382 /0.333469). All identities match the independent local paired import and per-image aggregates were rechecked. The original timing profile failed safely before full scoring; a specifically approved, reference-verified separable CPU metric implementation passed the same gate and scored all blocks in407.99s, with0.660GiB allocated. This remains the project floating-RGB protocol, not an official hidden-benchmark reproduction.

**The approved engineering delivery and bounded experiments are complete. Publishable novelty is not established.** [Delivery and runnable examples](docs/delivery.md), [current final audit](reports/engineering_v2/final_engineering_audit_20261027.md), [68-test final acceptance](reports/engineering_v2/acceptance_final_delivery_20261027.json), [Poisson import audit](reports/engineering_v2/poisson_completed_import_audit_20261027.json), [SIDD1280 audit](reports/engineering_v2/sidd_full_results_audit_20261027.json). Further seeds, larger studies and a genuinely new method are research work requiring a new plan; they are not claimed completed. Submitted reports are unchanged. Historical frozen sources/packages remain intact.

## Environment and offline verification

The local `.venv` reuses installed packages from the existing `ai_env` via system-site-packages; no global environment was modified. Tested: Python 3.12.13, torch 2.12.0, installed torchvision 0.27.0, NumPy 1.26.4, Pillow 12.2.0. Exact baseline dependency closure: [Mac lock](requirements/locks/macos-arm64-python312.txt). A clean environment can install requirements.txt explicitly; optional.txt is only for separately approved comparisons. The Mac lock is not a CUDA lock.

```bash
source .venv/bin/activate
python -m pip check
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_test.py
python -m py_compile train.py eval.py inference.py models/*.py losses/*.py utils/*.py
python train.py --dry-run --epochs 1 --batch-size 2
```

The dry run selects configs/smoke.yaml: a tiny 3-layer/8-channel test network, authored procedural images, two optimizer updates, fixed validation, checkpoint/reload and odd-size inference. It is not the standard 20-layer/64-channel research baseline. Run outputs go to unique ignored experiments/runs directories. No expensive training is needed for verification.

An explicit `--device cpu`, `mps` or `cuda` is supported. Auto detection is allowed for dry-run/inference; research training requires an explicit device. MPS is available on the local M5 outside Codex's restricted sandbox; worker shared memory also requires the normal execution environment. Do not bypass failures by silently moving a research run to CPU. CPU/MPS default FP32. CUDA fp16 uses torch.amp and GradScaler; CUDA has not been exercised on this Mac.

## Supplied data and manifests

Both approved official DIV2K HR archives are acquired locally: 900 verified RGB PNG references, full 720/80/100 and screening 32/8/16 source-disjoint splits. See the [acquisition report](reports/div2k_acquisition.md) and [manifest conventions](data/manifests/README.md). Image payloads remain ignored by Git. The default configuration uses the full JPEG project split; configs/jpeg/div2k_screening.yaml uses the smaller profiling/pilot subset. This repurposed JPEG test split is not the official SR test. Other large data/weight downloads retain separate approval.

To verify existing local data (no download or training):

```bash
python scripts/prepare_div2k.py
python scripts/prepare_manifest.py --config configs/jpeg/div2k_screening.yaml --validate-only
```

For other supplied clean images, choose new output manifest paths and set the corresponding config paths:

```bash
python scripts/prepare_manifest.py --data-root . --images data/raw/TRAIN_IMAGES --split train --dataset DATASET --output data/manifests/train.jsonl
python scripts/prepare_manifest.py --data-root . --images data/raw/VAL_IMAGES --split val --dataset DATASET --output data/manifests/val.jsonl
python scripts/prepare_manifest.py --config configs/default.yaml --validate-only
```

Split sources before crops. All manifests are checked together for source/known-scene/content leakage at training startup. Paths are relative to dataset.root. To build supplied pairs, add `--input-images RELATIVE_INPUT_DIR --scale 1|2|4`; filenames must match target stems, optionally ending x2/x4 for LR inputs. The generic builder does not guess SIDD-specific identifiers or parse MAT blocks. Explicit paired input hashes/dimensions are mandatory.

Synthetic profiles accept clean-only sources; measured SIDD/official DIV2K profiles require paired inputs and bypass synthesis. JPEG is encoded before random crop so its grid is retained. JPEG sources already compressed are a distinct double-compression setting. Training images smaller than target patch fail instead of being resized. Default file input is 8-bit RGB PNG/JPEG; untagged RGB is interpreted as sRGB. Grayscale/alpha/unsupported ICC/gamma/high bit depth require explicit external conversion. Export is clamped PNG with overwrite refusal.

## Training and checkpoint use

The acquired DIV2K manifests are ready. Before a separately approved bounded research run, profile the chosen model/workload and copy suitable hardware settings from configs/hardware into a task configuration. These reference overrides are not automatically merged. Start with profiling rather than assuming the full model fits.

```bash
python scripts/profile.py --config configs/smoke.yaml --device cpu --warmup 1 --steps 2 --output NEW_PROFILE.json
python train.py --config TASK_CONFIG.yaml --device cuda --max-steps 1000 --max-seconds 1800
python train.py --config RUN/config.yaml --resume RUN/latest.pth --device cuda
python eval.py --checkpoint RUN/best_model.pth --device cpu --output NEW_REPORT.json
python inference.py --input INPUT.png --output NEW_OUTPUT.png --weights RUN/best_model.pth --device mps
```

These commands create new outputs. `--max-seconds` stops at an optimizer boundary; exit code 2 and completed=false identify incomplete runs. Exact resume preserves the original schedule, source hashes, loss/data and architecture; changing max_steps to simulate interruption is incorrect. Weights-only initialization is not implemented as resume. Best weights are chosen using fixed development PSNR, earliest step on ties; test data are never used for selection.

eval defaults to the checkpoint's fixed validation source. To use a supplied manifest, pass `--data-dir ROOT --manifest RELATIVE.jsonl --split val|test`. Task/scale/model overrides conflicting with checkpoint metadata fail. Ordinary evaluation uses project_rgb_float_v1 with project_ssim_v1: clamp predictions at scoring, no quantization, RGB, scale border for SR/zero otherwise, per-image PSNR average, valid 11x11 Gaussian SSIM, CPU float64. Infinite PSNR is `null` plus an explicit flag in strict JSON. No silent skipped samples. Phase-study evaluation additionally records its fixed target crop, panel, quality, JPEG input reference and edge errors. Test scoring requires a frozen selected-checkpoint/test/source/codec record; --full-image is a separate endpoint. Pinned SwinIR Y-channel benchmark replay is implemented; optional LPIPS remains staged.

[Free GPU notebook](notebooks/free_gpu.ipynb) only launches shared code. No free allocation is guaranteed; use no paid tier. The baseline is single-device, and two GPUs do not combine memory. The full 20-layer model has a measured local MPS/128-pixel workload; see [feasibility](reports/feasibility.md). Profiling excludes validation and labels its checkpoints as profiling-only. Actual Colab CUDA profiles are recorded separately; the Mac itself has no CUDA device. The [runbook](docs/runbook.md) contains exact matched-study, freeze, evaluation and comparison commands.
