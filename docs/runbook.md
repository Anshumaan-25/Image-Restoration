# Restoration execution runbook

Authority: [approved design](architecture/approved_design.md), [SDLC plan](architecture/sdlc_plan.md), [current status](architecture/implementation_status.md). Run from the repository root with the recorded compatible environment. No weights are downloaded on import or execution.

The approved screening and longer B diagnostic are complete. [Screening](../reports/results.md), [longer-baseline results](../reports/baseline_diagnostic.md) and [selected longer-baseline card](../reports/studies/baseline_learning_v1/model_card.md) identify artifacts. Commands below are interfaces, not authorization to repeat long runs or score protected data. [Next research controls](architecture/research_checkpoint.md) retain separate approval.

Current execution preference: **Colab Free for remaining long training**, following the [cloud runbook](cloud_runbook.md) and [notebook](../notebooks/free_gpu.ipynb). The local MPS commands below remain reference interfaces. CUDA/FP16 needs a fresh matched A/B/C study, actual-device profile and a reviewed bounded run; do not reuse the Mac B as its control. No additional local research training is started.

## Verify and inspect data

```bash
source .venv/bin/activate
python scripts/verify.py --device mps --data-check --output reports/NEW_acceptance.json
python scripts/prepare_manifest.py --config configs/jpeg/div2k_screening.yaml --validate-only
```

Use --device cpu when MPS is unavailable. All output report/run names must be new. Verification uses procedural/authored fixtures and real-data metadata checks; its scores are not research results. DIV2K paths/hashes and 720/80/100 plus 32/8/16 splits are in [the acquisition report](../reports/div2k_acquisition.md). The original validation images are the protected JPEG project test, not development data or the official SR test.

## Measure feasibility

```bash
python scripts/profile.py --config configs/jpeg/phase_consistency.yaml --device mps --precision fp32 --batch-size 2 --accumulation-steps 4 --warmup 10 --steps 50 --max-seconds 180 --output reports/NEW_mps_profile.json
python scripts/benchmark.py --config configs/jpeg/phase_consistency.yaml --device mps --sizes 256 512 --warmup 10 --iterations 30 --max-seconds 120 --output reports/NEW_latency.json
```

Profile includes real loading/degradation, two views, model backward and optimizer updates, with no validation or model selection. It writes a profiling-only checkpoint that cannot resume a research run or be evaluated as a selected model. Incomplete profiles retain a report and exit 2; warmup may exhaust the cap before measured updates. Training time caps are soft at optimizer/validation boundaries and exclude initial setup. CUDA uses the same profile command with --device cuda --precision fp16 and a measured common microbatch; aim for <=12 GiB peak allocated. MPS driver allocation is not peak CUDA VRAM. Free GPU allocation is not guaranteed.

## Plan and execute the matched study

```bash
python scripts/run_ablation.py --study configs/jpeg/study.yaml --device mps --precision fp32 --batch-size 2 --accumulation-steps 4 --output experiments/studies/NEW_plan
```

This writes three exact arm/seed configurations without training. A=random phase/pixel loss; B=stratified/pixel; C=stratified/pixel+0.05 agreement. Same clean crops, augmentation, model initialization/seed, BN microbatch, optimizer, update horizon and two-view count. Development selection uses mean per-image worst-phase PSNR on the fixed 16-phase panel. Every phase report also includes JPEG input-quality references and edge diagnostics.

Long-run execution needs the separately bounded approval retained by AGENTS.md. The concrete command after that approval adds --execute and uses a new output directory:

```bash
python scripts/run_ablation.py --study configs/jpeg/study.yaml --device mps --precision fp32 --batch-size 2 --accumulation-steps 4 --output experiments/studies/NEW_screening --execute
```

Default screening: 3 runs x 1,000 successful updates, seed 42, 1,800 seconds per run. FP16 overflow retries consume new deterministic sample ordinals, not successful updates; 16 consecutive skips fail. Finite epoch runs report incomplete if skips prevent their successful-update target. No paid tier or automatic cloud upgrade. The runner stops after an incomplete run; it does not silently shorten or replace it.

Resume the same horizon with the saved config and latest checkpoint:

```bash
python train.py --config RUN/config.yaml --resume RUN/latest.pth --device mps --max-seconds 1800
```

The run lock prevents concurrent writers and releases automatically on process exit. Do not delete checkpoints/logs or change source/config/data to bypass compatibility failures. After finishing an interrupted arm, use the original plan to execute only its remaining configurations via train.py, keeping all horizons/settings unchanged. A completed screening is exploratory: it does not prove convergence or novelty. Final study seeds 42/43/44 and 5,000 updates per arm require a new validated study budget and explicit approval.

## Freeze, score and compare

First evaluate development results and finalize the selected configuration. Then freeze one best checkpoint per planned arm/seed before scoring test data:

```bash
python scripts/freeze_study.py --plan STUDY/plan.json --checkpoints STUDY/A_seed42/best_model.pth STUDY/B_seed42/best_model.pth STUDY/C_seed42/best_model.pth --test-manifest data/manifests/div2k_jpeg_v1/screening/test.jsonl --data-root . --output reports/NEW_frozen.json
python eval.py --checkpoint STUDY/A_seed42/best_model.pth --data-dir . --manifest data/manifests/div2k_jpeg_v1/screening/test.jsonl --split test --phase-panel all --frozen-study reports/NEW_frozen.json --device mps --output reports/NEW_A_test.json
python eval.py --checkpoint STUDY/C_seed42/best_model.pth --data-dir . --manifest data/manifests/div2k_jpeg_v1/screening/test.jsonl --split test --phase-panel all --frozen-study reports/NEW_frozen.json --device mps --output reports/NEW_C_test.json
python scripts/compare_study.py --baseline reports/NEW_A_test.json --candidate reports/NEW_C_test.json --output reports/NEW_C_minus_A.json
```

Repeat B-versus-A and C-versus-B and the predeclared Q10/Q40 mismatch conditions with --jpeg-quality. All 64 phases are scored on one fixed center target per image; this is a patch robustness endpoint. For ordinary full-image JPEG quality use --full-image instead of --phase-panel; keep it in a separate table. Test evaluation verifies selected checkpoint hashes, test manifest, codec and frozen source identity. No tuning after test inspection. The software enforces a freeze record, but procedural safeguards still depend on an honest researcher not creating a new freeze to tune on test data.

Comparison requires matching seeds, training settings/horizon, images, quality, phase panel, metric domain, codec, model and freeze. It averages paired seed deltas within each source image and bootstraps whole images. Never treat 64 correlated phases as 64 independent images. Infinite PSNR is encoded as null plus an explicit flag; paired differences with nonfinite endpoints fail rather than silently dropping images.

## Other tasks and inference

[SR configs](../configs/sr/edsr_x2.yaml) use supplied official aligned LR/HR pairs, target patch 128, separate x2/x4 EDSR heads, 16 residual blocks/64 features. Generic pair manifests match explicit stems; they do not regenerate official LR. Torch bicubic fixtures are labeled non-benchmark. [Denoising configs](../configs/denoising/awgn.yaml) separate unclipped AWGN, approximate linear-RGB Poisson and supplied SIDD sRGB pairs. SIDD identifiers/scenes must be explicitly prepared; MAT blocks and a benchmark replay are not implemented. No SR/SIDD images or weights were acquired under the DIV2K HR approval.

```bash
python inference.py --input INPUT.png --output NEW_OUTPUT.png --weights RUN/best_model.pth --device mps
```

Checkpoint metadata fixes model/task/scale. No training clamp; metric/export boundaries clamp. Grayscale, alpha, unsupported color profiles and high bit depth fail explicitly. Inference is full-image; there is no silent tiling or OOM migration. Large external images must be checked against available memory.

## Resource and research limits

[Novelty assessment](../reports/novelty_assessment.md) documents known overlap and unproven distinctions. There is no publishable-method claim. Restormer/SwinIR, optional feature losses/LPIPS, raw/Bayer and diffusion are not implied by CNN support. Use the [thin free-GPU notebook](../notebooks/free_gpu.ipynb) for shared-code CUDA verification; keep environments/results separate and export artifacts before runtime termination.
