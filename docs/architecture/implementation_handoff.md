# Implementation handoff — approved v1.0

**Approved for implementation**, D01–D10, 2026-10-20. Authority: [approved design](approved_design.md), [approval record](approval_record.md), [ADR](adr/0001-approved-restoration-design.md). AGENTS.md controls engineering/safety. This does not authorize particular data/weight downloads or long runs.

Follow tasks in order. Contracts are fixed by the approved design. Tests use stdlib unittest (pytest-compatible); required failures exit nonzero. Only unavailable hardware/optional packages may be explicitly skipped. Track status separately from design approval and experiment completion. See the current [SDLC plan](sdlc_plan.md), [runbook](../runbook.md) and [audit](../../reports/final_audit.md) before assigning work; T01–T06 core and learned EDSR are implemented and must not be recreated.

Research execution update: the matched 1,000-update screening and separately approved 5,000-update B diagnostic are complete/verified. [Current results](../../reports/baseline_diagnostic.md) and [ordered research continuation](research_checkpoint.md) define preserved artifacts, matching contracts and approval gates. The user now prefers Colab Free: [cloud runbook](../cloud_runbook.md). A new CUDA/FP16 study must rerun A/B/C together after actual-device profiling; the earlier local A/C-only proposal remains historical. Approved v1.0 is unchanged; no additional long runs are authorized by this handoff.

## T01 Runtime/config

Files: requirements/locks; requirements.txt; configs/default.yaml, smoke.yaml; utils/config.py, runtime.py, reproducibility.py; tests/test_config.py, test_runtime.py.

Strict nested YAML, one task/model/scale/operator. Reject unknown keys, invalid scalar values and unsupported combinations. Auto device chooses CUDA/MPS/CPU at startup; explicit unavailable requests fail. CUDA AMP/scaler; CPU/MPS FP32. Deterministic sample keys/RNG, no implicit network. Done: supported isolated environment recorded; validation/runtime tests and environment pip check pass.

## T02 Data/metrics

Files: utils/io.py, dataset_loader.py, degradation.py, metrics.py; scripts/prepare_manifest.py; tests/test_io.py, test_data.py, test_degradation.py, test_metrics.py.

Approved RGB CHW/BCHW/sample/manifest contracts; source IDs/hash/split checks. AWGN sigma/255; declared linear-RGB Poisson; measured-pair bypass; aligned official SR pairs; actual Pillow 4:4:4 JPEG. Fixed validation without changing caller RNG.

Test tiny/odd shapes, EXIF, grayscale/alpha/profile/16-bit rejection, missing/duplicate/leaked pairs, invalid crops, empty datasets and nonfinite metrics. CPU float64 Gaussian valid-window SSIM, correct PSNR/infinity and per-image mean. Done: analytic PSNR, independent SSIM parity <=1e-6, noise statistics/repeatability and split/crop tests pass offline.

## T03 Baseline/losses

Files: models/baselines.py, common.py, __init__.py; losses/l1_loss.py, composite.py; tests/test_baselines.py, test_losses.py.

DnCNN RGB: 20 convolutions, width 64, BN, x-noise; tiny overrides labeled tests. Identity/bicubic references. No forward clamp; exact output shapes. L1/Charbonnier -> scalar + detached float logs. Disabled feature loss needs no weights. Done: real-model finite nonzero backward, variable shapes and short procedural overfit. Unsupported models fail explicitly; EDSR/transformers are later tasks.

## T04 Trainer/resume/CLIs

Files: train.py, eval.py, inference.py; utils/checkpoint.py, logger.py; tests/test_training.py, test_checkpoint.py, test_cli.py; tests/smoke_test.py.

Single-device AdamW; per-successful-update warmup/cosine, accumulation/clip/AMP; unique run outputs. Atomic optimizer-boundary checkpoint with architecture/config/data/operator/protocol/source/environment/RNG/cursor/best state. Resume checks compatibility; changing a run's total schedule is not how interruption is simulated.

Offline dry-run: two updates, validation, checkpoint/reload/inference; preserve --dry-run --epochs 1 --batch-size 2. Test partial accumulation, nonfinite/skipped steps, wall-time stop, path collisions and CLI/checkpoint conflicts. Done: exact 4-step CPU versus 2+save/resume+2 parity including BN/optimizer/scheduler/next sample; all mandatory commands pass. Accelerator checks are separate.

## T05 Profile/free cloud

Files: scripts/profile.py; hardware configs; notebooks/free_gpu.ipynb; reports/feasibility.md.

10 warmup/50 measured updates on actual workload; allocated/reserved CUDA memory or labeled MPS measurements. Profile <=12 GiB allocated target, freeze study microbatch; do not assume two GPU memory pooling. Thin notebook calls shared modules. Done: measured cost projection; actual data and bounded pilot require separate approvals.

## T06 Candidate phase study

Files: two-view data additions; losses/consistency.py; train.py; configs/jpeg/{random_phase,stratified_phase,phase_consistency,study}.yaml; scripts/run_ablation.py; tests/test_jpeg_phase.py, test_consistency.py; reports/novelty_assessment.md.

Use exact shared-target phase construction in approved design; no roll/wraparound or post-JPEG augmentation. input_alt collates; concatenate to 2B for one model forward. Common pixel loss=0.5*(L1a+L1b); candidate adds eta*L1(pa,pb), both branches differentiable. A=random, B=stratified, C=stratified+consistency; match crops/init/seeds/schedule/two-view budgets.

Done: target alignment, 64-phase cycle, both-branch gradients and eta=0 parity tested. Prior-art differentiation precedes novelty claims; screening/final runs remain bounded-data/compute gated. Code checks alone are not scientific results.

## T07 Optional task comparators

Files: models/restormer.py, swinir.py; EDSR baseline; sr/denoising configs; tests/test_models.py, test_reference_parity.py; reports/pretrained_provenance.md.

Approved heads/norm/range/scale matrix; pad/unpad tiny/odd inputs. Minimal licensed ports with pinned commits; strict, individually approved checksum/provenance-verified weights. No LDM/custom Restormer-SR. Done: FP32 reference parity (initial atol/rtol 1e-5; investigate differences), shapes and per-model inference reports. Pretrained evaluation is separate from project training.

## T08 Frozen evaluation/report

Files: eval.py phase analysis; reports/experiments.md, results.md, limitations.md, CSV/JSON; README.md; tests/test_report.py.

Fixed dev selection/protected test; per-image worst-phase patch endpoint separate from full-image quality; seed variation/image-grouped uncertainty. Same crops/error scales; no dropped failures/invented metrics; label tiled approximations. Done: saved-checkpoint rerun with exact counts/hashes, actual latency/memory and honest novelty status. Full SR/SIDD/AWGN/Poisson research coverage is staged, not implied by a JPEG pilot.

## Mandatory verification

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_test.py
python -m py_compile train.py eval.py inference.py models/*.py losses/*.py utils/*.py
python train.py --dry-run --epochs 1 --batch-size 2
```

Use the recorded environment. Keep data/weights/run artifacts ignored; never delete artifacts to make a test pass. Record implemented, tested and experimentally validated status separately in implementation_status.md.
