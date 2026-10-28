# Final engineering audit — 27 October 2026

**Approved engineering delivery and bounded experiments are complete. Publishable novelty remains unestablished.** No whole-scientific-project100% claim is made. No approved run remains pending; no GPU is needed to use the CPU-capable delivered system.

## Functionality and actual results

- JPEG: preserved matched A/B/C5,000-update pilot, fixed dev selections, held-out16/all64 phases/Q10,Q20,Q40, ordinary full-image Q20, visuals and T4 memory/latency. Single-seed consistency benefit is not established.
- Poisson:5,000 successful updates, selected4,500, zero AMP skips; original full-state4,100 checkpoint recovered under matched explicit environment gates. New900-update training took354.76s; original whole-run duration was not retained. Final nested ZIP/checkpoint/source84 identities verified. CPU dev8 replay at peaks30/60/120 gives28.290368/30.564402/32.701839dB and0.820696/0.876466/0.912439 SSIM. Actual selected-model inference exported exactly the direct quantized prediction. No protected Poisson test was scored.
- Gaussian: actual official supervised Restormer sigma15/25/50 centre128 test16 comparisons remain verified and separate from project training.
- SR: official lightweight supervised SwinIR x2/x4 strictly loads and reproduces the declared supplied-pair Set5 Y protocol; learned EDSR train/checkpoint/export integration is implemented, without a natural-image trained EDSR result claim.
- SIDD: all1,280 actual public-validation blocks complete, unique identities and pair hashes identical to independent local MAT import. Independently reaggregated PSNR40.0155710590 / SSIM0.9224636314; input23.6623819637 /0.3334690053. All scores finite. This is project RGB float/border0/no quantization/valid Gaussian11 SSIM, not a hidden benchmark.

## Architecture, correctness and integration

Approved D01–D10 are unchanged. No GANs, implicit downloads, silent OOM migration/tiling, training-output clamp, dataset deletion or paid resource was introduced. Current84 core source files still match the frozen Poisson training and acceptance identity. The original61-file JPEG source and historical packages/results remain intact.

[Final acceptance](acceptance_final_delivery_20261027.json): **68 tests passed, zero skips**. Unit tests, actual model backward smoke, syntax compilation, two-update CPU train/validation/checkpoint reload and pip dependency checks all passed. Tests cover tiny/odd shapes, strict configs, finite gradients/ranges, paired manifests/leakage, device conflicts, full-state exact CPU resume, successful AMP accounting and bounded cache parity/ownership/eviction. Actual official-weight strict load, upstream CPU parity and MPS tolerance checks remain retained.

The initial SIDD profile correctly stopped before full scoring:0.9797s/block projected1,254.04s above the960s gate. Its artifacts are retained. The separately approved optimized attempt used an equivalent CPU float64 separable Gaussian implementation outside the unchanged core. Two offline metric parity/invalid-input tests and three complete/partial/timeout loop tests passed. Actual20-pair cloud comparison matched PSNR exactly and SSIM within1.55e-14. The new profile projected360.30s, used0.660GiB allocated, and full scoring completed in407.99s within the1,200s cap, with progress every32 blocks. No threshold was relaxed.

## Reproducible delivery

[Portable package](</Users/anshu/Documents/Minor Project/Image_Degradatation_Restoration/dist/restoration_engineering_v3.zip>): **232,472,964 bytes**,150 inventoried payloads,84 core source files, eight actual checkpoint assets, licenses/receipts, small demonstration images, configs/tests and fixed result evidence.

SHA256: `4551ed10207c99f45c4284c017a0097a46d4206aca561534292260ce9f73c3d1`.

[Standalone verification](engineering_release_v3_standalone_verification.json) extracted the verified ZIP into a separate directory, rechecked every size/hash, ran the offline two-update dry-run/checkpoint reload, and executed actual CPU inference for JPEG A/B/C, Poisson, Restormer Gaussian/real and SwinIR x2/x4. All eight outputs are RGB and have the required dimensions. These fixture runs verify functionality, not additional quality benchmarks. [Delivery guide](../../docs/delivery.md) supplies exact commands. Inference requires no GPU or network; actual research environments/locks are retained separately.

[Poisson import audit](poisson_completed_import_audit_20261027.json), [SIDD full audit](sidd_full_results_audit_20261027.json), [machine-readable final audit](final_engineering_audit_20261027.json).

## Remaining scientific and operational limits

Novelty is unresolved; the candidate does not establish added consistency benefit. JPEG uses one training seed and small subsets; Poisson evidence is development-only; Gaussian/SIDD/SR comparators are author-pretrained. No convergence, general sensor simulation, full natural-image EDSR training, universal restoration or state-of-the-art claim is made. Multiple seeds, broader untouched holdouts or a stronger new research method require a separate study and approvals.

Git remains unborn/uncommitted with no remote, so reproducibility relies on recorded source hashes rather than a fabricated commit. Cross-platform bitwise replay is not promised. Colab later showed a concurrent notebook-save conflict/disconnection; all approved execution helpers and final Drive exports were already retained/downloaded/verified, and no remote overwrite was attempted. The saved local execution scripts are sufficient for review. Submitted reports are unchanged. Spending:$0.
