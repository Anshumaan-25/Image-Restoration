# Matched CUDA study with independent CPU development replay

Latest evidence date: 2026-10-23. All A/B/C arms completed 5,000 successful CUDA FP16 updates, seed 42. C resumed from update 3,700 and completed in a separately approved invocation. No further training is required for these arms.

The operator reports independent CPU checkpoint replay on the same eight development sources and 16-phase Q20 panel. All selected aggregate scores reproduce the training logs to displayed precision. The complete 43.61 MiB export is now preserved locally. Its outer SHA256, all inner snapshot checksums, selected checkpoint hashes, 384 per-image phase observations, source/config matching and bootstrap recomputation passed the local import audit on 2026-10-25. Protected test data remain unscored.

| Arm | Selected step | Worst-phase PSNR dB | Mean PSNR dB | SSIM |
|---|---:|---:|---:|---:|
| A | 4700 | 30.096330 | 30.186341 | 0.857434 |
| B | 4400 | 30.162796 | 30.257334 | 0.857106 |
| C | 4900 | 30.165986 | 30.261093 | 0.856455 |

| Paired difference | Delta dB | Image-bootstrap 95% interval dB |
|---|---:|---|
| B minus A | +0.066466 | [-0.000767, 0.137848] |
| C minus B | +0.003190 | [-0.020560, 0.024813] |
| C minus A | +0.069656 | [-0.000529, 0.137097] |

Every interval crosses zero. These are descriptive image-bootstrap intervals on the selection-development subset and one training seed, not independent confirmatory evidence or a characterization of seed variability. Added consistency has no demonstrated reliable advantage; SSIM is lower for C than B. Publishable novelty remains unestablished.

Evidence: [full import audit](studies/cuda_jpeg_5000_v1/import_audit.json) and [original operator output](proposals/colab_matched5000_v1/independent_cpu_dev_operator_report.json). All three selections and the existing 16-image held-out manifest are [frozen](studies/cuda_jpeg_5000_v1/frozen_study.json), without scoring. A [bounded CPU phase-test proposal](studies/cuda_jpeg_5000_v1/final_test_proposal.json) is prepared but not authorized or executed. The Mac codec differs from the cloud study, so final scoring targets the matching Colab CPU codec. Retain earlier local results as separate experiments. Both submission Word documents now include completed C and the independent development results; final protected-test, ordinary full-image, qualitative export and latency evidence remain pending.
