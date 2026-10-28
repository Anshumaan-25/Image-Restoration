# Limitations and incomplete work

The compact engineering release, approved 3x1,000-update screening and 5,000-update baseline diagnostic are verified. The scientific project, especially required new-method/publishable novelty, is **not complete**.

- At 1,000 updates gains were tiny and C trailed B. The longer B baseline gains 0.955 dB worst-phase on dev8 and 0.881 dB on supplementary dev72, but matched 5,000-update A/C results do not exist. No successful candidate-method claim is justified.
- One seed, 32 training images and eight checkpoint-selection images cannot establish convergence/seed robustness. The extra 72 development sources were unused for selection, but their check was prepared after interim improvement and is supplementary, not confirmatory protected testing. The 16-phase panel does not replace all-64-phase testing. Different cosine horizons confound an isolated duration-effect claim.
- Protected real-test data are unscored. All-64-phase, Q10/Q40 mismatch and final full-image test endpoints remain behind freeze/approval. Integrity checks are not model scoring.
- Codec/color/quantization/split choices are project-specific. No official SR/SIDD/JPEG literature protocol or state-of-the-art comparison is reproduced.
- Restormer/SwinIR, feature loss/LPIPS and SIDD MAT parsing remain staged; raw/Bayer/diffusion deferred. EDSR x2/x4 is fixture-verified, not natural-image trained/benchmarked. AWGN/Poisson pipelines are verified, not experimentally benchmarked.
- CPU/MPS FP32 exercised; CUDA control flow regression-tested including simulated overflow, but actual T4 precision/memory/throughput unverified. Dual GPU memory is not pooled; free allocation is not guaranteed.
- MPS sampled driver allocation is not peak memory. Synchronized synthetic-input latency excludes I/O/metrics. Research runtime includes training/validation/saves.
- Exact resume is checked in the fixed CPU environment, not promised bitwise across accelerators/versions. Git has no commit/remote; source hashes/dependency records supply identity.

See [results](results.md), [audit](final_audit.md), [status](../docs/architecture/implementation_status.md), [research checkpoint](../docs/architecture/research_checkpoint.md). All artifacts remain retained; no missing evidence is replaced with invented scores.
