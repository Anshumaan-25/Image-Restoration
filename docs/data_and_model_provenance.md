# Data and model provenance

## Data already acquired

DIV2K official HR training/validation ZIPs were explicitly approved on 2026-10-21. Source HTTPS URLs, exact byte counts, local SHA256, ZIP CRC and manifest/image identities are in [acquisition evidence](../reports/div2k_acquisition.json). No publisher checksum was supplied; local hashes identify the acquired content. Files retain their research-dataset provenance and original owners' rights. No data redistribution license is invented.

For the JPEG study, original train IDs 0001–0720 are training, 0721–0800 development, and original validation 0801–0900 protected project test. Screening uses first 32/8/16 IDs respectively. These are project JPEG splits, not an official SR test. Known source/exact-content leakage is checked; scene identity is unavailable. Manifest records carry RGB8/sRGB conventions and original image paths without altering reference PNGs.

No SIDD, benchmark SR LR or other dataset was acquired under this approval. Supplied SIDD sRGB pairs must retain measured noisy/clean identities and scene metadata; generic stem pairing is not a SIDD parser. MAT blocks/raw images require a separate validated adapter.

## Implemented models

- DnCNN-style RGB residual CNN (20 convolutions, width 64, BN), project L1/Charbonnier training. See [Zhang et al., Beyond a Gaussian Denoiser](https://arxiv.org/abs/1608.03981). No official checkpoint or exact-paper reproduction is claimed.
- EDSR baseline topology, 16 residual blocks/64 features, no BN, scale-specific PixelShuffle, normalized RGB mean (0.4488,0.4371,0.4040). Independently implemented from the architecture, with [official source](https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/edsr.py) as an architectural reference. No upstream weight conversion or pretrained parity claim. Tiny test configurations are explicitly labeled.
- Identity and torch bicubic references have no pretrained parameters. Torch fixture resizing is not claimed MATLAB benchmark-equivalent.

All learned project models initialize from scratch with recorded seeds. No pretrained weights, VGG, LPIPS, latent autoencoder, GAN/discriminator or adversarial component was downloaded or imported for the study. Future pretrained models require exact checkpoint identity, source license, training provenance and individual download approval; architecture name alone is insufficient provenance.

## Reproduction identity

Each run records resolved config, dataset fingerprints and source identities, model specification, operator/codec/environment, optimizer/scheduler/scaler, successful-step/cursor counters and RNG states. Exact CPU interrupted replay is tested. Hardware/release bitwise equivalence is not promised. The repository has no commit/remote yet; run and QA records retain explicit file hashes instead of fabricating a Git identity. The study freezes source files before actual protected-test evaluation.

See [novelty assessment](../reports/novelty_assessment.md) for prior-art distinctions and [runbook](runbook.md) for execution contracts.
