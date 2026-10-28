# Restoration engineering v3 — start here

Verified approved engineering scope; no established novelty/SOTA claim. No training or downloads are necessary for the commands below. Use an existing compatible Python/PyTorch environment (see requirements/locks); CPU works on all platforms. Replace cpu with mps on Apple Silicon or cuda on an allocated NVIDIA runtime. Choose a fresh output path each time.

From this extracted folder:

```bash
python train.py --dry-run --epochs 1 --batch-size 2 --device cpu
python inference.py --input samples/jpeg_q20.png --output outputs/jpeg.png --weights weights/jpeg_B.pth --device cpu
python inference.py --input samples/poisson_peak60.png --output outputs/poisson.png --weights weights/poisson_best.pth --device cpu
python inference.py --input samples/real_noise_crop.png --output outputs/real.png --weights weights/real_denoising.pth --pretrained-id restormer_sidd --weights-sha256 4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387 --device cpu
python inference.py --input samples/bird_x2.png --output outputs/sr_x2.png --weights weights/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --pretrained-id swinir_x2 --weights-sha256 193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7 --device cpu
python inference.py --input samples/bird_x4.png --output outputs/sr_x4.png --weights weights/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth --pretrained-id swinir_x4 --weights-sha256 09fad24e32ae62722e1a055efde9921328f4137981bab0a42a4a3a806306c58e --device cpu
```

Gaussian restoration uses weights/gaussian_color_denoising_blind.pth with --pretrained-id restormer_gaussian and --weights-sha256 4e2314dae77bb907184da6e2ac625b921491d6b33de26a52512eff1c6bb37fbd. Supply an RGB noisy image. JPEG A and C selections are included for study inspection; B is the example command, not a newly reselected checkpoint. Poisson selection is4500/5000; its included results are development-only. Real-noise crop is only a tiny inference fixture, not the full SIDD score panel. Set5 images come from the pinned official SwinIR repository; source provenance and model licenses are included. Full1,280-block SIDD scoring uses the declared project RGB protocol and official supervised pretrained weights.

RELEASE_INVENTORY.json records all payloads. Retain licenses and provenance. Full datasets are intentionally absent; original acquisition and long-run approvals are not permissions to restart training. Source tests/configs are included; evaluation reports retain their original paths/environments as historical provenance. See docs/delivery.md for results and limitations. No submitted report is included or modified. The existing checkpoint configs may refer to the original dataset location; inference does not require that dataset. Historical exact JPEG research replay requires its separately preserved61-file source/codec, not this current84-file source.
