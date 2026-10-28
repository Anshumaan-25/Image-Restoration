# Screening checkpoint card

Exploratory **development-only** checkpoints, trained from scratch, $0. No novel-method, benchmark, convergence or production-quality claim. [Results](../../results.md), [integrity](integrity.json), [summary](summary.json), [limitations](../../limitations.md).

## Contract and intended use

Reproduce the approved Q20 RGB JPEG screening and exercise shared inference. `dncnn_rgb_l1`: 20 convolutions, width 64, BatchNorm, 669,379 parameters, input minus predicted residual. Forward: finite floating RGB BCHW -> unrestricted same-size RGB BCHW. File input: 8-bit sRGB RGB PNG/JPEG; unsupported color/alpha/bit depth fails. Export clamps to [0,1]. No adversarial/pretrained components or ensemble.

These weights target the named Q20 RGB 4:4:4 setting. SR, SIDD, AWGN, Poisson, unknown-quality/4:2:0/double JPEG and raw/Bayer are not validated uses. Correct detail can be altered and individual images can worsen.

## Training, selection and identity

32 DIV2K train sources 0001–0032, eight dev 0721–0728; project test unscored. Each arm: seed 42, 1,000 MPS FP32 successful updates, 128-pixel targets, clean batch 2, accumulation 4, two views, L1/AdamW. C adds 0.05 agreement. Best selection uses fixed dev image-worst-phase PSNR every 100 updates: A=1000, B/C=900.

| Arm | Selected local checkpoint | SHA256 |
|---|---|---|
| A | `experiments/studies/mps_screening_v1/A_seed42/best_model.pth` | `3e113b9220e08d1a06b583b076898a9bd5127f39833cc6ecc9fd99fb0bc38d2a` |
| B | `experiments/studies/mps_screening_v1/B_seed42/best_model.pth` | `66eda9081d377434050664c9c3da6a328781c7634275923d7ebee3a23a3280f7` |
| C | `experiments/studies/mps_screening_v1/C_seed42/best_model.pth` | `cbaabf9470e8f3a7275b310f6a7826fe12c872c60d6f7c6fa5f6974b03bea9c4` |

Weights remain local, ignored by Git/source bundles. Run directories retain optimizer/scheduler/RNG/config/source/data identity. Git is unborn; commit null plus explicit source hashes identify this implementation. Load only trusted project artifacts, not arbitrary third-party pickle checkpoints.

## Quality and local inference

Eight selected-dev images/16 phases: input worst-phase PSNR 29.230958 dB; A/B/C 29.239696/29.256013/29.241428. Full-image input 31.281650; A/B/C 31.286015/31.294960/31.293056. C trails B by 0.014584 dB on the selected endpoint. No protected-test/multiple-seed evidence.

From repository root, recorded environment, **new output path**:

```bash
.venv/bin/python inference.py --input reports/studies/mps_screening_v1/0721_input.png --output outputs/NEW_B.png --weights experiments/studies/mps_screening_v1/B_seed42/best_model.pth --device mps
```

CPU supported via `--device cpu`; actual CUDA unverified. Full-image inference has no silent tiling/OOM migration. [CLI verification](trained_cli_verification.json) confirms byte-identical export on the supplied crop. [Feasibility](../../feasibility.md) records speed/memory and limitations.
