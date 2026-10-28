# DIV2K JPEG project manifests

Prepared from the two explicitly approved official HR PNG archives. All 900 images passed preparation and independent manifest validation. Archive hashes, per-manifest hashes and acquisition checks are in [reports/div2k_acquisition.json](../../reports/div2k_acquisition.json); each JSONL record includes its image SHA256.

| View | Training | Development (val) | Protected project test |
|---|---|---|---|
| div2k_jpeg_v1/full | 0001–0720 (720) | 0721–0800 (80) | 0801–0900 (100) |
| div2k_jpeg_v1/screening | 0001–0032 (32) | 0721–0728 (8) | 0801–0816 (16) |

Screening sets are immutable subsets of their corresponding full splits, not additional independent data. The original DIV2K validation images are repurposed as this JPEG study's held-out project test; this is not the official SR test protocol. Do not use them for training, checkpoint selection or hyperparameter tuning. Integrity/format inspection is separate from scoring or tuning.

Paths are relative to the project data root. Images remain under data/raw/DIV2K_train_HR and data/raw/DIV2K_valid_HR, with source archives retained. No crops are pre-generated. Records include image IDs, source identity, dimensions, range/color/bit-depth conventions, file SHA256 and source archive identity. The preparation helper refuses to overwrite different existing images or manifests.

Recreate/verify local preparation only (no downloads or training):

```bash
python scripts/prepare_div2k.py
python scripts/prepare_manifest.py --config configs/jpeg/div2k_screening.yaml --validate-only
```

The default configuration points to the full project split. configs/jpeg/div2k_screening.yaml is the smaller T4 profiling/pilot configuration; actual research execution is separately bounded and approved. Test evaluation additionally requires an explicit test manifest/split after the experimental setup is frozen.

Source: [official DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/). Cite Agustsson and Timofte, NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study, CVPR Workshops 2017. Original image rights remain with their owners; use the official research dataset terms.
