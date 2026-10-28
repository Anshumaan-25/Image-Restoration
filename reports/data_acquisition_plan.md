# DIV2K acquisition plan — completed 2026-10-21

The user explicitly approved both HR archives and source-disjoint split preparation on 2026-10-21. Acquisition and preparation are complete. See the [completion report](div2k_acquisition.md), [archive/manifest hashes](div2k_acquisition.json), and data_acquisition_plan.json. HTTP HEAD requests to the official host returned status 200 with these lengths:

| Archive | Exact bytes | Decimal GB | Destination |
|---|---:|---:|---|
| [Training HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) | 3,530,603,713 | 3.531 | data/raw/DIV2K_train_HR.zip |
| [Validation HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) | 448,993,893 | 0.449 | data/raw/DIV2K_valid_HR.zip |
| Total | 3,979,597,606 | 3.980 | Both ignored archives |

Approved action: download from the official HTTPS host, record actual byte counts and SHA256, extract into new dataset directories without overwriting existing files, and prepare immutable manifests. The retained archives and extracted PNGs contain 7,962,864,936 bytes (about 7.42 GiB, excluding filesystem overhead); the original 8–12 GB planning allowance was sufficient. No purchased compute or pretrained weights are included.

Primary JPEG research split: IDs 0001–0720 train, 0721–0800 development, 0801–0900 protected project test. Initial screening subsets are first 32/8/16 IDs within those splits. This is a repurposed JPEG research split, not the official SR test. No LR archive is needed for JPEG. Later SR and SIDD acquisition remain separate.

Downloads do not authorize long training. Manifest validation passed. Next measure the actual 128-pixel/64-channel baseline workload before proposing a bounded real-data pilot. The candidate still needs a nearest-prior-art distinction before novelty claims.
