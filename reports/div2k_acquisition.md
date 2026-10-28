# DIV2K acquisition and preparation — complete

Approved 2026-10-21 (Asia/Kolkata). This report records the acquisition-stage checks; the subsequent approved screening is reported separately in [current results](results.md). Both official HR archives were downloaded and retained locally. All **900 references** passed RGB8/sRGB decoding, file SHA256 and dimension checks. Source IDs and exact file hashes are disjoint across the full train/development/test splits. Scene identities are unavailable; these checks do not establish semantic scene independence.

| Manifest view | Train | Development | Protected project test |
|---|---:|---:|---:|
| Full | 720 (0001–0720) | 80 (0721–0800) | 100 (0801–0900) |
| Screening | 32 (0001–0032) | 8 (0721–0728) | 16 (0801–0816) |

Screening manifests are subsets of their corresponding full splits. The original DIV2K validation images are the JPEG project's protected test; this is not the official SR test protocol. Test images were inspected for integrity and format, with no model scoring, tuning or selection.

Archive payload: 3,979,597,606 bytes. Extracted PNG payload: 3,983,267,330 bytes. Combined payload: 7,962,864,936 bytes (about 7.42 GiB, excluding filesystem overhead). No partial downloads remain. Archives/images are ignored by Git; versioned JSONL metadata remain available under [data/manifests](../data/manifests/README.md).

| Archive | SHA256 |
|---|---|
| DIV2K_train_HR.zip | `9d0b9c463f6e35b6c62cc6a930ee2224f670b34c1df841a57670f9acf0f6c335` |
| DIV2K_valid_HR.zip | `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc` |

Both archives match the approved byte counts and exact expected image lists; ZIP CRC checks passed. These are recorded local SHA256 identities; no publisher SHA256 was supplied. Full image/manifests hashes and dimensions are in [the machine-readable acquisition report](div2k_acquisition.json).

The real-data loader check passed: source 0001 produced deterministic CPU float32 input/target tensors of shape (3,128,128), finite in [0,1], with actual JPEG Q20 degradation changing the input. The [loader record](div2k_loader_check.json) contains the assertions. The shared manifest CLI validated screening train=32 and val=8. All 28 unit tests, smoke, syntax and two-update procedural MPS dry-run checks passed; see [verification](verification.json).

Prepared configurations: [screening](../configs/jpeg/div2k_screening.yaml), [full](../configs/jpeg/div2k_full.yaml), and the full-split [default](../configs/default.yaml). Both training configurations select development data exclusively; test evaluation requires an explicit test manifest after selection is frozen.

At the end of acquisition, no real-data training or pretrained-weight download had started. Subsequent bounded profiling and the explicitly approved three-arm screening have now completed; [feasibility](feasibility.md), [results](results.md) and [final audit](final_audit.md) supersede that stage's execution status. CUDA and publishable novelty remain unverified; no pretrained weights were downloaded.
