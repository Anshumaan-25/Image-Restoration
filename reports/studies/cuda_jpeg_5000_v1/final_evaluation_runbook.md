# Frozen CPU phase evaluation handoff

Status: prepared, not authorized or executed. No further training or GPU allocation is required. The Mac has Pillow12.2/JPEG9.0; the study and independent Colab CPU replay have Pillow11.3/JPEG6.2. Retain the latter codec for matched degradation.

## Inputs

Upload `dist/colab_frozen_test16_v1.zip` and `reports/studies/cuda_jpeg_5000_v1/colab_final_evaluation.py` into private Drive `MyDrive/RestorationProject`. Keep the existing original `colab_restoration_v1.zip` and all three final snapshots. The new ZIP is 67.20 MiB and contains only acquired protected reference PNGs, their existing manifest, a frozen selection record and inventory; no scores or weights. Its SHA256 is `3c570cda7806d96535264373652fa85634a4d450a118937130cc44729e982519`.

## Approval scope

See final_test_proposal.json. Fixed selected checkpoints A4700, B4400, C4900; seed42. Test identities0801–0816, all64 phases. Primary Q20; mismatch Q10/Q40 separately. Nine CPU FP32 jobs, 600 seconds each, at most90 minutes of subprocess allowance plus setup and final saving, $0, no model training, new dataset/model download or automatic retry. All-stage model selection/tuning remains frozen. Ordinary full-image test and latency/qualitative outputs are separate remaining work.

Approval must be explicit before using `--approved`. Default execution raises before accessing cloud files. The launcher refuses to reuse an existing `final_test_v1` destination; interrupted scoring requires inspection and a reviewed recovery rather than blind rerun. Completed per-job reports, logs, environment and execution state are saved to Drive immediately.

## Execution after approval

Mount Drive in a CPU Colab session. Verify the launcher hash using the adjacent local .sha256 record before invoking. Run `python /content/drive/MyDrive/RestorationProject/colab_final_evaluation.py --approved` using the active notebook interpreter with streamed merged stdout/stderr. No pip install or reprofile is needed. The wrapper restores missing original code, verifies source/bundle/codec/checkpoints, extracts only matching test payload, executes fixed core eval.py commands and creates paired image-bootstrap comparisons by quality.

On any timeout, source/codec/hash mismatch or other exception, stop and preserve all artifacts. Never weaken the evaluator's frozen-study safeguards. The final results remain a small one-seed project JPEG study, not an official DIV2K SR benchmark or proof of novelty.
