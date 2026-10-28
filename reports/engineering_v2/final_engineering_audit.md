# Engineering v2 audit — 2026-10-26

Status: **implemented and locally verified; external execution and scientific requirements remain open**. No claim of full scientific completion. Approved v1.0 constraints are unchanged; spending is $0; submitted reports were not edited.

- All 65 core tests passed, zero skips; smoke, syntax, CPU dry-run/checkpoint reload and pip dependency check passed. See [acceptance](acceptance_extension_v2.json).
- Restormer Gaussian/real and supervised lightweight SwinIR x2/x4 ports passed strict actual-weight CPU upstream parity and measured MPS tolerance. Four individually approved official assets and receipts are retained.
- Gaussian sigma15/25/50 centre128 test16 and supplied Set5 x2/x4 evaluations were actually executed and independently replayed with identical aggregate scores. They are pretrained comparisons, not project-trained restoration or new-method claims. [Frozen extension replay](final_pretrained_replay_verification.json).
- The original project-trained JPEG study remains frozen. Its source, selected weights, test scores, uncertainty, full-image predictions and latency/memory evidence were preserved. Its consistency extension does not establish a reliable added benefit.
- The prepared Poisson CUDA bundle matches the final 83-file source identity. Its actual ZIP extraction, new-directory bootstrap, plan backup and long-run authorization guard passed offline integration with mocked CUDA/profile/training; this is explicitly not a GPU profile.
- A separate `Poisson_approved_v2.ipynb` notebook is connected to a free T4. Google Drive mounting failed before uploads or profile (`ValueError: mount failed`); a normal-browser authorization/mount is required. Neither the actual profile nor natural-image training has started. The already approved run remains conditional on all measured gates.
- Genuine SIDD public-validation acquisition remains externally blocked by the authors’ Google Drive quota. Importer fixtures and real checkpoint parity are verified; actual SIDD data/quality scores are not.
- A self-contained pretrained inference kit was integrity checked. Standalone extracted CPU inference passed with an actual official checkpoint and RGB 288x288 output; see pretrained_release_inference_verification.json. The completed project-trained JPEG kit remains separate v1 evidence.

Major implementation areas: task-specific transformer ports/factory/configs, official-weight integrity/import and acquisition registry, paired SIDD MAT ingestion, explicit SR replay metrics, mixed Poisson training-level sampling, bounded raw-pretrained evaluation and expanded tests. Exact file hashes and evidence are in [machine-readable audit](final_engineering_audit.json); usage is in [task runbook](../../docs/task_runbook_v2.md).

Remaining work requires the actual cloud run/data, not more placeholder code. Publishable novelty cannot be guaranteed by more software checks or pretrained model comparisons. Any new architectural research direction requires a separately approved protocol with unused test data.
