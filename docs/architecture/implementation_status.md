# Implementation status — approved v1.0 / SDLC release

## Current delivery status — 27 October 2026

Approved engineering delivery and bounded experiments are complete: project JPEG A/B/C pilot,5,000-update Poisson baseline with selected4,500 checkpoint and three-level development replay, official pretrained Gaussian/Set5 x2/x4 comparisons, and all1,280 SIDD public-validation blocks. Current source passes68 tests and required smoke/syntax/CPU dry-run/dependency checks. Actual source/weight/reference/device parity, complete snapshot recovery, export identities/counts and selected-model inference were verified. SIDD metrics are40.015571dB/0.922464 under the project protocol; this is not a hidden benchmark. Publishable novelty remains unestablished. [Delivery](../delivery.md), [current audit](../../reports/engineering_v2/final_engineering_audit_20261027.md).

The sections below are retained historical records of earlier stages, not current blockers or competing instructions.


## Current engineering extension status — 2026-10-26

The approved Restormer denoiser and SwinIR lightweight x2/x4 adapters are now implemented, with official source/license pins, native torch helper replacements, strict tensor/state-dict/reference parity, finite backward and actual MPS checks. Four specifically approved official supervised checkpoints were acquired and strictly validated. Gaussian denoising on16 fixed centre patches at sigma15/25/50 and supplied Set5 x2/x4 Y-channel replay completed; these are author-pretrained comparisons, not project-trained models or novel contributions.

SIDD uint8 sRGB paired MAT-v5/HDF5 import is implemented, and all1,280 true public-validation pairs were acquired/imported. A frozen40-block subset replay measured40.0871235dB /0.93164947 SSIM under the project RGB protocol; independent MAT/PNG pixel alignment and per-image aggregation checks passed. This is not a full SIDD benchmark. A corrected mixed-photon Poisson data path and explicit fixed-level evaluator passed procedural train/checkpoint/CLI integration. Its natural-image5k/90min Colab run is conditionally approved but awaits a passing actual CUDA preflight: the uncached profile failed its time gate, and a bitwise-equivalent bounded CPU cache is being reprofiled. No protected Poisson test was scored.

[68-test full acceptance](../../reports/engineering_v2/acceptance_cache_v3.json), [actual checkpoint parity](../../reports/engineering_v2/actual_remaining_pretrained_verification.json), [result audit](../../reports/engineering_v2/extension_results_audit.json), [task runbook](../task_runbook_v2.md). Historical JPEG artifacts remain unchanged and use their preserved61-file frozen source, not the evolving v2 checkout. Academic novelty remains unestablished. The earlier status sections below are retained as historical records.


Architecture D01–D10: accepted. Engineering release scope: **implemented and verified**. Approved screening and longer-baseline diagnostic: **completed and verified**; full scientific study remains incomplete. Novelty: **not established**. Authority/evidence: [SDLC plan](sdlc_plan.md), [runbook](../runbook.md), [final audit](../../reports/final_audit.md), [core acceptance](../../reports/sdlc_acceptance_poststudy.json), [screening](../../reports/results.md), [longer baseline](../../reports/baseline_diagnostic.md).

## Implemented and checked

| Task | Current state |
|---|---|
| T01 Runtime/config | Strict YAML, CPU/MPS/CUDA policy, current torch.amp interfaces, deterministic keys/RNG, recorded compatible existing environment. Python 3.12.13/torch 2.12.0 used without global package changes. |
| T02 Data/metrics | RGB PNG/JPEG I/O, source/hash split checks, actual JPEG/AWGN/approximate linear-RGB Poisson and supplied LR/HR pairs, float64 PSNR/SSIM. Actual DIV2K 900 references and six manifests validated; 720/80/100 full and 32/8/16 screening. SIDD MAT, reference benchmark replay and learned metrics remain staged. |
| T03 Models/losses | Standard DnCNN RGB, learned EDSR x2/x4 (16 blocks, width 64), identity/bicubic, L1/Charbonnier and differentiable two-view agreement. No output clamp or implicit feature weights. Both standard SR models passed finite backward. |
| T04 Trainer/CLIs | AdamW, successful-update warmup/cosine, accumulation/clip/AMP, atomic resume with complete state and source identities, exclusive writer locks, finite-gradient guards, bounded retry sampling after simulated AMP skips. Baseline and phase CPU exact resume verified. |
| T05 Feasibility | Actual full-model/128-pixel real-DIV2K two-view MPS profile: 10 warmups + 50 measured updates; 0.9894 seconds/update mean. Synchronized 256/512 model-only MPS latency measured. [Budget](../../reports/feasibility.md). CUDA/T4 memory/AMP unverified. |
| T06 JPEG candidate | Exact phase construction, 64-phase deterministic cycles, matched A/B/C configs, fixed 16-phase dev selection, all-phase evaluation, input references/edge diagnostics and plan-only-by-default runner implemented. All three approved 1,000-update screening arms completed. Candidate C trails B on selected development data; [prior-art assessment](../../reports/novelty_assessment.md) and results establish no publishable contribution. |
| T07 Comparators | Learned EDSR support implemented. Restormer/SwinIR, approved pretrained weights and optional VGG/LPIPS remain staged; no empty model promises or fake checkpoints. |
| T08 Evaluation/reporting | Protected-test freeze/source/codec/checkpoint guards, full-image versus phase endpoints, nested infinity JSON and paired image-grouped bootstrap implemented. Authored-fixture integration passed. Screening and 5,000-update B selected checkpoints independently replayed; full-image, supplementary dev72, edge/interior, fixed-patch and exact trained CLI exports verified. Protected real-test evaluation remains pending. |

## Release verification

The [post-study release record](../../reports/sdlc_acceptance_poststudy.json) reports **47 tests passed, zero skips**, 15.939 seconds of unittest execution. All required smoke, syntax, two-update MPS dry-run, pip dependency and screening-manifest CLI checks passed. Source fingerprints were unchanged during acceptance and match all screening arms. [Standard SR verification](../../reports/standard_sr_verification.json) and [SR CLI verification](../../reports/sr_cli_verification.json) provide additional shape/backward/train-checkpoint-export evidence. [Trained JPEG CLI](../../reports/studies/mps_screening_v1/trained_cli_verification.json) produced byte-identical output to direct inference.

Fresh review/regression fixes covered hidden profile validation; missing study/control code and SR; relabeled-image leakage; concurrent checkpoint writers; overflow retries exhausting successful-step budgets; repeated-overflow failure; Q10 metadata mismatch; and nested nonfinite reporting. Actual CUDA has not been exercised; simulated overflow tests validate control flow only.

## Delivery and remaining gates

[Runbook](../runbook.md), [SDLC plan](sdlc_plan.md), exact configs, a shared-code free-GPU notebook, script-based QA and a [final audit](../../reports/final_audit.md) document executable behavior and limitations. The approved implementation handoff remains the architectural contract and links this current status; do not recreate completed modules.

Short bounded profiles preceded the approved screening: all three arms completed in 56.44 minutes, zero skips. At 1,000 updates B gains only 0.025055 dB worst-phase; C trails B by 0.014584 dB. The separately approved unchanged B diagnostic completed **5,000 updates in 89.12 minutes**, zero skips; selected update 4,800 gains **0.954915 dB** worst-phase / **0.857950 dB** full-image PSNR on dev8, plus **0.881471 dB** worst-phase on supplementary dev72 unused for selection. Postprocessing/replay/aggregation/CLI checks passed, and an unmatched-budget comparison was correctly rejected. [Longer-baseline evidence](../../reports/baseline_diagnostic.md). No protected real-test scoring, download, spending, pipeline change or new dataset acquisition occurred. [Next research checkpoint](research_checkpoint.md) defines unapproved additional controls; no run remains active.

Publishable novelty and longer-budget **candidate** benefit remain unresolved. The baseline now shows useful development restoration; it is not a novel method. Transformer/feature comparators, official SR/SIDD replay, SIDD MAT and raw/Bayer remain staged. No state-of-the-art, convergence or full original-proposal completion is claimed.

Git remains unborn/uncommitted on codex/architecture-review with no configured remote. Files were not staged or committed; checkpoints and QA retain null commit plus explicit source hashes. Images/archives, run checkpoints and study outputs remain ignored. A cloud allocation and exact submission timestamp remain unconfirmed.

2026-10-21 continuation preference: Colab Free for remaining long research training. [Cloud preparation](../cloud_runbook.md) provides a small private train32/dev8 bundle, real-device profiling, matched CUDA A/B/C planning and versioned checkpoint snapshots. Local preparation checks do not validate CUDA/FP16 or Drive; those remain pending actual allocation. The old MPS B stays a separate result. No additional local research run or cloud long run is authorized by this documentation.

[Fresh cloud-preparation QA](../../reports/cloud_preparation_verification_v1.json): 47 core tests plus seven offline cloud-helper tests passed; required smoke/syntax/two-update CPU dry-run/dependency checks passed. Exact CPU checkpoint replay survives snapshot transfer, and all 61 core source hashes still match the old trained baseline. Actual CUDA/Drive verification is pending.
