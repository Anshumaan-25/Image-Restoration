# SDLC delivery plan — approved architecture v1.0

## Current delivery status — 27 October 2026

M0–M6 engineering and the approved bounded research executions are complete and verified. Broader task coverage now includes official Restormer/SwinIR comparisons, complete5,000-update Poisson training with development replay, and all1,280 SIDD public-validation blocks. Final68-test acceptance, actual artifacts and standalone delivery checks are linked from [delivery](../delivery.md) and [current audit](../../reports/engineering_v2/final_engineering_audit_20261027.md). No run is pending. Novelty, training-seed replication and broader scientific generalization remain unestablished; they are not engineering acceptance claims. Earlier tables below document historical execution stages.


## Current execution status — 2026-10-26

Broader approved task adapters and four official supervised pretrained weights are verified, with actual Gaussian, Set5 x2/x4 and a declared SIDD40 public-validation subset evaluated. Current local acceptance is68 tests plus required integration checks. Poisson training remains conditional on the unchanged CUDA profile gates; a bounded decoded-source cache is being profiled after the original pipeline failed the time projection. [Current extension audit](../../reports/engineering_v2/final_engineering_audit.md).

The approved bounded JPEG pilot is now complete: matched A/B/C 5k training, frozen test16/all64/Q10,Q20,Q40, full-image Q20, fixed visuals and T4 latency/memory. [Audit](../../reports/studies/cuda_jpeg_5000_v1/completion_results_audit.json) and [handover](../jpeg_pilot_handover.md) contain current evidence. Novelty, extra training seeds, larger-scale validation and staged task/comparator studies remain open; no additional runs are authorized by this status update. Earlier milestone tables below describe their original execution stage.


Owner: this implementation session. Status: engineering release verified; approved controlled screening completed and analyzed. Authority: AGENTS.md and approved_design.md (D01–D10). User requests autonomous implementation, QA, integration and final audit. This does not authorize paid compute, new large downloads, destructive actions or additional expensive research runs.

## Requirements and completion boundary

Deliver the approved compact restoration system: source-disjoint data, deterministic degradation, learned RGB denoising/JPEG baseline, learned SR x2/x4, resumable training, inference, reproducible evaluation, and the matched JPEG phase experiment. Preserve the no-GAN, $0, unclamped training, explicit device, fixed-dev/protected-test and honest-results requirements. Models and research results must never be inferred from design documents.

The approved week-one research priority remains the JPEG study. Restormer/SwinIR and optional perceptual/LPIPS comparators are staged independently; diffusion remains deferred. Publishing novelty is a research requirement whose satisfaction requires evidence, not an implementation acceptance test. A verified engineering deliverable is not a completed scientific study.

## Audit findings

| ID | Finding and evidence | Severity | Action / acceptance |
|---|---|---|---|
| A01 | profile.py invokes run_training, which always validates full-resolution images at the last update; time/memory measurements can include unrequested validation. | High | Explicit profile mode, no model selection/evaluation; actual workload metadata and incomplete-run reporting. |
| A02 | Approved two-view phase construction, consistency loss, stratified sampling and worst-phase selection are absent. | High | Exact aligned crops, shared model pass, both-branch gradients, deterministic panels and matched controls. |
| A03 | Checkpoints retain manifest fingerprints but not source identities; eval accepts an independently relabeled overlapping manifest. | High | Retain source/hash identities and reject cross-split overlap during evaluation. |
| A04 | SR factory permits only nonlearned bicubic; project has no trainable SR path. | High | Approved EDSR baseline, scale-specific heads, shape/backward/integration checks. |
| A05 | User-required novelty remains unestablished; shift-invariant JPEG and restoration equivariance already exist. | High | Primary-source nearest-prior-art assessment; no claim of originality from a search's failure to find an exact match. |
| A06 | No controlled study runner, grouped comparison report, complete runbook or final audit. | Medium | Validated plan-only runner, bounded execution, protocol/seed checks, image-grouped uncertainty and honest pending-results report. |
| A07 | Tests cover the offline baseline, not candidate resume, phase evaluation, actual 128-pixel workload or cloud CUDA. | High | Add targeted regression/integration tests and measured local checks; explicitly distinguish unallocated CUDA. |

## Dependency-aware milestones

| Milestone | Dependencies | Modules/deliverables | Acceptance criteria | Status |
|---|---|---|---|---|
| M0 Requirements/audit | Existing approval/data | This plan; requirements traceability; prior-art notes | Scope/gates explicit, findings linked to fixes | Verified |
| M1 Core correctness | M0 | config, data, trainer/checkpoint/eval guards; regression tests | Existing suite passes; data identity and failure paths tested; exact resume retained | Verified |
| M2 JPEG candidate | M1 | phase data/loss, training integration, three configs, tests | All 64 phases align to identical targets; repeatable crops; eta=0 and gradient checks; candidate exact CPU resume | Verified |
| M3 Scientific evaluation | M2 | fixed development panel, full phase evaluation, comparison/uncertainty and runner | Fixed dev-only selection; protocol/counts/identity checked; no test tuning; independent endpoint fixtures | Verified |
| M4 Learned SR | M1 | EDSR x2/x4, configs and integration tests | Odd/tiny LR shapes; finite backward; paired scale alignment; train/checkpoint/inference round trip | Verified |
| M5 Feasibility/integration | M2–M4 | bounded profile, local measurements, notebook/runbook | Real 128x128 baseline and two-view workloads measured; no hidden full-image validation; CPU/MPS checks; CUDA explicitly pending if absent | Verified for local release |
| M6 Final QA/documentation | M1–M5 | Full regression/smoke/syntax, CLI end-to-end, audit and status/report docs | Commands recorded; features vs experiments vs unavailable resources separated; no stale completion claims | Verified for local release |
| M7 Research execution | M3/M5 + bounded-run approval/free hardware | Screening, seed runs, frozen protected-test report | Matched arms, saved checkpoints, exact datasets/protocols; measured results only | Screening and longer B diagnostic complete/verified; full study/novelty incomplete; remaining run/test gates retained |

## Verification and contracts

Each significant component receives focused unittest/integration tests before the full regression run. Required full commands:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python tests/smoke_test.py
.venv/bin/python -m py_compile train.py eval.py inference.py models/*.py losses/*.py utils/*.py scripts/*.py
.venv/bin/python train.py --dry-run --epochs 1 --batch-size 2
.venv/bin/python -m pip check
```

All data/model contracts are in approved_design.md. Checkpoints include resolved configuration, data/operator/protocol identity, RNG/cursor, optimizer/scheduler/scaler and source provenance. Public CLIs preserve prior arguments. Invalid combinations fail before starting a run. No network activity in imports/tests.

Anticipate: patch/context undersize; phases outside 0–7; mismatched branches; float/nonfinite inputs; tiny SR images; final partial accumulation; resume after phase sampling; output collision; codec quantization; CPU/MPS precision limitations; full-resolution memory; incomparable protocols; test leakage; correlated phase samples; missing free GPU. Retain all artifacts and report failures, never silently omit them.

## Resource policy

Short bounded local QA/profiling under the user's autonomous-work instruction is permitted. No unbounded research training, free-cloud account manipulation, large weights/datasets or paid resources. Prepare concrete executable configurations and measured cost estimates before requesting expensive-run approval. Existing DIV2K approval remains in force.

## Progress log

- Initial audit: 28 baseline tests previously passed; DIV2K 900 references acquired and validated. No natural-image training/novelty claim.
- Plan created before implementation changes in this session.

- Core implementation: aligned JPEG views/consistency, fixed development panel, protected-test freeze, grouped comparison, learned EDSR x2/x4, source leakage guards and exclusive run locks implemented. Focused tests pass.
- Fresh review found AMP skipped updates could exhaust max-step sampling; changed max-step sampling to a deterministic unbounded ordinal stream, terminating on successful updates with a 16-consecutive-overflow guard. Epoch sampling remains finite and reports incomplete schedules honestly.
- Local diagnostic full-model two-view profile: 10 warmups + 50 measured real-DIV2K updates, approximately 1.01 seconds/update on MPS FP32, batch 2 / accumulation 4. Repeat isolated after final QA changes; not a quality result or CUDA projection.

- Release audit: 47 tests passed without skips; all mandatory verification commands, real screening manifest validation, standard SR backward/CLI checks and actual MPS cost/latency evidence recorded. Scientific study/novelty remain pending.

- User explicitly approved local screening: three matched 1,000-update runs, 30 minutes per arm, $0 and no protected test scoring. Primary worst-phase development metric is logged before freezing the study sources. Prelaunch acceptance rerun precedes execution.

- Screening completed all 3,000 updates in 56.44 minutes, zero skips, identical pipeline source hashes. Selected checkpoint metrics independently replayed/aggregated. B exceeds input by only 0.025055 dB worst-phase / 0.013309 dB full-image; C trails B by 0.014584 dB on selected development data. No candidate success/novelty claim. [Full report](../../reports/results.md).
- Post-study acceptance: 47 tests, zero skips, smoke/syntax/MPS dry-run/dependencies/manifests passed. Trained CLI export exactly matches direct inference. Short fixed-patch and gradient/mode diagnostics found learning but no proven cause for weak restoration. [Audit](../../reports/final_audit.md).
- The separately approved longer B diagnostic completed 5,000 updates in 89.12 minutes, zero skips, selected step 4,800. It gains 0.954915 dB worst-phase / 0.857950 dB full-image on dev8; supplementary dev72 gain 0.881471 dB. Replay, independent aggregation, exact CLI export, unchanged pipeline and unmatched-budget rejection passed. [Complete report](../../reports/baseline_diagnostic.md).
- Next gated research tasks are matched A/C at the same useful 5,000-update B recipe, then seed/prior-art assessment and separately frozen protected evaluation. Exact additional configs are prepared, unapproved and unexecuted; no automatic expansion. The user's requested shutdown follows saved/verified current-work delivery, not an invented claim that scientific novelty is complete.

- 2026-10-21 cloud preference: move remaining long training to Colab Free. New preparation milestone: private code/train32/dev8 bundle -> integrity/environment checks -> short QA -> actual CUDA/FP16 profile -> reviewed bounded A/B/C plan -> approval -> matched cloud training -> dev replay/report. Dependencies/acceptance and recovery contracts are in [cloud runbook](../cloud_runbook.md). The former local A/C-only proposal is historical; all three CUDA arms must share one environment. No new long local run, cloud allocation or cloud result is implied by preparation. Cloud helper tests stay under notebooks to preserve frozen research-source fingerprints.

- 2026-10-22 Colab setup: user reports verified 32/8 images and allocated Tesla T4/14.56 GiB, Python3.13.15/torch2.11.0+cu130. Core QA command completed, then cloud helper tests failed. Clean v1 ZIP reproduction identified two missing-`reports/` fixture errors. Creating the directory passes all seven original archive helper tests; updated fixture tests also pass without that directory. Corrected notebook initializes it and exposes subprocess output. Three new notebook regression checks plus fresh 47-test core acceptance/smoke/syntax/two-update CPU dry-run/pip check pass. All 61 frozen training-source hashes remain unchanged. User rerun and real FP16 profile remain pending; no long training started. [Evidence](../../reports/colab_setup_fix_v1.json).

- Subsequent operator outputs confirm all seven cloud helper checks, CPU smoke, and two real CUDA FP32 updates/checkpoint reload passed. Actual full-model FP16 profile completed 10+50 updates at 1.267546 s/update, 0.322014 GiB peak allocation/0.375 GiB reserved, zero skips. Update-only 5k projection: 105.63 minutes per arm, 5.28 hours for all three. Prepared/strictly validated matched 5k A/B/C proposal with revised 9,000-second caps per arm/27,000 combined, $0; old one-hour caps are infeasible. [Proposal](../../reports/proposals/colab_matched5000_v1/proposal.json) awaits specific long-run approval. No implementation/source change or local/cloud long research training was performed during planning.

- 2026-10-22: direct user approval “yeah approve cloud study” authorizes the concrete matched CUDA A/B/C 5k study with 9,000 seconds per arm/27,000 combined, $0, no downloads or protected scoring. [Approval](../../reports/proposals/colab_matched5000_v1/approval.json). Prepared two Colab operator cells to seal profile/source/environment, persist reviewed plan/approval to Drive and launch A only. [Local validation](../../reports/proposals/colab_matched5000_v1/launch_validation.json) passed syntax/config/fresh-bundle preparation and mock launch dispatch checks; no actual CUDA/local research training was run by this validation. Operator launch/status remains pending; stop on failed/incomplete arm before any next arm.

- Operator now reports A running with verified-backup messages through step2,700/microstep10,800 (54%). Explicit AFK request authorizes queuing approved B/C automatically after A. Prepared a separate fail-closed continuation cell; A is not interrupted and no frozen source/helper is changed. Four offline coordinator tests pass, including predecessor failure, incomplete B, duplicate and approval guards; fresh 47-test acceptance and required smoke/syntax/CPU dry-run/pip checks pass. [Record](../../reports/proposals/colab_matched5000_v1/afk_continuation_record.json), [validation](../../reports/proposals/colab_matched5000_v1/continuation_validation.json). UI queuing and subsequent completion remain unconfirmed. Abrupt quota termination cannot force a final checkpoint; only complete prior Drive snapshots are recovery points.

- Colab operator subsequently reports A completed all5,000 updates, 40,000 clean crops/20,000 microsteps, zero skipped updates, 25 clipped updates, 6,676.1 seconds including setup/backups (111.27 minutes), with final step5,000 snapshot saved to Drive. Selected best_step=4,700; the pasted summary's development scores are from final step5,000 and must not be labeled selected-checkpoint results. Final-step worst-phase PSNR30.065720 versus input29.209203 (+0.856517 dB), eight dev images/16 phases; no test score. [Evidence](../../reports/proposals/colab_matched5000_v1/A_completion_operator_report.json). Independent selected-checkpoint hash/replay and B/C start/completion remain pending; no novelty claim.

- Later operator output confirmed B start/backups through3,300; the subsequent screenshot shows the guarded continuation started C and saved C through3,700/microstep14,800 before Colab refused GPU reconnection due to usage limits. B success is inferred from the guard, but final B summary/ZIP and latest C ZIP require direct verification. [Interruption record](../../reports/proposals/colab_matched5000_v1/gpu_quota_interruption_record.json). Preserve A/B; recover only C from a verified complete snapshot under matching source/environment and original5k horizon once free compatible CUDA is available. About1,300 updates remain if step3,700 is the latest valid backup; do not infer a quota-reset time or guarantee a final-save hook. No paid/CPU fallback, checkpoint deletion, quota bypass or unapproved new resume invocation was launched.


## 2026-10-23 Cloud study and independent development replay

All three 5,000-update CUDA arms are complete, including approved C recovery. Operator-pasted CPU replay confirms selected development scores to displayed precision on eight images and 16 Q20 phases. All three paired image-bootstrap intervals include zero. [Current cloud development results](../../reports/cloud_development_results.md) supersede older pending-C status; full artifact transfer, freeze and protected test remain pending. No new training or test scoring was performed in this documentation update.


## 2026-10-25 Imported cloud artifacts audited and test selections frozen

The 43.61 MiB cloud export matched SHA256 55795ddf1a1b3dc44903db8c137fe29cf93ba2914e5886d7c0b90ee031aab834. All three final snapshots and selected-model identities passed verification. Local reaggregation reproduces all per-image phase metrics and paired image-bootstrap intervals. All 61 frozen core source files remain unchanged. [Audit](../../reports/studies/cuda_jpeg_5000_v1/import_audit.json).

Frozen A4700/B4400/C4900 plus existing test IDs0801–0816; no protected scores calculated. Built and CRC/SHA checked a 67.20 MiB private test-only transfer. Prepared a CPU-only nine-job Q20/Q10/Q40 phase-evaluation wrapper, default denied without approval, with 600-second caps per job and Drive saves after each success. Four offline safeguard tests passed; actual Colab final scoring remains unexecuted. Submission reports updated to completed C and honest negative/uncertain development findings.
